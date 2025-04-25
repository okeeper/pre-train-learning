from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from peft import PeftModel
import os
from collections import deque
import inspect

def load_model_and_tokenizer(model_path, lora_path=None):
    """加载模型和分词器，可选加载LoRA权重"""
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 如果提供了LoRA路径，则加载LoRA权重
    if lora_path and os.path.exists(lora_path):
        print(f"正在加载LoRA权重: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, history=None, max_new_tokens=1024, temperature=0.7, system_prompt=None):
    """生成回复，使用Qwen格式的对话模板
    
    参数:
    - model: 已加载的模型
    - tokenizer: 对应的分词器
    - prompt: 用户当前的输入
    - history: 对话历史列表，格式为[(user_message, assistant_message), ...]
    - max_new_tokens: 最大生成的新token数量
    - temperature: 生成温度，控制随机性
    - system_prompt: 自定义系统提示，如果为None则使用默认系统提示
    
    返回:
    - 模型生成的回复文本
    """
    try:
        if history is None:
            history = []
        
        # 默认系统提示词，如果用户想要小说阅读专家，可以修改这里
        if system_prompt is None:
            system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in novel reading and literary analysis."
        
        # 构建完整prompt
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        # 添加历史对话
        for h_user, h_assistant in history:
            if h_user is not None:
                full_prompt += f"<|im_start|>user\n{h_user}<|im_end|>\n"
            if h_assistant is not None:
                full_prompt += f"<|im_start|>assistant\n{h_assistant}<|im_end|>\n"
        
        # 添加当前用户输入
        full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 生成回复
        inputs = tokenizer(full_prompt, return_tensors="pt")
        # 确保输入数据被发送到正确的设备
        for k, v in inputs.items():
            inputs[k] = v.to(model.device)
        
        # 设置生成参数
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": 0.7,
            "pad_token_id": tokenizer.pad_token_id,
        }
        
        # 检查模型API是否支持stop_words参数
        generate_signature = inspect.signature(model.generate)
        if "stopping_criteria" in generate_signature.parameters:
            from transformers import StoppingCriteriaList, StoppingCriteria
            
            # 自定义停止标准
            class StopOnTokens(StoppingCriteria):
                def __init__(self, stop_token_ids):
                    self.stop_token_ids = stop_token_ids
                
                def __call__(self, input_ids, scores, **kwargs):
                    for stop_id in self.stop_token_ids:
                        if input_ids[0][-1] == stop_id:
                            return True
                    return False
            
            # 获取终止词的token IDs
            stop_words_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_words_ids)])
            gen_kwargs["stopping_criteria"] = stopping_criteria
        
        # 生成回复
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # 解码输出
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取助手的回复部分
        assistant_start = "<|im_start|>assistant\n"
        assistant_end = "<|im_end|>"
        
        # 查找最后一个助手部分
        last_assistant_start = full_response.rfind(assistant_start)
        if last_assistant_start != -1:
            response_start = last_assistant_start + len(assistant_start)
            response_end = full_response.find(assistant_end, response_start)
            if response_end != -1:
                response = full_response[response_start:response_end].strip()
            else:
                # 如果没有找到结束标记，就取所有剩余文本
                response = full_response[response_start:].strip()
        else:
            # 回退到简单解码
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        return response
    except Exception as e:
        print(f"生成回复时发生错误: {str(e)}")
        return "抱歉，生成回复时发生错误。"

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="运行大语言模型聊天")
    parser.add_argument("--model_path", type=str, required=True, default="Qwen/Qwen2.5-1.5B-Instruct", help="模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA权重路径（可选）")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="生成的最大新token数量（默认：1024）")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成文本的温度参数（默认：0.7）")
    args = parser.parse_args()
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.lora_path)
    
    print("模型加载完成！开始对话（输入'quit'退出，输入'clear'清除历史记录）：")
    print(f"当前设置：max_new_tokens={args.max_new_tokens}, temperature={args.temperature}")
    
    # 使用deque保存最近3条对话历史
    history = deque(maxlen=3)
    
    # 开始对话循环
    while True:
        user_input = input("\n用户: ")
        
        # 处理特殊命令
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            history.clear()
            print("已清除历史对话记录")
            continue
            
        # 构建提示词
        prompt = f"用户：{user_input}\n助手："
        
        try:
            response = generate_response(
                model, 
                tokenizer, 
                prompt, 
                history=list(history),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            # 提取助手的回复部分
            assistant_response = response.split("助手：")[-1].strip()
            print(f"\n助手: {assistant_response}")
            
            # 更新历史记录
            history.append((user_input, assistant_response))
            
        except Exception as e:
            print(f"生成回复时发生错误: {str(e)}")

if __name__ == "__main__":
    main() 