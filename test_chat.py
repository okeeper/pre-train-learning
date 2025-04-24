from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from peft import PeftModel
import os
from collections import deque

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

def generate_response(model, tokenizer, prompt, history=None, max_new_tokens=1024, temperature=0.7):
    """生成回复，可包含历史对话"""
    # 构建包含历史记录的完整提示
    full_prompt = prompt
    if history:
        history_text = ""
        for h_user, h_assistant in history:
            history_text += f"用户：{h_user}\n助手：{h_assistant}\n"
        full_prompt = history_text + prompt
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

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