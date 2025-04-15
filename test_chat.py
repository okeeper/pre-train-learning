from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_and_tokenizer(model_path):
    """加载模型和分词器"""
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=2048, temperature=0.7):
    """生成回复"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # 替换为你的模型路径
    model_path = "path/to/your/finetuned/model"
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    print("模型加载完成！开始对话（输入 'quit' 退出）：")
    
    # 开始对话循环
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'quit':
            break
            
        # 构建提示词（根据你的训练格式调整）
        prompt = f"用户：{user_input}\n助手："
        
        try:
            response = generate_response(model, tokenizer, prompt)
            # 提取助手的回复部分（根据实际输出格式调整）
            assistant_response = response.split("助手：")[-1].strip()
            print(f"\n助手: {assistant_response}")
            
        except Exception as e:
            print(f"生成回复时发生错误: {str(e)}")

if __name__ == "__main__":
    main() 