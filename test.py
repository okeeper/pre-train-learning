from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from peft import PeftModel
import os
from collections import deque
import inspect

if __name__ == "__main__":
    print("开始运行测试程序...")
    parser = argparse.ArgumentParser(description="运行大语言模型聊天")
    parser.add_argument("--model_path", type=str, required=True, default="Qwen/Qwen2.5-1.5B-Instruct", help="模型路径")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA权重路径")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="最大生成令牌数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    parser.add_argument("--history_length", type=int, default=3, help="保存的对话历史轮数")
    args = parser.parse_args()
    print(f"开始运行测试程序...{args}")