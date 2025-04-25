#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用标准DataCollatorForSeq2Seq微调Qwen模型的SFT脚本
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,  # 使用标准数据整理器
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# 设置日志格式
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# SFT数据集类
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length=2048):
        """
        初始化SFT数据集
        
        Args:
            data_path: 数据文件或目录路径
            tokenizer: 用于编码文本的tokenizer
            max_seq_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []
        
        # 加载数据
        if os.path.isdir(data_path):
            # 如果是目录，则加载目录下所有json文件
            files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.json')]
        else:
            # 如果是单个文件
            files = [data_path]
        
        # 逐个文件加载
        for file_path in files:
            self._load_data_file(file_path)
        
        logger.info(f"加载了 {len(self.examples)} 个样本，来自 {len(files)} 个文件")
    
    def _load_data_file(self, file_path):
        """加载单个数据文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 根据文件格式进行处理
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "instruction" in item and "response" in item:
                            self.examples.append(item)
        except Exception as e:
            logger.warning(f"加载文件 {file_path} 时出错: {e}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """获取处理后的单个样本"""
        example = self.examples[idx]
        
        # 提取指令和回答
        instruction = example.get("instruction", "")
        response = example.get("response", "")
        system = example.get("system", "你是一个有用的助手。")
        
        # Qwen模型使用的格式化消息 - ChatML格式
        conversation = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        target = f"{response}<|im_end|>"
        
        # 编码输入部分
        prompt_ids = self.tokenizer.encode(conversation, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        
        # 检查长度并截断
        max_length = self.max_seq_length
        total_length = len(prompt_ids) + len(target_ids)
        
        if total_length > max_length:
            # 优先保留目标回答，如果太长则截断指令
            keep_target_len = min(len(target_ids), max_length // 2)
            keep_prompt_len = max_length - keep_target_len
            
            if keep_prompt_len < len(prompt_ids):
                prompt_ids = prompt_ids[-keep_prompt_len:]
            target_ids = target_ids[:keep_target_len]
        
        # 创建标签 - 只为目标部分计算损失
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        
        # 创建注意力掩码
        attention_mask = [1] * len(input_ids)
        
        # 确保不超过最大长度
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        attention_mask = attention_mask[:max_length]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# 自定义日志回调
class SimpleLoggingCallback:
    """简化版的日志记录回调，只在主进程中记录训练进度"""
    
    def __init__(self, log_interval=10, use_wandb=False):
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.global_step = 0
        self.logging_loss = 0.0
        self.best_loss = float('inf')
        self.start_time = datetime.now()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """在每次日志记录时调用"""
        if not logs:
            return
        
        # 检查是否为主进程
        is_main_process = args.local_rank in [-1, 0]
        if not is_main_process:
            return
        
        # 更新步数和损失
        self.global_step += 1
        
        # 每log_interval步输出一次日志
        if self.global_step % self.log_interval == 0:
            # 计算训练速度
            elapsed_time = datetime.now() - self.start_time
            steps_per_second = self.global_step / elapsed_time.total_seconds()
            
            # 获取关键指标
            loss = logs.get("loss", 0)
            learning_rate = logs.get("learning_rate", 0)
            
            # 更新最佳损失
            if loss < self.best_loss:
                self.best_loss = loss
            
            # 格式化输出
            log_message = (
                f"步骤: {self.global_step} | "
                f"损失: {loss:.4f} | "
                f"最佳损失: {self.best_loss:.4f} | "
                f"学习率: {learning_rate:.7f} | "
                f"速度: {steps_per_second:.2f} 步/秒 | "
                f"用时: {str(elapsed_time).split('.')[0]}"
            )
            
            print(log_message)
            
            # 上报wandb (如果启用)
            if self.use_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "train/loss": loss,
                            "train/learning_rate": learning_rate,
                            "train/steps_per_second": steps_per_second,
                            "train/best_loss": self.best_loss
                        })
                except ImportError:
                    pass

def setup_wandb(args, is_main_process):
    """设置Weights & Biases日志记录"""
    if not args.use_wandb or not is_main_process:
        return False
    
    try:
        import wandb
        
        # 如果非主进程，禁用wandb
        if not is_main_process:
            os.environ["WANDB_MODE"] = "disabled"
            return False
        
        # 设置运行名称
        run_name = args.run_name if args.run_name else f"sft-qwen-{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # 初始化wandb
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=args.wandb_group,
            config=vars(args)
        )
        
        return True
    except ImportError:
        logger.warning("Weights & Biases 未安装。使用 `pip install wandb` 安装它以启用日志记录。")
        return False

def get_model_tokenizer(args):
    """加载模型和分词器"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"  # SFT通常使用右侧填充
    )
    
    # 确保有padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型配置
    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    
    # 设置设备映射
    device_map = "auto"
    if args.no_cuda:
        device_map = {"": "cpu"}
    
    # 量化设置
    quantization_config = None
    if args.quantization:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if args.fp16 and not args.no_cuda else None,
    )
    
    # 应用LoRA (如果启用)
    if args.use_lora:
        # 如果使用量化，需要准备模型
        if args.quantization:
            model = prepare_model_for_kbit_training(model)
        
        # 设置LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target.split(",") if args.lora_target else None
        )
        
        # 应用LoRA配置
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    """主函数"""
    # 参数解析
    parser = argparse.ArgumentParser(description="Qwen模型的监督微调(SFT)")
    
    # 模型和数据参数
    parser.add_argument("--model_name_or_path", type=str, required=True, help="模型路径或huggingface模型名称")
    parser.add_argument("--data_path", type=str, required=True, help="SFT数据文件或目录路径")
    parser.add_argument("--output_dir", type=str, default="./sft_output", help="输出目录")
    
    # 训练参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="学习率")
    parser.add_argument("--max_steps", type=int, default=1000, help="最大训练步数")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--fp16", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_cuda", action="store_true", help="禁用CUDA")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地排名")
    
    # LoRA参数
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA丢弃率")
    parser.add_argument("--lora_target", type=str, default=None, 
                        help="LoRA目标模块,逗号分隔,例如:'q_proj,v_proj,k_proj,o_proj'")
    
    # 量化参数
    parser.add_argument("--quantization", action="store_true", help="是否使用4bit量化")
    
    # 日志参数
    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    parser.add_argument("--save_steps", type=int, default=500, help="保存检查点的步数")
    
    # Wandb参数
    parser.add_argument("--use_wandb", action="store_true", help="是否使用Weights & Biases记录训练")
    parser.add_argument("--wandb_project", type=str, default="qwen-sft", help="W&B项目名称")
    parser.add_argument("--wandb_group", type=str, default=None, help="W&B运行组")
    parser.add_argument("--run_name", type=str, default=None, help="训练运行的名称")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化分布式训练
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
    is_main_process = args.local_rank in [-1, 0]
    
    # 设置wandb
    use_wandb = setup_wandb(args, is_main_process)
    args.use_wandb = use_wandb
    
    # 加载模型和分词器
    model, tokenizer = get_model_tokenizer(args)
    
    # 加载数据集
    train_dataset = SFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    
    # 使用标准的DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,  # 可选，会用于处理decoder_input_ids
        padding=True,
        max_length=args.max_seq_length,
        pad_to_multiple_of=8 if args.fp16 else None,  # 对于混合精度训练很有用
    )
    
    # 检查上次检查点
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"找到上次检查点: {last_checkpoint}")
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=1,  # 设为1允许自定义回调处理所有日志
        save_steps=args.save_steps,
        no_cuda=args.no_cuda,
        fp16=args.fp16,
        local_rank=args.local_rank,
        remove_unused_columns=False,  # 保留所有列以确保自定义数据集正常工作
        seed=args.seed,
        report_to="none",  # 禁用默认的报告，使用自定义回调
        save_strategy="steps",
        save_total_limit=2,  # 只保留最新的两个检查点
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,  # 使用标准的DataCollatorForSeq2Seq
        callbacks=[SimpleLoggingCallback(log_interval=args.log_interval, use_wandb=use_wandb)]
    )
    
    # 从检查点恢复
    if last_checkpoint is not None:
        logger.info(f"从检查点恢复训练: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("开始新的训练")
        trainer.train()
    
    # 保存模型和分词器
    if is_main_process:
        logger.info("训练完成，保存模型...")
        model_to_save = model.module if hasattr(model, "module") else model
        
        # 只保存LoRA权重(如果启用LoRA)
        if args.use_lora:
            model_to_save.save_pretrained(args.output_dir)
            logger.info(f"LoRA权重已保存到 {args.output_dir}")
        else:
            # 保存完整模型
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logger.info(f"完整模型已保存到 {args.output_dir}")
        
        # 上传到wandb (如果启用)
        if use_wandb:
            try:
                import wandb
                # 只上传配置文件和元数据，不上传模型权重
                wandb.save(os.path.join(args.output_dir, "config.json"))
                
                # 记录训练总结
                wandb.run.summary.update({
                    "training_completed": True,
                    "total_steps": trainer.state.global_step,
                    "final_loss": trainer.state.log_history[-1].get("loss", None),
                    "model_size": sum(p.numel() for p in model.parameters()) / 1e6,  # 百万参数
                    "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
                })
                wandb.finish()
                logger.info("已上传训练总结到Weights & Biases")
            except Exception as e:
                logger.warning(f"上传到wandb失败: {e}")

if __name__ == "__main__":
    main()