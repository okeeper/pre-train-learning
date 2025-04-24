import os
import json
import glob
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    QWenModel, 
    QWenConfig, 
    QWenTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
import logging
import argparse
from datasets import load_dataset
import wandb  # 导入wandb包

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="使用小说数据预训练Qwen模型")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="要预训练的Qwen模型路径或名称",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="包含xd_chunks_*.json文件的数据目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="保存模型和日志的输出目录",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="每个GPU的训练批次大小",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="初始学习率",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="权重衰减",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="训练轮次数",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,  # 修改默认值为-1，表示不使用max_steps
        help="训练的最大步数，设为-1时使用num_train_epochs参数",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="日志记录步数",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="保存模型的步数",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="最大序列长度",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="是否使用混合精度训练",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed配置文件路径",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="xd_chunks_*.json",
        help="用于匹配数据文件的通配符模式",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="用于分布式训练的本地排名",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true", 
        default=True,
        help="启用梯度检查点以节省内存",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=True,
        help="是否使用Weights & Biases进行实验跟踪",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Weights & Biases项目名称",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases运行名称",
    )
    parser.add_argument(
        "--wandb_watch",
        type=str,
        default="gradients",
        choices=["all", "gradients", "parameters", "False"],
        help="wandb的watch级别 (default: gradients)",
    )
    args = parser.parse_args()
    return args

# 自定义数据集类，用于加载多个JSON文件
class NovelChunksDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: QWenTokenizer, max_seq_length: int, file_pattern: str = "xd_chunks_*.json"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []
        
        # 获取所有符合文件模式的JSON文件
        json_files = glob.glob(os.path.join(data_dir, file_pattern))
        logger.info(f"找到{len(json_files)}个数据文件: {json_files}")
        
        # 加载所有文件的数据
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if "text" in item:
                            self.examples.append(item["text"])
                        elif isinstance(item, str):
                            self.examples.append(item)
                else:
                    logger.warning(f"文件{json_file}格式不符合预期")
        
        logger.info(f"总共加载了{len(self.examples)}个文本片段")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # 对文本进行编码
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # 需要将张量从形状[1, seq_len]转换为[seq_len]
        item = {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
        }
        
        # 为语言模型训练设置标签
        item["labels"] = item["input_ids"].clone()
        
        return item

def setup_parallel_training(args):
    """配置并行训练环境"""
    num_gpus = torch.cuda.device_count()
    logger.info(f"检测到 {num_gpus} 个GPU设备")
    
    if num_gpus > 1:
        logger.info(f"将使用所有 {num_gpus} 个GPU进行训练")
        # 如果未指定deepspeed但有多个GPU可用，自动生成配置
        if args.deepspeed is None:
            logger.info("自动生成DeepSpeed配置")
            ds_config = {
                "fp16": {"enabled": args.fp16},
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": args.learning_rate,
                        "weight_decay": args.weight_decay
                    }
                },
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {"device": "cpu", "pin_memory": True},
                    "contiguous_gradients": True,
                    "overlap_comm": True
                },
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "train_batch_size": args.per_device_train_batch_size * num_gpus * args.gradient_accumulation_steps,
                "train_micro_batch_size_per_gpu": args.per_device_train_batch_size
            }
            
            # 保存生成的配置
            ds_config_path = os.path.join(args.output_dir, "ds_auto_config.json")
            os.makedirs(args.output_dir, exist_ok=True)
            with open(ds_config_path, 'w') as f:
                json.dump(ds_config, f, indent=2)
            
            args.deepspeed = ds_config_path
            logger.info(f"自动生成的DeepSpeed配置已保存到：{ds_config_path}")
    
    return args

def setup_wandb(args):
    """设置Weights & Biases记录"""
    if args.use_wandb:
        logger.info("初始化Weights & Biases")
        run_name = args.wandb_name if args.wandb_name else f"qwen-pretrain-{args.model_name_or_path.split('/')[-1]}"
        
        # 初始化wandb
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
        
        # 记录环境信息
        wandb.config.update({
            "gpu_count": torch.cuda.device_count(),
            "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        })
        
        logger.info(f"Weights & Biases已初始化: {wandb.run.name}")
        return True
    return False

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 配置多GPU训练
    args = setup_parallel_training(args)
    
    # 初始化wandb
    using_wandb = setup_wandb(args)
    
    # 检查是否有上一次训练的检查点
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"找到检查点: {last_checkpoint}，将从此处恢复训练")
    
    # 加载Qwen模型和分词器
    logger.info(f"加载模型: {args.model_name_or_path}")
    
    model_config = QWenConfig.from_pretrained(args.model_name_or_path)
    
    # 启用梯度检查点以节省GPU内存
    if args.gradient_checkpointing:
        model_config.use_cache = False
        logger.info("启用梯度检查点以节省GPU内存")
    
    tokenizer = QWenTokenizer.from_pretrained(args.model_name_or_path)
    model = QWenModel.from_pretrained(
        args.model_name_or_path, 
        config=model_config
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # 设置tokenizer和模型用于预训练
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 加载训练数据
    logger.info(f"加载训练数据，使用文件模式: {args.file_pattern}")
    train_dataset = NovelChunksDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        file_pattern=args.file_pattern
    )
    
    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 使用自回归语言建模而非掩码语言建模
    )
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True if last_checkpoint is None else False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs if args.max_steps <= 0 else None,  # 添加轮次参数
        max_steps=args.max_steps if args.max_steps > 0 else None,  # 如果max_steps大于0，使用它
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=args.fp16,
        deepspeed=args.deepspeed,
        save_total_limit=3,  # 仅保存最后3个检查点
        remove_unused_columns=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        dataloader_num_workers=4,
        report_to=["wandb"] if args.use_wandb else None,
        local_rank=args.local_rank,
        ddp_find_unused_parameters=False,
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # 如果启用wandb并设置了watch，监控模型
    if args.use_wandb and args.wandb_watch != "False":
        wandb.watch(
            model,
            log=args.wandb_watch,
            log_freq=args.logging_steps
        )
    
    # 开始训练
    logger.info("开始训练")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 保存最终模型
    logger.info("保存最终模型")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 如果启用了wandb模型记录
    if args.use_wandb:
        logger.info("将模型上传到Weights & Biases")
        artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
        artifact.add_dir(args.output_dir)
        wandb.log_artifact(artifact)
    
    # 如果使用wandb，完成运行
    if args.use_wandb:
        wandb.finish()
    
    logger.info("预训练完成!")

if __name__ == "__main__":
    main()
