import os
import json
import glob
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
import logging
import argparse
from datasets import load_dataset
import wandb
from datetime import datetime
from transformers.trainer_callback import TrainerCallback
import time
import sys

# 强制刷新所有标准输出
os.environ["PYTHONUNBUFFERED"] = "1"

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(stream=sys.stdout),  # 直接输出到标准输出
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "training.log"))  # 同时保存到文件
    ]
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
        default=8,
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
        default=1.0,
        help="训练轮次数",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
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
        default=1024,
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
        "--file_pattern",
        type=str,
        default="xd_chunks_*.json",
        help="用于匹配数据文件的通配符模式，可用逗号分隔多个模式",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true", 
        help="启用梯度检查点以节省内存",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="是否使用Weights & Biases进行实验跟踪",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen-pretraining",
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
        default="parameters",
        choices=["all", "gradients", "parameters", "False"],
        help="wandb的watch级别 (default: gradients)",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        dest="local_rank",  # 指向同一个目标变量
        default=-1,
        help=argparse.SUPPRESS,  # 在帮助信息中隐藏这个重复参数
    )
    args = parser.parse_args()
    return args

# 自定义数据集类，用于加载多个JSON文件
class NovelChunksDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: AutoTokenizer, max_seq_length: int, file_pattern: str = "xd_chunks_*.json"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []
        
        # 处理逗号分隔的模式
        patterns = [p.strip() for p in file_pattern.split(',')]
        json_files = []
        
        # 获取所有符合文件模式的JSON文件
        for pattern in patterns:
            matched_files = glob.glob(os.path.join(data_dir, pattern))
            json_files.extend(matched_files)
        
        # 去除可能的重复文件
        json_files = list(set(json_files))
        
        logger.info(f"找到{len(json_files)}个数据文件: {json_files}")
        
        # 加载所有文件的数据
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "content" in item:
                                self.examples.append(item["content"])
                            elif isinstance(item, dict) and "text" in item:
                                self.examples.append(item["text"])
                            elif isinstance(item, str):
                                self.examples.append(item)
                    elif isinstance(data, dict) and "content" in data:
                        self.examples.append(data["content"])
                    elif isinstance(data, dict) and "text" in data:
                        self.examples.append(data["text"])
                    else:
                        logger.warning(f"文件{json_file}格式不符合预期: {type(data)}")
            except Exception as e:
                logger.error(f"处理文件{json_file}时出错: {str(e)}")
        
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


def setup_wandb(args):
    """设置Weights & Biases记录"""
    if args.use_wandb:
        logger.info("初始化Weights & Biases")
        run_name = args.wandb_name if args.wandb_name else f"qwen-pretrain-{args.output_dir.split('/')[-1]}"
        
        # 添加这行来禁用wandb的进度条
        os.environ["WANDB_CONSOLE"] = "off"
        
        # 初始化wandb
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
        
        # 记录环境信息
        gpu_count = torch.cuda.device_count()
        wandb.config.update({
            "gpu_count": gpu_count,
            "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        })
        
        logger.info(f"Weights & Biases已初始化: {wandb.run.name}")
        return True
    return False

def print_training_config(args, model_config, train_dataset, effective_batch_size):
    """使用简单的制表符对齐方式打印训练配置"""
    from datetime import datetime
    
    # 获取当前时间
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 检测GPU数量
    gpu_count = torch.cuda.device_count()
    
    # 估算模型参数量
    params_count = ""
    if hasattr(model_config, "num_hidden_layers") and hasattr(model_config, "hidden_size"):
        hidden_size = model_config.hidden_size
        n_layers = model_config.num_hidden_layers
        if hasattr(model_config, "vocab_size"):
            vocab_size = model_config.vocab_size
            # 估算参数量（基于Transformer架构的粗略估计）
            emb_params = vocab_size * hidden_size  # 词嵌入参数
            layer_params = 12 * hidden_size * hidden_size * n_layers  # 每层参数
            estimated_params = (emb_params + layer_params) / 1_000_000
            params_count = f"约 {estimated_params:.1f}M 参数"
    
    # 使用简单的分隔线
    separator = "-" * 80
    
    # 打印基本信息
    print("\n\n")
    print(separator)
    print("Qwen 预训练配置")
    print(separator)
    
    # 基本信息部分
    print(f"开始时间:\t{now}")
    print(f"模型名称:\t{args.model_name_or_path}")
    if params_count:
        print(f"模型规模:\t{params_count}")
    
    # 模型架构部分
    print("\n模型架构:")
    print(f"\t隐藏层数:\t{model_config.num_hidden_layers}")
    print(f"\t隐藏维度:\t{model_config.hidden_size}")
    if hasattr(model_config, "num_attention_heads"):
        print(f"\t注意力头数:\t{model_config.num_attention_heads}")
    if hasattr(model_config, "vocab_size"):
        print(f"\t词表大小:\t{model_config.vocab_size}")
    
    # 训练数据部分
    print("\n训练数据:")
    print(f"\t数据目录:\t{args.data_dir}")
    print(f"\t文件模式:\t{args.file_pattern}")
    print(f"\t数据样本数:\t{len(train_dataset):,} 个样本")
    print(f"\t最大序列长度:\t{args.max_seq_length}")
    
    # 训练设置部分
    print("\n训练设置:")
    print(f"\tGPU数量:\t{gpu_count} 个")
    print(f"\t每设备批次大小:\t{args.per_device_train_batch_size}")
    print(f"\t梯度累积步数:\t{args.gradient_accumulation_steps}")
    print(f"\t有效总批次大小:\t{effective_batch_size}")
    
    if args.max_steps > 0:
        print(f"\t训练步数:\t{args.max_steps:,}")
        total_samples = args.max_steps * effective_batch_size
        epochs_equiv = args.max_steps * effective_batch_size / len(train_dataset)
        print(f"\t预计训练样本数:\t{total_samples:,} (约 {epochs_equiv:.2f} 轮)")
    else:
        print(f"\t训练轮次:\t{args.num_train_epochs:.1f} 轮")
        estimated_steps = int(len(train_dataset) * args.num_train_epochs / effective_batch_size)
        print(f"\t预计总步数:\t{estimated_steps:,}")
    
    # 优化器设置部分
    print("\n优化器设置:")
    print(f"\t学习率:\t{args.learning_rate:.1e}")
    print(f"\t权重衰减:\t{args.weight_decay}")
    
    # 加速技术部分
    print("\n加速技术:")
    print(f"\tFP16混合精度:\t{'启用' if args.fp16 else '禁用'}")
    print(f"\t梯度检查点:\t{'启用' if args.gradient_checkpointing else '禁用'}")
    
    # 保存与监控部分
    print("\n保存与监控:")
    print(f"\t输出目录:\t{args.output_dir}")
    print(f"\t日志步数:\t{args.logging_steps}")
    print(f"\t保存步数:\t{args.save_steps}")
    print(f"\tWeights & Biases:\t{'启用' if args.use_wandb else '禁用'}")
    if args.use_wandb and wandb.run:
        if args.wandb_project:
            print(f"\tWandB项目:\t{args.wandb_project}")
        else:
            print(f"\tWandB项目:\t{'未指定'}")
        print(f"\tWandB运行:\t{wandb.run.name}")
    
    # 其他信息部分
    print("\n其他信息:")
    print(f"\t随机种子:\t{args.seed}")
    
    # 预计的训练时间
    tokens_per_step = effective_batch_size * args.max_seq_length
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = int(len(train_dataset) * args.num_train_epochs / effective_batch_size)
    
    # 计算粗略的训练时间估计
    tokens_per_second = 500  # 单GPU粗略估计，实际取决于硬件
    if torch.cuda.is_available() and hasattr(model_config, 'hidden_size'):
        size_factor = model_config.hidden_size / 1024
        tokens_per_second = tokens_per_second / size_factor * gpu_count
        
    estimated_seconds = (tokens_per_step * total_steps) / tokens_per_second
    estimated_hours = estimated_seconds / 3600
    
    days = int(estimated_hours // 24)
    hours = int(estimated_hours % 24)
    minutes = int((estimated_hours * 60) % 60)
    
    time_str = ""
    if days > 0:
        time_str += f"{days}天 "
    time_str += f"{hours}小时 {minutes}分钟"
    
    print(f"\t预计训练时长:\t{time_str}")
    
    # 结束部分
    print(separator)
    print("训练已开始...")
    print(separator)
    print("\n")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 检查是否在分布式环境中
    is_distributed = args.local_rank != -1
    if is_distributed:
        # 先设置GPU设备，再初始化分布式环境
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            
        # 仅初始化一次分布式环境
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
            
        logger.info(f"使用分布式训练，rank={args.local_rank}，设备={torch.cuda.current_device()}")
    
    # 定义主进程变量，用于替代重复的条件判断
    is_main_process = not is_distributed or args.local_rank == 0
    
    # 创建输出目录(只在主进程)
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # 添加训练时间标记
        run_time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
        wandb_run_name = f"qwen-pretrain-{run_time_str}"
        args.wandb_name = args.wandb_name or wandb_run_name
    
    # 初始化wandb(只在主进程)
    if is_main_process and args.use_wandb:
        using_wandb = setup_wandb(args)
    
    
    # 加载模型配置和分词器
    logger.info(f"加载模型配置: {args.model_name_or_path}")
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    if args.gradient_checkpointing:
        model_config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 加载模型
    logger.info(f"加载预训练模型: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        config=model_config
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # 加载数据集(所有进程需要自己的副本)
    logger.info(f"加载训练数据: {args.data_dir}")
    train_dataset = NovelChunksDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        file_pattern=args.file_pattern
    )
    
    # 计算有效批次大小
    gpu_count = torch.cuda.device_count()
    world_size = torch.distributed.get_world_size() if is_distributed else 1
    effective_batch_size = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
    
    # 仅在主进程打印训练配置
    if is_main_process:
        print_training_config(args, model_config, train_dataset, effective_batch_size)
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=-1 if args.num_train_epochs > 0 else args.max_steps,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs > 0 else None,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=args.fp16,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to=["wandb"] if args.use_wandb and is_main_process else [],
        run_name=args.wandb_name,
        # 分布式训练参数
        local_rank=args.local_rank,
        ddp_find_unused_parameters=False,
        # 禁用tqdm进度条，使用简单日志
        disable_tqdm=is_distributed,  # 分布式时禁用tqdm
        # 日志设置
        logging_first_step=True,
        logging_nan_inf_filter=False,
        # 确保正确显示loss
        label_smoothing_factor=0.0,
    )
    
    logger.info(f"训练参数: logging_steps={training_args.logging_steps}, save_steps={training_args.save_steps}")
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 初始化Trainer - 移除了自定义回调
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # 监控模型(仅主进程)
    if args.use_wandb and args.wandb_watch != "False" and is_main_process:
        wandb.watch(model, log=args.wandb_watch, log_freq=args.logging_steps)
    
    # 开始训练
    logger.info("开始训练...")
    
    # 强制同步所有进程时指定设备
    if is_distributed:
        # 获取当前设备ID
        device_id = torch.cuda.current_device()
        # 明确指定设备进行barrier操作
        torch.distributed.barrier(device_ids=[device_id])
    
    train_result = trainer.train()
    
    # 输出训练结果
    if is_main_process:
        logger.info(f"训练结果: {train_result}")
        metrics = train_result.metrics
        logger.info(f"训练指标: {metrics}")
    
    # 保存模型(仅主进程)
    if is_main_process:
        logger.info("保存最终模型")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # 保存训练参数
        if training_args.local_rank == 0:
            with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
                import json
                json.dump(vars(args), f, indent=4)
        
        # wandb记录(仅主进程)
        if args.use_wandb:
            # 不上传模型，只记录训练完成
            logger.info("记录训练指标到wandb，不上传模型")
            wandb.finish()
    
    # 训练结束后
    # 确保分布式环境正确关闭
    if is_distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    logger.info("预训练完成!")

if __name__ == "__main__":
    main()
