import os
import json
import glob
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed
)
import logging
import argparse
import wandb
from datetime import datetime
from transformers.trainer_callback import TrainerCallback
import sys
import signal
import tqdm  # 修复：添加 tqdm 导入

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 根据你的日志，检测到2个GPU
os.environ["DS_SKIP_CUDA_CHECK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["MALLOC_CONF"] = "background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs/sft_training.log"))
    ]
)
logger = logging.getLogger(__name__)

# 重定向 tqdm 输出到日志
class TqdmToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.last_msg = ""

    def write(self, buf):
        if buf.strip() and buf.strip() != self.last_msg:
            self.last_msg = buf.strip()
            self.logger.log(self.level, buf.strip())
            
    def flush(self):
        pass

tqdm.std.sys.stdout = TqdmToLogger(logger)

args = None

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="使用小说数据进行Qwen模型监督微调")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/data/hf-models/Qwen3-8B",
        help="要微调的Qwen模型路径或名称",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="包含SFT数据的目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/sft/xd_sft",
        help="保存模型和日志的输出目录",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="每个GPU的训练批次大小",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="每个GPU的评估批次大小",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,  # 提高到1e-4，避免梯度过小
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
        default=1.5,
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
        default=1,
        help="日志记录步数",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="保存模型的步数",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="评估模型的步数",
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
        default=False,  # 禁用FP16以确保稳定性
        help="是否使用混合精度训练",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="sft_data_*.json",
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
        default="qwen-sft",
        help="Weights & Biases项目名称",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="xd_sft",
        help="Weights & Biases运行名称",
    )
    parser.add_argument(
        "--wandb_watch",
        type=str,
        default="parameters",
        choices=["all", "gradients", "parameters", "False"],
        help="wandb的watch级别",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",  # 使用adamw_torch以兼容旧版transformers
        choices=["adamw_torch", "adamw_8bit"],
        help="优化器类型",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="分布式训练的本地排名",
    )
    parser.add_argument(
        "--eval_split_ratio",
        type=float,
        default=0.1,
        help="评估数据集的划分比例",
    )
    parser.add_argument(
        "--shuffle_before_split",
        action="store_true",
        help="在划分训练/评估数据集前是否打乱数据",
    )

    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed 配置文件路径",
    )
    args = parser.parse_args()
    return args 

# 自定义数据集类
class SFTDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: AutoTokenizer, max_seq_length: int, file_pattern: str):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []
        
        patterns = [p.strip() for p in file_pattern.split(',')]
        json_files = []
        
        for pattern in patterns:
            if os.path.isfile(os.path.join(data_dir, pattern)):
                json_files.append(os.path.join(data_dir, pattern))
            else:
                matched_files = glob.glob(os.path.join(data_dir, pattern))
                json_files.extend(matched_files)
        
        json_files = list(set(json_files))
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(f"找到{len(json_files)}个数据文件: {json_files}")
        
        for json_file in json_files:
            try:
                if json_file.endswith('.jsonl'):
                    logger.info(f"加载JSONL文件: {json_file}")
                    with open(json_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                item = json.loads(line.strip())
                                self._process_item(item)
                            except json.JSONDecodeError as je:
                                logger.warning(f"解析JSONL行时出错: {str(je)}")
                else:
                    logger.info(f"加载JSON文件: {json_file}")
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                self._process_item(item)
                        elif isinstance(data, dict):
                            self._process_item(data)
                        else:
                            logger.warning(f"文件{json_file}格式不符合预期: {type(data)}")
            except Exception as e:
                logger.error(f"处理文件{json_file}时出错: {str(e)}")
        
        logger.info(f"总共加载了{len(self.examples)}个SFT样本")

    def _process_item(self, item):
        if isinstance(item, dict):
            instruction = None
            response = None
            
            if "instruction" in item and "output" in item:
                instruction = item["instruction"]
                response = item["output"]
                if "input" in item and item["input"]:
                    instruction = f"{instruction}\n{item['input']}"
            elif "conversations" in item and isinstance(item["conversations"], list):
                if len(item["conversations"]) >= 2:
                    instruction = item["conversations"][0].get("value", "")
                    response = item["conversations"][1].get("value", "")
            elif "instruction" in item and "response" in item:
                instruction = item["instruction"]
                response = item["response"]
            elif "question" in item and "answer" in item:
                instruction = item["question"]
                response = item["answer"]
            elif "prompt" in item and "completion" in item:
                instruction = item["prompt"]
                response = item["completion"]
            
            if instruction and response:
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]
                formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                
                if len(self.examples) < 5:
                    logger.info(f"Formatted text sample: {formatted_text[:100]}...")
                
                self.examples.append({"text": formatted_text})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]["text"]

        if idx < 5:
            logger.info(f"Sample {idx} text: {text[:100]}...")

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        if idx < 5:
            logger.info(f"Sample {idx} input_ids: {input_ids[:20]}...")
            logger.info(f"Sample {idx} attention_mask: {attention_mask[:20]}...")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

# 自定义数据整理器
class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels_batch = self.tokenizer.pad(
            {"input_ids": labels},
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"]

        labels_batch = labels_batch.masked_fill(labels_batch == self.tokenizer.pad_token_id, -100)

        batch["labels"] = labels_batch
        
        return batch


def split_dataset(dataset, eval_ratio=0.1, seed=42, shuffle=True):
    if eval_ratio <= 0 or eval_ratio >= 1:
        return dataset, None
    
    dataset_size = len(dataset)
    eval_size = int(dataset_size * eval_ratio)
    
    indices = list(range(dataset_size))
    if shuffle:
        rng = torch.Generator()
        rng.manual_seed(seed)
        torch.randperm(dataset_size, generator=rng, out=torch.tensor(indices, dtype=torch.long))
    
    train_indices = indices[eval_size:]
    eval_indices = indices[:eval_size]
    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)
    
    return train_dataset, eval_dataset 

def print_training_config(args, model_config, train_dataset, eval_dataset, is_distributed):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_count = torch.cuda.device_count()
    params_count = ""
    if hasattr(model_config, "num_hidden_layers") and hasattr(model_config, "hidden_size"):
        hidden_size = model_config.hidden_size
        n_layers = model_config.num_hidden_layers
        if hasattr(model_config, "vocab_size"):
            vocab_size = model_config.vocab_size
            emb_params = vocab_size * hidden_size
            layer_params = 12 * hidden_size * hidden_size * n_layers
            estimated_params = (emb_params + layer_params) / 1_000_000
            params_count = f"约 {estimated_params:.1f}M 参数"
    
    separator = "-" * 80
    effective_batch_size = args.per_device_train_batch_size * gpu_count * args.gradient_accumulation_steps

    logger.info("\n\n")
    logger.info(separator)
    logger.info("Qwen SFT微调配置")
    logger.info(separator)
    logger.info(f"开始时间:\t{now}")
    logger.info(f"模型名称:\t{args.model_name_or_path}")
    if params_count:
        logger.info(f"模型规模:\t{params_count}")
    logger.info("\n模型架构:")
    logger.info(f"\t隐藏层数:\t{model_config.num_hidden_layers}")
    logger.info(f"\t隐藏维度:\t{model_config.hidden_size}")
    if hasattr(model_config, "num_attention_heads"):
        logger.info(f"\t注意力头数:\t{model_config.num_attention_heads}")
    if hasattr(model_config, "vocab_size"):
        logger.info(f"\t词表大小:\t{model_config.vocab_size}")
    logger.info("\n训练数据:")
    logger.info(f"\t数据目录:\t{args.data_dir}")
    logger.info(f"\t文件模式:\t{args.file_pattern}")
    logger.info(f"\t训练样本数:\t{len(train_dataset):,} 个样本")
    if eval_dataset:
        logger.info(f"\t评估样本数:\t{len(eval_dataset):,} 个样本")
    logger.info(f"\t最大序列长度:\t{args.max_seq_length or '未指定'}")
    logger.info("\n训练设置:")
    logger.info(f"\tGPU数量:\t{gpu_count} 个")
    logger.info(f"\t每设备训练批次大小:\t{args.per_device_train_batch_size}")
    if eval_dataset:
        logger.info(f"\t每设备评估批次大小:\t{args.per_device_eval_batch_size}")
    logger.info(f"\t梯度累积步数:\t{args.gradient_accumulation_steps}")
    logger.info(f"\t有效总批次大小:\t{effective_batch_size}")
    if args.max_steps > 0:
        logger.info(f"\t训练步数:\t{args.max_steps:,}")
        total_samples = args.max_steps * effective_batch_size
        epochs_equiv = args.max_steps * effective_batch_size / len(train_dataset)
        logger.info(f"\t预计训练样本数:\t{total_samples:,} (约 {epochs_equiv:.2f} 轮)")
    else:
        logger.info(f"\t训练轮次:\t{args.num_train_epochs:.1f} 轮")
        estimated_steps = int(len(train_dataset) * args.num_train_epochs / effective_batch_size)
        logger.info(f"\t预计总步数:\t{estimated_steps:,}")
    logger.info("\n优化器设置:")
    logger.info(f"\t学习率:\t{args.learning_rate:.1e}")
    logger.info(f"\t权重衰减:\t{args.weight_decay}")
    logger.info("\n加速技术:")
    logger.info(f"\tFP16混合精度:\t{'启用' if args.fp16 else '禁用'}")
    logger.info(f"\t梯度检查点:\t{'启用' if args.gradient_checkpointing else '禁用'}")
    logger.info("\n保存与监控:")
    logger.info(f"\t输出目录:\t{args.output_dir}")
    logger.info(f"\t日志步数:\t{args.logging_steps}")
    logger.info(f"\t保存步数:\t{args.save_steps}")
    if eval_dataset:
        logger.info(f"\t评估步数:\t{args.eval_steps}")
    logger.info(f"\tWeights & Biases:\t{'启用' if args.use_wandb else '禁用'}")
    if args.use_wandb and wandb.run:
        logger.info(f"\tWandB项目:\t{args.wandb_project}")
        logger.info(f"\tWandB运行:\t{wandb.run.name}")
    logger.info("\n其他信息:")
    logger.info(f"\t随机种子:\t{args.seed}")
    tokens_per_step = effective_batch_size * args.max_seq_length
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = int(len(train_dataset) * args.num_train_epochs / effective_batch_size)
    tokens_per_second = 500
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
    logger.info(f"\t预计训练时长:\t{time_str}")
    logger.info(separator)
    logger.info("SFT微调已开始...")
    logger.info(separator)
    logger.info("\n")


def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    args = parse_args()
    set_seed(args.seed)
    
    is_distributed = False
    is_main_process = True
    
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        run_time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
        wandb_run_name = f"qwen-sft-{run_time_str}" if not args.wandb_name else args.wandb_name
        args.wandb_name = wandb_run_name
    
    if is_main_process and args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"
    
    logger.info(f"加载模型配置: {args.model_name_or_path}")
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    if args.gradient_checkpointing:
        model_config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.warning("分词器没有pad_token或eos_token，将使用默认填充标记")
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    logger.info(f"加载预训练模型: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        torch_dtype=torch.float32,
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    logger.info(f"加载SFT训练数据: {args.data_dir}")
    full_dataset = SFTDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        file_pattern=args.file_pattern
    )
    
    train_dataset, eval_dataset = split_dataset(
        full_dataset,
        eval_ratio=args.eval_split_ratio,
        seed=args.seed,
        shuffle=args.shuffle_before_split
    )
    
    if is_main_process:
        print_training_config(args, model_config, train_dataset, eval_dataset, is_distributed)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=-1 if args.num_train_epochs > 0 else args.max_steps,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs > 0 else None,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        fp16=args.fp16,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        report_to=["wandb"] if args.use_wandb and is_main_process else [],
        run_name=wandb.run.name if args.use_wandb and is_main_process else None,
        disable_tqdm=not is_main_process,
        local_rank=args.local_rank,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim=args.optim,
        adam_beta1=0.8,
        adam_beta2=0.99,
        save_total_limit=3,
        deepspeed=args.deepspeed,
    )
    
    logger.info(f"训练参数: logging_steps={training_args.logging_steps}, save_steps={training_args.save_steps}")
    
    data_collator = CustomDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        padding="max_length",
        max_length=args.max_seq_length
    )
    
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        if args.use_wandb and args.wandb_watch != "False" and is_main_process:
            wandb.watch(model, log=args.wandb_watch, log_freq=args.logging_steps)
        
        logger.info("开始SFT微调...")
        train_result = trainer.train()
        
        if is_main_process:
            logger.info(f"训练结果: {train_result}")
            metrics = train_result.metrics
            logger.info(f"训练指标: {metrics}")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        raise
    
    if is_main_process:
        logger.info("保存最终模型")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        if training_args.local_rank == -1:
            with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
        
        logger.info(f"保存最终模型成功: {args.output_dir}")
        if args.use_wandb:
            logger.info("记录训练指标到wandb")
            wandb.finish()
    
    logger.info("SFT微调完成!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"发生异常: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            logger.info("正在清理CUDA缓存...")
            torch.cuda.empty_cache()
        logger.info("程序退出，资源已清理")