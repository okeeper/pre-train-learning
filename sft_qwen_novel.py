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
from torch.distributed import gather_object
import signal
# 导入PEFT/LoRA相关库
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

# 设置可见的GPU 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# 设置环境变量跳过DeepSpeed的CUDA版本检查
os.environ["DS_SKIP_CUDA_CHECK"] = "1"
# 解决tokenizers警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 内存优化设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
# 内存碎片化处理
os.environ["MALLOC_CONF"] = "background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(stream=sys.stdout),  # 直接输出到标准输出
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs/sft_training.log"))  # 同时保存到文件
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

# 替换 tqdm 的文件描述符
import tqdm.std
tqdm.std.sys.stdout = TqdmToLogger(logger)

args = None

# 定义模板格式
DEFAULT_TEMPLATE = """<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="使用小说数据进行Qwen模型监督微调")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
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
        "--per_device_eval_batch_size", 
        type=int,
        default=1,
        help="每个GPU的评估批次大小",
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
        default=2e-5,
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
        "--eval_steps",
        type=int,
        default=500,
        help="评估模型的步数",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
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
        default=None,
        help="Weights & Biases运行名称",
    )
    parser.add_argument(
        "--wandb_watch",
        type=str,
        default="parameters",
        choices=["all", "gradients", "parameters", "False"],
        help="wandb的watch级别 (default: parameters)",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        choices=["adamw_8bit", "adamw_torch", "adamw_torch_fp16", "adamw_torch_fp16_distributed", "adamw_torch_fp16_distributed_zero2"],
        help="优化器类型 (default: adamw_torch)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", -1)),  # 支持两种启动方式
        help="分布式训练的本地排名",
    )
    # 添加DeepSpeed参数
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed配置文件路径",
    )
    # 添加LoRA相关参数
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="是否使用LoRA进行参数高效微调",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA秩，较小的值占用更少的内存，较大的值提供更多的容量",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA缩放系数",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout概率",
    )
    parser.add_argument(
        "--lora_target_modules", 
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="需要添加LoRA的目标模块，以逗号分隔",
    )
    # 添加量化参数
    parser.add_argument(
        "--quantization",
        action="store_true",
        help="是否使用4bit量化以减少内存使用",
    )
    # 添加SFT特有参数
    parser.add_argument(
        "--template",
        type=str,
        default=DEFAULT_TEMPLATE,
        help="对话模板格式",
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

    args = parser.parse_args()
    return args 

# 自定义数据集类，用于SFT微调
class SFTDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: AutoTokenizer, max_seq_length: int, file_pattern: str = "sft_data_*.json", template: str = DEFAULT_TEMPLATE):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.template = template
        self.examples = []
        
        # 处理逗号分隔的模式
        patterns = [p.strip() for p in file_pattern.split(',')]
        json_files = []
        
        # 获取所有符合文件模式的JSON文件
        for pattern in patterns:
            if os.path.isfile(pattern):
                # 支持直接传入文件路径
                json_files.append(pattern)
            else:
                matched_files = glob.glob(os.path.join(data_dir, pattern))
                json_files.extend(matched_files)
        
        # 去除可能的重复文件
        json_files = list(set(json_files))
        
        # 只在主进程打印信息
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(f"找到{len(json_files)}个数据文件: {json_files}")
        
        # 加载所有文件的数据
        for json_file in json_files:
            try:
                # 检查文件扩展名，处理不同格式
                if json_file.endswith('.jsonl'):
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        logger.info(f"加载JSONL文件: {json_file}")
                    # 处理JSONL格式
                    with open(json_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                item = json.loads(line.strip())
                                self._process_item(item)
                            except json.JSONDecodeError as je:
                                logger.warning(f"解析JSONL行时出错: {str(je)}")
                else:
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        logger.info(f"加载JSON文件: {json_file}")
                    # 处理JSON格式
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
        
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(f"总共加载了{len(self.examples)}个SFT样本")

    def _process_item(self, item):
        """处理单个数据项目，支持不同格式的SFT数据"""
        if isinstance(item, dict):
            # 尝试各种常见的指令微调数据格式
            instruction = None
            response = None
            
            # 处理Alpaca格式
            if "instruction" in item and "output" in item:
                instruction = item["instruction"]
                response = item["output"]
                if "input" in item and item["input"]:
                    instruction = f"{instruction}\n{item['input']}"
            
            # 处理ShareGPT格式
            elif "conversations" in item and isinstance(item["conversations"], list):
                # 从对话记录中提取指令和回复
                if len(item["conversations"]) >= 2:
                    instruction = item["conversations"][0].get("value", "")
                    response = item["conversations"][1].get("value", "")
            
            # 处理通用格式（instruction/response）
            elif "instruction" in item and "response" in item:
                instruction = item["instruction"]
                response = item["response"]
            
            # 处理问答类格式
            elif "question" in item and "answer" in item:
                instruction = item["question"]
                response = item["answer"]
            
            # 处理prompt/completion格式（如OpenAI格式）
            elif "prompt" in item and "completion" in item:
                instruction = item["prompt"]
                response = item["completion"]
            
            # 如果有效则添加到样本中
            if instruction and response:
                # 格式化模板
                formatted_text = self.template.format(instruction=instruction, response=response)
                self.examples.append({"text": formatted_text})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]["text"]
        
        if self.max_seq_length:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                text,
                truncation=True,
                return_tensors="pt",
            )

        # 处理标签 - 对于SFT，我们将不需要对应于instruction的标记设置为-100
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # 为标签创建副本
        labels = input_ids.clone()
        
        # 找到assistant部分的起始位置（根据模板中的格式）
        # 注意：这种方法依赖于特定的模板格式，如果模板改变，这里需要相应修改
        assistant_token_id = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)[0]
        
        # 寻找assistant标记的位置
        assistant_pos = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0]
        if len(assistant_pos) > 0:
            # 在助手回复前的所有标记设置为-100，表示不计算损失
            labels[:assistant_pos[-1]] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def split_dataset(dataset, eval_ratio=0.1, seed=42, shuffle=True):
    """将数据集分割为训练集和评估集"""
    if eval_ratio <= 0 or eval_ratio >= 1:
        return dataset, None
    
    dataset_size = len(dataset)
    eval_size = int(dataset_size * eval_ratio)
    
    # 如果需要，先打乱数据
    indices = list(range(dataset_size))
    if shuffle:
        rng = torch.Generator()
        rng.manual_seed(seed)
        torch.randperm(dataset_size, generator=rng, out=torch.tensor(indices, dtype=torch.long))
    
    # 分割数据集
    train_indices = indices[eval_size:]
    eval_indices = indices[:eval_size]
    
    # 创建数据集子集
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)
    
    return train_dataset, eval_dataset 

def setup_wandb(args):
    """设置Weights & Biases记录"""
    if args.use_wandb:
        logger.info("初始化Weights & Biases")
        run_name = args.wandb_name if args.wandb_name else f"qwen-sft-{args.output_dir.split('/')[-1]}"
        
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

def print_training_config(args, model_config, train_dataset, eval_dataset, is_distributed):
    """打印SFT训练配置"""
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

    # 计算有效批次大小,兼容单卡
    effective_batch_size = args.per_device_train_batch_size * gpu_count * args.gradient_accumulation_steps

    # 打印基本信息
    logger.info("\n\n")
    logger.info(separator)
    logger.info("Qwen SFT微调配置")
    logger.info(separator)
    
    # 基本信息部分
    logger.info(f"开始时间:\t{now}")
    logger.info(f"模型名称:\t{args.model_name_or_path}")
    if params_count:
        logger.info(f"模型规模:\t{params_count}")
    
    # 模型架构部分
    logger.info("\n模型架构:")
    logger.info(f"\t隐藏层数:\t{model_config.num_hidden_layers}")
    logger.info(f"\t隐藏维度:\t{model_config.hidden_size}")
    if hasattr(model_config, "num_attention_heads"):
        logger.info(f"\t注意力头数:\t{model_config.num_attention_heads}")
    if hasattr(model_config, "vocab_size"):
        logger.info(f"\t词表大小:\t{model_config.vocab_size}")
    
    # 训练数据部分
    logger.info("\n训练数据:")
    logger.info(f"\t数据目录:\t{args.data_dir}")
    logger.info(f"\t文件模式:\t{args.file_pattern}")
    logger.info(f"\t训练样本数:\t{len(train_dataset):,} 个样本")
    if eval_dataset:
        logger.info(f"\t评估样本数:\t{len(eval_dataset):,} 个样本")
    logger.info(f"\t最大序列长度:\t{args.max_seq_length}")
    
    # 训练设置部分
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
    
    # 优化器设置部分
    logger.info("\n优化器设置:")
    logger.info(f"\t学习率:\t{args.learning_rate:.1e}")
    logger.info(f"\t权重衰减:\t{args.weight_decay}")
    
    # 加速技术部分
    logger.info("\n加速技术:")
    logger.info(f"\tFP16混合精度:\t{'启用' if args.fp16 else '禁用'}")
    logger.info(f"\t梯度检查点:\t{'启用' if args.gradient_checkpointing else '禁用'}")
    
    # 保存与监控部分
    logger.info("\n保存与监控:")
    logger.info(f"\t输出目录:\t{args.output_dir}")
    logger.info(f"\t日志步数:\t{args.logging_steps}")
    logger.info(f"\t保存步数:\t{args.save_steps}")
    if eval_dataset:
        logger.info(f"\t评估步数:\t{args.eval_steps}")
    logger.info(f"\tWeights & Biases:\t{'启用' if args.use_wandb else '禁用'}")
    if args.use_wandb and wandb.run:
        if args.wandb_project:
            logger.info(f"\tWandB项目:\t{args.wandb_project}")
        else:
            logger.info(f"\tWandB项目:\t{'未指定'}")
        logger.info(f"\tWandB运行:\t{wandb.run.name}")
    
    # 其他信息部分
    logger.info("\n其他信息:")
    logger.info(f"\t随机种子:\t{args.seed}")
    
    # 添加LoRA配置信息
    if args.use_lora:
        logger.info("\nLoRA配置:")
        logger.info(f"\tLoRA秩:\t{args.lora_rank}")
        logger.info(f"\tLoRA Alpha:\t{args.lora_alpha}")
        logger.info(f"\tLoRA Dropout:\t{args.lora_dropout}")
        logger.info(f"\tLoRA目标模块:\t{args.lora_target_modules}")
    
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
    
    logger.info(f"\t预计训练时长:\t{time_str}")
    
    # 结束部分
    logger.info(separator)
    logger.info("SFT微调已开始...")
    logger.info(separator)
    logger.info("\n")

def main():
    # 忽略 SIGHUP 信号
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    
    # 优化CUDA设置
    if torch.cuda.is_available():
        # 设置缓存内存分配模式
        torch.cuda.empty_cache()
    
    args = parse_args()
    set_seed(args.seed)
    
    # 检查是否在分布式环境中
    is_distributed = args.local_rank != -1 or torch.distributed.is_initialized()
    if is_distributed:
        # 先设置GPU设备，再初始化分布式环境
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            
        # 仅初始化一次分布式环境
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
            
        # 使用barrier确保所有进程同步启动
        torch.distributed.barrier()
            
        logger.info(f"使用分布式训练，rank={args.local_rank}，设备={torch.cuda.current_device()}")
    
    # 定义主进程变量
    is_main_process = args.local_rank == -1 or args.local_rank == 0
    
    # 创建输出目录(只在主进程)
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # 添加训练时间标记
        run_time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
        wandb_run_name = f"qwen-sft-{run_time_str}"
        args.wandb_name = args.wandb_name or wandb_run_name
    
    # 初始化wandb(只在主进程)
    if is_main_process and args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"qwen-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"
    
    # 加载模型配置和分词器
    logger.info(f"加载模型配置: {args.model_name_or_path}")
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    if args.gradient_checkpointing:
        model_config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # 设置分词器的填充标记（如果没有）
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.warning("分词器没有pad_token或eos_token，将使用默认填充标记")
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    # 加载模型 - 根据是否使用量化调整参数
    logger.info(f"加载预训练模型: {args.model_name_or_path}")
    
    # 设置量化配置
    quantization_config = None
    if args.quantization:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("使用4比特量化加载模型")
        except ImportError:
            logger.warning("未安装bitsandbytes库，无法使用量化功能")
            logger.warning("请使用pip install bitsandbytes>=0.39.0安装")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        config=model_config,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if args.fp16 and torch.cuda.is_available() else None,
    )
    
    # 启用梯度检查点以节省内存
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # 应用LoRA (如果启用)
    if args.use_lora:
        logger.info("使用LoRA进行参数高效微调")
        
        # 如果使用量化，需要准备模型
        if args.quantization:
            model = prepare_model_for_kbit_training(model)
            
        # 解析目标模块
        target_modules = args.lora_target_modules.split(",")
        logger.info(f"LoRA目标模块: {target_modules}")
        
        # 设置LoRA配置
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数信息
        if is_main_process:
            model.print_trainable_parameters()
    
    # 加载数据集
    logger.info(f"加载SFT训练数据: {args.data_dir}")
    full_dataset = SFTDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        file_pattern=args.file_pattern,
        template=args.template
    )
    
    # 分割数据集
    train_dataset, eval_dataset = split_dataset(
        full_dataset, 
        eval_ratio=args.eval_split_ratio,
        seed=args.seed,
        shuffle=args.shuffle_before_split
    )
    
    # 仅在主进程打印训练配置
    if is_main_process:
        print_training_config(args, model_config, train_dataset, eval_dataset, is_distributed)

    # 设置训练参数
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
        remove_unused_columns=False,
        report_to=["wandb"] if args.use_wandb and is_main_process else [],
        run_name=wandb.run.name if args.use_wandb and is_main_process else None,
        
        # 只在非主进程禁用进度条
        disable_tqdm=not is_main_process,
        
        # 分布式训练参数
        local_rank=args.local_rank,
        dataloader_num_workers=4,  # 增加数据加载器工作线程
        dataloader_pin_memory=True,  # 启用内存固定加速数据传输
        ddp_find_unused_parameters=False,  # 关闭未使用参数检测提高性能
        
        # 添加8位优化器支持
        optim=args.optim,

        # 添加DeepSpeed支持
        deepspeed=args.deepspeed,

        # 评估配置
        evaluation_strategy="steps" if eval_dataset else "no",
        
        # Adam优化器配置
        adam_beta1=0.8,
        adam_beta2=0.99,
        
        # 保存策略
        save_total_limit=3,  # 只保留最近的3个检查点
    )
    
    logger.info(f"训练参数: logging_steps={training_args.logging_steps}, save_steps={training_args.save_steps}")
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # 确保填充到8的倍数以提高性能
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 监控模型(仅主进程)
    if args.use_wandb and args.wandb_watch != "False" and is_main_process:
        wandb.watch(model, log=args.wandb_watch, log_freq=args.logging_steps)
    
    # 开始训练
    logger.info("开始SFT微调...")
    
    train_result = trainer.train()
    
    # 输出训练结果
    if is_main_process:
        logger.info(f"训练结果: {train_result}")
        metrics = train_result.metrics
        logger.info(f"训练指标: {metrics}")
    
    # 保存模型(仅主进程)
    if is_main_process:
        logger.info("保存最终模型")
        
        # 对于LoRA模型，只保存LoRA权重
        if args.use_lora:
            model.save_pretrained(args.output_dir)
            logger.info(f"LoRA权重已保存到: {args.output_dir}")
        else:
            # 保存完整模型
            trainer.save_model(args.output_dir)
        
        tokenizer.save_pretrained(args.output_dir)
        
        # 保存训练参数
        if training_args.local_rank == 0:
            with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
                import json
                json.dump(vars(args), f, indent=4)
        
        # 打印模型目录
        logger.info(f"保存最终模型成功: {args.output_dir}")
        # wandb记录(仅主进程)
        if args.use_wandb:
            # 不上传模型，只记录训练完成
            logger.info("记录训练指标到wandb，不上传模型")
            wandb.finish()
    
    # 训练结束后
    # 确保分布式环境正确关闭
    if is_distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    logger.info("SFT微调完成!")

if __name__ == "__main__":
    try:    
        main()
    except Exception as e:
        logger.error(f"发生异常: {str(e)}")
        raise
    finally:
        # 无论是正常结束还是异常退出，都确保清理资源
        if torch.distributed.is_initialized():
            logger.info("正在关闭分布式训练环境...")
            torch.distributed.destroy_process_group()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            logger.info("正在清理CUDA缓存...")
            torch.cuda.empty_cache()
            
        logger.info("程序退出，资源已清理")

        if args and args.use_wandb:
            wandb.finish()