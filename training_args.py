from dataclasses import dataclass
import torch
from datetime import datetime
from pathlib import Path
import os
import json

@dataclass
class TrainingArgs:
    """模型参数配置类，用于管理模型训练的基本参数"""
    model_name: str
    model_short_name: str
    output_dir: Path
    device: torch.device
    dtype: torch.dtype
    model_version: str = "V1"
    train_file: str = "/data/training_data/xd/xd_pre_training_data_knowledge_graph.json" #"/data/training_data/xd/xd_pre_training_data_512+4096.json" #"./novel_data/xd_sft_training_11250.json" #"/data/training_data/xd/xd_sft_training_11250.json"
    config_path: str = "./config/training_config.yaml"
    wandb_project: str = "novel"
    local_files_only: bool = False
    num_proc: int = 16    #处理数据的并发数，CPU数量/训练时使用的GPU数量
    token_max_length: int = 8000
    attn_implementation: str = "flash_attention_2"  #flash_attention_2 
    system_prompt: str = ""  # 新增system_prompt参数
    attention_dropout: float = 0.0
    
    
    @classmethod
    def create(cls, model_name: str, training_type: str = "sft") -> 'TrainingArgs':
        """创建训练参数实例
        
        Args:
            model_name: 模型名称或路径
            training_type: 训练类型，默认为sft
            
        Returns:
            TrainingArgs: 模型参数实例
        """
        # 初始化输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = model_name.rstrip('/').split('/')[-1]  # 从完整路径中提取模型名称
        
        # 读取版本号并自增
        version = cls._get_and_increment_version()
        model_version = f"V{version}"
        
        output_dir = Path("outputs") / f"{cls.wandb_project}" / f"{training_type}_{model_short_name}_{model_version}_{timestamp}"
         
        # 判断进程使用哪个GPU
        if "LOCAL_RANK" in os.environ:
            gpu_index = int(os.environ.get("LOCAL_RANK", 0))
        else:
            gpu_index = 0

        # 检测设备和精度
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float32
        elif torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_index}")
            dtype = torch.bfloat16
            print(f"Rank={gpu_index} 使用GPU: {device}")
        else:
            device = torch.device("cpu")
            dtype = torch.float32
        
        
        return cls(
            device=device,
            dtype=dtype,
            model_name=model_name,
            model_short_name=model_short_name,
            output_dir=output_dir,
            model_version=model_version
        )
    
    @staticmethod
    def _get_and_increment_version() -> int:
        """从.ver.json文件中读取版本号并自增
        
        Returns:
            int: 当前版本号
        """
        ver_file = Path(".ver.json")
        
        # 如果文件不存在，创建文件并设置初始版本为1
        if not ver_file.exists():
            version_data = {"version": 1}
            with open(ver_file, 'w', encoding='utf-8') as f:
                json.dump(version_data, f)
            return 1
        
        # 读取当前版本号
        try:
            with open(ver_file, 'r', encoding='utf-8') as f:
                version_data = json.load(f)
                current_version = version_data.get("version", 1)
        except (json.JSONDecodeError, FileNotFoundError):
            # 如果文件损坏或无法读取，重置版本号
            current_version = 1
        
        # 自增版本号并保存
        next_version = current_version + 1
        version_data = {"version": next_version}
        
        with open(ver_file, 'w', encoding='utf-8') as f:
            json.dump(version_data, f)
        
        return current_version
    
    def to_dict(self) -> dict:
        """将配置转换为字典

        该方法用于将TrainingArgs类的实例转换为字典形式，方便后续处理和保存。
        """
        return vars(self)
