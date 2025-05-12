import os
import json
import torch
import signal
import shutil
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    EarlyStoppingCallback,
    AutoConfig
)
from datasets import Dataset, DatasetDict
from datetime import datetime
import wandb
from training_config import TrainingConfig
from training_args import TrainingArgs
from training_util import log_info, is_main_process, output_training_args
import argparse
import deepspeed

def handle_exit(output_dir, signum=None, frame=None):
    """处理退出时的清理工作，可同时作为信号处理函数使用"""
    if signum is not None:
        log_info("\n接收到中断信号，正在清理并退出...")
    
    # 清理 DeepSpeed 环境
    if deepspeed.comm.is_initialized():
        log_info("\n正在清理 DeepSpeed 分布式环境...")
        deepspeed.comm.destroy_process_group()
    
    if output_dir and is_main_process():
        try:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                log_info(f"\n已清理输出目录：{output_dir}")
        except Exception as e:
            log_info(f"\n清理输出目录时出错：{str(e)}")
    
    if signum is not None:
        exit(1)


def freeze_layers(model, num_layers_to_freeze=0):
    """锁定模型的底部层
    Args:
        model: 要处理的模型
        num_layers_to_freeze: 要冻结的层数，0表示不冻结任何层
    """
    if num_layers_to_freeze <= 0:
        return
    
    # 计算模型总层数
    total_layers = 0
    for name, _ in model.named_parameters():
        if "layers" in name:
            layer_num = int(name.split(".")[2])
            total_layers = max(total_layers, layer_num + 1)
    
    log_info(f"\n模型总层数: {total_layers} 层")
    log_info(f"正在冻结模型底部的 {num_layers_to_freeze}/{total_layers} 层...")
    
    for name, param in model.named_parameters():
        if "layers" in name:
            layer_num = int(name.split(".")[2])
            if layer_num < num_layers_to_freeze:
                param.requires_grad = False
    
    # 计算可训练的参数量
    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    log_info(f"\n模型总参数量: {all_params/1e6:.2f}M")
    log_info(f"可训练参数量: {trainable_params/1e6:.2f}M")
    log_info(f"参数冻结比例: {(1 - trainable_params/all_params)*100:.2f}%")


def main(training_args: TrainingArgs):
    # 加载训练配置
    log_info("\n正在加载训练配置参数...")
    config = TrainingConfig.from_yaml(training_args.config_path)

    # 初始化wandb（仅在主进程中）
    if is_main_process():
        wandb.init(
            project=training_args.wandb_project,
            config=config.to_dict(),
            tags=[training_args.model_short_name, "pre_trainning"],
            name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dir="./wandb",
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"  # 其他进程禁用wandb

    log_info("============初始化的相关配置============")
    for key, value in training_args.to_dict().items():
        log_info(f"  -{key}: {value}")
    

    log_info(f"\n开始加载模型 {training_args.model_name}...")

    # 加载模型配置
    model_config = AutoConfig.from_pretrained(
        training_args.model_name,
        trust_remote_code=True,
        local_files_only=training_args.local_files_only
    )
    # 训练的时候不需要use_cache
    model_config.use_cache = False
    # 调整模型配置
    if training_args.attention_dropout > 0:
        model_config.attention_dropout = training_args.attention_dropout
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name,
        trust_remote_code=True,
        local_files_only=training_args.local_files_only
    )
    
    # 设置padding_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_name,
        config=model_config,
        trust_remote_code=True,
        torch_dtype=training_args.dtype,
        low_cpu_mem_usage=True,
        #use_cache=False,  # 禁用缓存以节省内存
        local_files_only=training_args.local_files_only,
        attn_implementation=training_args.attn_implementation
    )

    if training_args.attention_dropout > 0:
        log_info(f"加载后模型的配置信息：\n{model.config}")
    
    # 可以根据需要打开下面的代码来冻结指定数量的层
    #freeze_layers(model, num_layers_to_freeze=24)  # 比如冻结底部12层
    
    # 打印模型配置信息
    log_info(f"模型所在的设备信息：{next(model.parameters()).device}")

    def prepare_features(examples):
        """将文本数据转换为模型输入格式"""
        all_input_ids = []
        all_attention_masks = []
        
        for idx, text in enumerate(examples["text"]):
            if not text:
                log_info(f"警告：第{idx}条数据为空，已跳过")
                continue
            
            #这里需要显示增加eos token吗？Qwen的eos token是<|im_end|>，貌似不应该加
            #text = text + tokenizer.eos_token
            
            # 编码文本并进行截断
            encoded = tokenizer.encode(
                text,
                add_special_tokens=True
            )
            
            # 添加解码打印，观察编码结果
            #if idx == 0:  # 只打印第一条数据，避免输出过多
            #    decoded = tokenizer.decode(encoded)
            #    log_info(f"\n编码示例 (前50个字符):\n解码: {decoded}")
            
            # 创建输入和标签
            input_ids = encoded
            attention_mask = [1] * len(input_ids)

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)


        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks
        }
    
    def load_dataset(train_file):
        """从JSON文件加载预训练数据集"""
        if not os.path.exists(train_file):
            raise ValueError(f"训练文件不存在：{train_file}")
        
        log_info(f"正在加载预训练数据集：{train_file}")
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                texts = []
                for idx, item in enumerate(data, 1):
                    if not isinstance(item, dict):
                        log_info(f"警告：第 {idx} 条数据格式无效，已跳过")
                        continue
                    
                    text = item.get("text", "")
                    if text:
                        texts.append(text)
                
                if not texts:
                    raise ValueError("没有找到有效的训练数据！请检查JSON文件格式")
                
                log_info(f"成功加载了 {len(texts)} 条预训练数据")
                
                # 创建数据集并分割
                full_dataset = Dataset.from_dict({"text": texts})
                #split_dataset = full_dataset.train_test_split(test_size=0.01, seed=42)
                
                return DatasetDict({
                    "train": full_dataset,
                    #"validation": split_dataset["test"]
                })
                
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON文件格式错误：{str(e)}")
        except Exception as e:
            raise Exception(f"加载数据集时出错：{str(e)}")
    
    # 加载数据集
    dataset = load_dataset(training_args.train_file)
    
    # 对数据集进行处理
    log_info("\n正在对数据集进行批量处理...")
    tokenized_dataset = dataset.map(
        prepare_features,
        batched=True,
        batch_size=10000,
        num_proc=training_args.num_proc,  # 与CPU数量/GPU数量相同
        remove_columns=dataset["train"].column_names,
        desc="正在处理数据集"
    )

    # 添加长度过滤
    log_info(f"\n正在过滤长度超过token_max_length={training_args.token_max_length}的数据...")
    filtered_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) <= training_args.token_max_length,
        num_proc=training_args.num_proc,  # 与CPU数量/GPU数量相同
        desc="正在过滤超长数据"
    )
    
    # 显示数据集信息
    log_info("=== 数据集信息 ===")
    log_info(f"【原始的】训练集大小: {len(dataset['train'])} 条数据")
    log_info(f"【处理后】训练集大小: {len(tokenized_dataset['train'])} 条数据")
    log_info(f"【过滤后】训练集大小: {len(filtered_dataset['train'])} 条数据")

    # 配置训练参数
    config.eval_strategy = 'no'
    config.metric_for_best_model = "loss"
    base_config = config.to_dict()
    training_args = TrainingArguments(
        output_dir=str(training_args.output_dir),
        gradient_checkpointing=True,
        report_to="wandb" if is_main_process() else [],
        seed=88,
        data_seed=168,
        bf16=True if training_args.dtype == torch.bfloat16 else False,
        run_name=wandb.run.name if is_main_process() else None,
        adam_beta1 = 0.8,
        adam_beta2 = 0.99,
        deepspeed="config/ds_config.json",
        **base_config
    )
    
    # 使用DataCollatorForLanguageModeling进行数据整理
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 不使用掩码语言建模
        pad_to_multiple_of=8,  # 确保填充到8的倍数以提高性能
    )

    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=filtered_dataset["train"],
        data_collator=data_collator,
        
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)],
    )
    
    try:
        output_training_args(training_args)
        log_info("\n开始进行【预训练】...")
        trainer.train()

        # 确保所有进程都完成训练
        if deepspeed.comm.is_initialized():
            log_info("\n等待所有进程完成训练...")
            deepspeed.comm.barrier()
            log_info("\n所有进程已完成训练")

        # 保存最佳模型
        if is_main_process():
            
            # 更新模型配置
            trainer.model.config.update({
                "attention_dropout": 0.0,
                "use_cache": True
            })

            trainer.save_model(os.path.join(training_args.output_dir, "best_model"))
            tokenizer.save_pretrained(training_args.output_dir)
            log_info(f"最佳模型已保存,Config={trainer.model.config}")
            log_info(f"最佳模型已保存,Path={os.path.join(training_args.output_dir, 'best_model')}")


    except Exception as e:
        log_info(f"\n训练过程中出现错误：{str(e)}")
        raise e


if __name__ == "__main__":
    # 1. 解析模型名
    parser = argparse.ArgumentParser(description='预训练脚本')
    parser.add_argument('--model_path', type=str)
    args, _ = parser.parse_known_args()

    local_files_only = False
    # 如果没有指定参数，使用默认模型
    if not args.model_path:
        args.model_path = "Qwen/Qwen2.5-1.5B-Instruct"
        local_files_only = False
    else:
        # 检查模型路径是否存在
        local_files_only = os.path.exists(args.model_path)

    log_info(f"正在使用模型：{args.model_path} local_files_only={local_files_only}")
    
    # 2. 创建训练参数实例
    training_args = TrainingArgs.create(args.model_path, training_type="pt")
    training_args.local_files_only = local_files_only

    log_info(f"训练后的模型和检查点将保存在：{training_args.output_dir}")

    # 3. 注册信号处理器
    if is_main_process():
        signal.signal(signal.SIGINT, lambda signum, frame: handle_exit(training_args.output_dir, signum, frame))
    
    # 4. 执行主函数
    try:
        main(training_args)
    finally:
        if deepspeed.comm.is_initialized():
            log_info("\n程序退出，正在清理 DeepSpeed 分布式环境...")
            deepspeed.comm.destroy_process_group()
