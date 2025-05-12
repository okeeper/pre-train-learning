#!/bin/bash

# 设置CUDA内存分配策略，解决内存碎片问题
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# 设置NCCL优化参数
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0  
export NCCL_BUFFSIZE=2097152
export NCCL_IB_TIMEOUT=22
export NCCL_ASYNC_ERROR_HANDLING=1

# 设置DeepSpeed和PyTorch性能参数
export DS_PIPE_RESERVE_PERCENT=50
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_DISTRIBUTED_DEBUG=INFO

# 在完全退出时清理PID
trap "pkill -P $$" EXIT

# 使用DeepSpeed运行训练，固定CUDA设备顺序
# 注意：rank=local_rank映射可能会自动调整，所以使用explicit rank mapping策略
deepspeed --num_gpus=2 --master_port=29500 pretrain_qwen_novel.py \
  --model_name_or_path /data/hf-models/Qwen3-8B \
  --output_dir output/qwen3_novel_full_pretrain \
  --wandb_name qwen3_novel_full_pretrain \
  --file_pattern "pretrain_output/novel_pretrain_data.jsonl" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 2048 \
  --num_train_epochs 1.0 \
  --learning_rate 2e-5 \
  --fp16 \
  --gradient_checkpointing \
  --use_wandb \
  --deepspeed ds_config.json 