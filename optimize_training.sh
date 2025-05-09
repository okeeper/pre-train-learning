#!/bin/bash

# 设置环境变量以优化训练速度
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 优化CUDA性能
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# 运行训练脚本
accelerate launch \
    --num_processes=2 \
    --mixed_precision=fp16 \
    --use_deepspeed \
    --deepspeed_config_file=./ds_config.json \
    pretrain_qwen_novel.py \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --bf16 False \
    --fp16 True \
    --torch_dtype float16 \
    --gradient_checkpointing True \
    --dataloader_prefetch_factor 4 \
    --dataloader_num_workers 8 \
    --ddp_find_unused_parameters False \
    --dataloader_pin_memory True \
    "$@"

# 添加此脚本的使用说明
echo "优化训练脚本使用方法:"
echo "bash optimize_training.sh [其他训练参数]" 