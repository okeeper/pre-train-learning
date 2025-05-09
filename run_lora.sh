#!/bin/bash

# 使用LoRA技术训练Qwen3-8B模型，大幅降低内存需求
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --config_file accelerate_config.yaml pretrain_qwen_novel.py \
  --model_name_or_path /data/hf-models/Qwen3-8B \
  --output_dir output/qwen3_novel_lora_pretrain \
  --wandb_name qwen3_novel_lora_pretrain \
  --file_pattern "pretrain_output/novel_pretrain_data.jsonl" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 2048 \
  --num_train_epochs 1.0 \
  --learning_rate 2e-5 \
  --gradient_checkpointing \
  --use_wandb \
  --use_lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" 