
# 多卡直接sft
CUDA_VISIBLE_DEVICES=0,1,2,3 \
deepspeed --num_gpus=2 sft_qwen_novel.py \
  --model_name_or_path /data/hf-models/Qwen3-8B \
  --output_dir output/sft/xd_sft \
  --wandb_name xd_sft \
  --file_pattern "sft/xd_final_sft.json,sft/alpaca_zh_demo.json" \
  --max_seq_length 1024 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1.5 \
  --learning_rate 1e-6 \
  --gradient_checkpointing \
  --logging_steps 1 \
  --use_wandb \
  --deepspeed ds_config.json

# 单卡
python sft_qwen_novel.py \
  --model_name_or_path /data/hf-models/Qwen3-8B \
  --output_dir output/sft/xd_sft \
  --wandb_name xd_sft \
  --file_pattern "sft/xd_final_sft.json,sft/alpaca_zh_demo.json" \
  --max_seq_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1.5 \
  --learning_rate 1e-6 \
  --gradient_checkpointing \
  --logging_steps 1 \
  --use_wandb
 