# 第一阶段：使用短文本训练
python -m torch.distributed.launch --nproc_per_node=2 pretrain_qwen_novel.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir output/qwen_novel_pretrain_stage1 \
  --wandb_name qwen_novel_pretrain_stage1 \
  --file_pattern "xd_chunks_32.json,xd_chunks_128.json" \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 128 \
  --num_train_epochs 1.5 \
  --learning_rate 1e-5 \
  --use_wandb

python pretrain_qwen_novel.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir output/qwen_novel_pretrain_stage1 \
  --wandb_name qwen_novel_pretrain_stage1 \
  --file_pattern "xd_chunks_32.json,xd_chunks_128.json" \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 128 \
  --num_train_epochs 1.5 \
  --learning_rate 1e-5 \
  --use_wandb

# 第二阶段：使用中等长度文本继续训练
python -m torch.distributed.launch --nproc_per_node=2 pretrain_qwen_novel.py \
  --model_name_or_path output/qwen_novel_pretrain_stage1 \
  --output_dir output/qwen_novel_pretrain_stage2 \
  --wandb_name qwen_novel_pretrain_stage2 \
  --file_pattern "xd_chunks_1024.json" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 1024 \
  --num_train_epochs 0.5 \
  --learning_rate 1e-5 \
  --logging_steps 1 \
  --use_wandb

# 1
python -m torch.distributed.launch --nproc_per_node=2 pretrain_qwen_novel.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir output/qwen_novel_pretrain_long \
  --wandb_name qwen_novel_pretrain_long \
  --file_pattern "xd_chunks_4096.json" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 4096 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --use_wandb


# 2 -m torch.distributed.launch --nproc_per_node=2 
python pretrain_qwen_novel.py \
  --model_name_or_path output/qwen_novel_pretrain_long \
  --output_dir output/qwen_novel_pretrain_short \
  --wandb_name qwen_novel_pretrain_short \
  --file_pattern "xd_chunks_128.json" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 128 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --logging_steps 5 \
  --use_wandb

# 3
python pretrain_qwen_novel.py \
  --model_name_or_path output/qwen_novel_pretrain_short \
  --output_dir output/qwen_novel_pretrain_knowledge \
  --wandb_name qwen_novel_pretrain_knowledge \
  --file_pattern "xd_knowledge_graph.json" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 2048 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --logging_steps 5 \
  --use_wandb

# 4
python pretrain_qwen_novel.py \
  --model_name_or_path output/qwen_novel_pretrain_knowledge \
  --output_dir output/qwen_novel_pretrain_knowledge2 \
  --wandb_name qwen_novel_pretrain_knowledge2 \
  --file_pattern "xd_chunks_1024.json" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 1024 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --logging_steps 5 \
  --use_wandb

python pretrain_qwen_novel.py \
  --model_name_or_path output/qwen_novel_pretrain_knowledge2 \
  --output_dir output/qwen_novel_pretrain_knowledge3 \
  --wandb_name qwen_novel_pretrain_knowledge3 \
  --file_pattern "xd_chunks_10240.json" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 8000 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --logging_steps 5 \
  --use_wandb


# 任务图谱
python -m torch.distributed.launch --nproc_per_node=2 pretrain_qwen_novel.py \
  --model_name_or_path output/qwen_novel_pretrain_stage2 \
  --output_dir output/qwen_novel_pretrain_knowledge \
  --wandb_name qwen_novel_pretrain_knowledge \
  --file_pattern "xd_knowledge_graph.json" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 512 \
  --num_train_epochs 1.0 \
  --learning_rate 1e-5 \
  --use_wandb