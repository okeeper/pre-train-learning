# 小说问答数据集生成工具

本工具用于从小说文本生成问答数据集，使用OpenAI API生成高质量的问答对。

## 环境配置

### 1. 创建并激活conda环境
```bash
conda create -n qwen_train python=3.10
conda activate qwen_train
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia


pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

### 2. 安装必要依赖


```bash
# 安装基础依赖
pip install -r requirements.txt
```


# 预训练

```shell
conda activate llama-factory
conda activate zye
```

## 分布式多线DPP启动
```shell
python -m torch.distributed.launch --nproc_per_node=2 pretrain_qwen_novel.py \
  --output_dir output/qwen1.5b_xd_pretrain \
  --file_pattern xd_chunks_32.json,xd_chunks_128.json,xd_chunks_1024.json,xd_chunks_4096.json \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --max_seq_length 4096 \
  --fp16 \
  --gradient_checkpointing \
  --num_train_epochs 3.0 \
  --logging_steps 10 \
  --save_steps 500 \
  --use_wandb
```

## 单卡预训练
```shell
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
```

## acclerate分布式启动
```shell
accelerate launch --config_file accelerate_config.yaml pretrain_qwen_novel.py \
  --model_name_or_path /data/hf-models/Qwen3-8B \
  --output_dir output/qwen3_novel_full_pretrain \
  --wandb_name qwen3_novel_full_pretrain \
  --file_pattern "pretrain_output/novel_pretrain_data.jsonl" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 4096 \
  --num_train_epochs 1.0 \
  --learning_rate 2e-5 \
  --fp16 \
  --gradient_checkpointing \
  --use_wandb \
  --deepspeed ds_config.json
  
```


## 启动chat程序
```shell

python test_chat.py --model_path output/output/qwen_novel_pretrain_long \
--max_new_tokens 512 --temperature 0.7

python test_chat.py --model_path output/qwen_novel_pretrain_short \
--max_new_tokens 512 --temperature 0.7

python test_chat.py --model_path output/qwen_novel_pretrain_knowledge \
--max_new_tokens 512 --temperature 0.7


# 原模型
python test_chat.py --model_path /data/hf-models/Qwen3-8B \
--quantization 4bit \
--max_new_tokens 1024 --temperature 0.7 \
--gpu_memory_efficient --history_length 2
--cpu_offload



python test_chat.py --model_path output/qwen3_novel_full_pt_1024 \
--max_new_tokens 1024 --temperature 0.7

# 使用LoRA
python test_chat.py --model_path output/qwen3_novel_full_pt_1024 \
--lora_path output/qwen3_novel_lora_pretrain \
--max_new_tokens 1024 --temperature 0.7


```


# 评估
```shell
# c3,squad_v2,
python model_evaluation.py \
    --model_path output/qwen_novel_pretrain_knowledge2 \
    --perplexity_dataset data/xd_eval_preplexity.txt \
    --novel_name "亵渎" \
    --qa_dataset ./data/xd_qa_alpaca_01.json,./data/xd_qa_alpaca_02.json \
    --single_choice_dataset ./data/xd_eval_choice.csv \
    --output_dir ./output/evaluation_results \
    --tasks perplexity,qa,generation,single_choice \
    --num_samples -1 \
    --batch_size 16 \
    --fp16 \
    --use_wandb

# Qwen原模型
python model_evaluation.py \
    --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --perplexity_dataset data/xd_eval_preplexity.txt \
    --novel_name "亵渎" \
    --qa_dataset ./data/xd_eval_qa2.csv \
    --single_choice_dataset ./data/xd_eval_choice.csv \
    --output_dir ./output/evaluation_results \
    --tasks perplexity,qa,generation,single_choice \
    --num_samples -1 \
    --batch_size 16 \
    --fp16 \
    --use_wandb

# 仅chunks_128 chunks_4096
python model_evaluation.py \
    --model_path output/qwen_novel_pretrain_short \
    --perplexity_dataset data/xd_eval_preplexity.txt \
    --novel_name "亵渎" \
    --qa_dataset ./data/xd_eval_qa2.csv \
    --single_choice_dataset ./data/xd_eval_choice.csv \
    --output_dir ./output/evaluation_results \
    --tasks perplexity,qa,generation,single_choice \
    --num_samples -1 \
    --batch_size 16 \
    --fp16 \
    --use_wandb

# 10240 chunks

python model_evaluation.py \
    --model_path output/qwen_novel_pretrain_knowledge3 \
    --perplexity_dataset data/xd_eval_preplexity.txt \
    --novel_name "亵渎" \
    --qa_dataset ./data/xd_eval_qa2.csv \
    --single_choice_dataset ./data/xd_eval_choice.csv \
    --output_dir ./output/evaluation_results \
    --tasks perplexity,qa,single_choice \
    --num_samples -1 \
    --batch_size 16 \
    --fp16 \
    --use_wandb


# 评估Qwen3 lora
python model_evaluation.py \
    --model_path /data/hf-models/Qwen3-8B \
    --use_lora \
    --lora_model_path "output/qwen3_novel_lora_pretrain" \
    --perplexity_dataset data/xd_eval_preplexity.txt \
    --novel_name "亵渎" \
    --qa_dataset ./data/xd_eval_qa2.csv \
    --single_choice_dataset ./data/xd_eval_choice.csv \
    --output_dir ./output/evaluation_results \
    --tasks perplexity,qa,single_choice \
    --batch_size 16 \
    --fp16 \
    --use_wandb

python model_evaluation.py \
    --model_path /data/hf-models/Qwen3-8B \
    --perplexity_dataset data/xd_eval_preplexity.txt \
    --novel_name "亵渎" \
    --qa_dataset ./data/xd_eval_qa2.csv \
    --single_choice_dataset ./data/xd_eval_choice.csv \
    --output_dir ./output/evaluation_results \
    --tasks perplexity,qa,single_choice \
    --batch_size 16 \
    --fp16 \
    --use_wandb
```


# 数据处理
```shell
nohup python novel_pretrain_data_generator.py --input data/xd_chunks_tokens_4096.json \
--openai-base-url "http://192.168.16.125:31958/v1" \
--openai-api-key "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIyIiwiYXBwTmFtZSI6InRlc3QiLCJleHAiOjI1NTA4MjI5MzN9.NexvMHMtds-MwbUfPNk1jBNOOV-nKvBxznCSpmvuqhA" \
--output data/pretrain_output > /dev/null 2>&1 &



python novel_pretrain_data_generator.py \
--resume-from "亵渎 盛世年华卷　章十三　天灾　上(2)" \
--openai-api-key "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIyIiwiYXBwTmFtZSI6InRlc3QiLCJleHAiOjI1NTA4MjI5MzN9.NexvMHMtds-MwbUfPNk1jBNOOV-nKvBxznCSpmvuqhA"
```



```
# 使用清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里云源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

pip install transformers==4.31.0  # 一个稳定且兼容性强的版本

# 降级PyTorch版本, 解决PyTorch 2.6的分布式张量(DTensor)功能与普通张量混合使用时出现了冲突。
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html -i https://mirrors.aliyun.com/pypi/simple

pip install -i https://mirrors.aliyun.com/pypi/simple torch==2.0.1+cu118
```



# 监控gpu启动
```

sh run_on_idle_gpu.sh -g 1 -m 30 -i 10 'accelerate launch --config_file accelerate_config.yaml pretrain_qwen_novel.py \
  --model_name_or_path /data/hf-models/Qwen3-8B \
  --output_dir output/qwen3_novel_full_pretrain \
  --wandb_name qwen3_novel_full_pretrain \
  --file_pattern "pretrain_output/novel_pretrain_data.jsonl" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 4096 \
  --num_train_epochs 1.0 \
  --learning_rate 2e-5 \
  --gradient_checkpointing \
  --use_wandb \
  --deepspeed ds_config.json'




nohup accelerate launch --config_file accelerate_config.yaml pretrain_qwen_novel.py \
  --model_name_or_path /data/hf-models/Qwen3-8B \
  --output_dir output/qwen3_novel_full_pretrain \
  --wandb_name qwen3_novel_full_pretrain \
  --file_pattern "pretrain_output/novel_pretrain_data.jsonl" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 4096 \
  --num_train_epochs 2.0 \
  --learning_rate 1e-7 \
  --logging_steps 1 \
  --learning_rate 2e-5 \
  --gradient_checkpointing \
  --use_wandb \
  --deepspeed ds_config.json \
2>&1 &
```

-m <阈值>：内存使用率阈值，低于此值视为空闲 (默认: 20.0%)
-u <阈值>：计算负载阈值，低于此值视为空闲 (默认: 10.0%)
-i <秒数>：检查间隔时间，单位秒 (默认: 60)
-n <次数>：连续几次检查空闲才执行命令 (默认: 3)
-g <GPU ID>：指定要监控的GPU ID，不指定则监控所有GPU
-r：命令执行完毕后继续监控并在GPU空闲时再次执行


# 监控
```
watch -n 1 nvidia-smi
```