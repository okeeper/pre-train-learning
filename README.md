# 小说问答数据集生成工具

本工具用于从小说文本生成问答数据集，使用OpenAI API生成高质量的问答对。

## 环境配置

### 1. 创建并激活conda环境
```bash
# 创建Python环境
conda create -n pre-train python=3.10
# 激活环境
conda activate pre-train
```

### 2. 安装必要依赖
```bash
# 安装基础依赖
pip install openai backoff tqdm

# 如果需要使用本地模型（可选）
pip install torch
pip install modelscope transformers peft datasets
```

### 3. OpenAI API配置
程序首次运行时会提示配置API，您可以选择：

1. 通过环境变量设置：
```bash
# Linux/Mac
export OPENAI_API_KEY="你的API密钥"
export OPENAI_API_BASE="你的API基础URL"  # 可选，如果使用代理

# Windows
set OPENAI_API_KEY=你的API密钥
set OPENAI_API_BASE=你的API基础URL  # 可选，如果使用代理
```

2. 运行程序时根据提示输入

## 使用说明

### 1. 准备小说文本
- 文件第一行应包含用`<>`括起的小说标题，如：`<西游记>`
- 章节标题行应该顶格或只有一个空格缩进
- 正文段落应该有两个以上空格缩进

### 2. 运行程序
```bash
# 确保在novel_qa环境中
conda activate novel_qa

# 运行程序
python prepare_qwen_qa.py
```

## 输出文件
程序会在`data`目录下生成：
- `novel_chunks.json`: 分割后的小说章节
- `novel_qa_data.jsonl`: 生成的问答训练数据

## 常用conda命令
```bash
# 查看当前环境
conda env list

# 退出环境
conda deactivate

# 删除环境（如需要）
conda remove --name novel_qa --all
```


# 预训练

```
conda activate llama-factory
conda activate zye
```


```
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

## chat
```

python test_chat.py --model_path output/output/qwen_novel_pretrain_long \
--max_new_tokens 512 --temperature 0.7

python test_chat.py --model_path output/qwen_novel_pretrain_short \
--max_new_tokens 512 --temperature 0.7

python test_chat.py --model_path output/qwen_novel_pretrain_knowledge \
--max_new_tokens 512 --temperature 0.7


# 原模型
python test_chat.py --model_path Qwen/Qwen2.5-1.5B-Instruct \
--max_new_tokens 512 --temperature 0.7




# 使用LoRA
python test_chat.py --model_path /path/to/your/model --lora_path /path/to/your/lora
```


# 评估
```
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
```