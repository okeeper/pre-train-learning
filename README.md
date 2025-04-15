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
