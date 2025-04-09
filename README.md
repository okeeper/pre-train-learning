# 自注意力机制预训练代码示例

这个项目提供了一个基于PyTorch实现的自注意力机制预训练代码示例，旨在帮助初学者理解Transformer模型中自注意力机制的工作原理和预训练过程。

## 环境配置

### Python虚拟环境创建

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 依赖安装

```bash
# 安装所有依赖
pip install -r requirements.txt
```

## 快速启动

### 数据准备

准备`train.txt`

### 模型预训练

```bash
# 启动预训练过程
python mini_gpt_pretrain.py

# 使用自定义参数启动预训练
python mini_gpt_pretrain.py --batch_size 32 --learning_rate 1e-4 --epochs 10
```

## 超参数设置建议

本项目默认超参数设置适合在资源受限的环境（如Mac M1）上进行小规模实验：

```python
VOCAB_SIZE = 1000  # 词汇表大小
EMBED_SIZE = 64    # 嵌入维度
NUM_HEADS = 4      # 多头注意力头数
BLOCK_SIZE = 32    # 最大序列长度
STRIDE = 4         # 滑动窗口步幅
BATCH_SIZE = 8     # 批次大小
NUM_LAYERS = 2     # Transformer 层数
DROPOUT = 0.1      # Dropout 比例
LEARNING_RATE = 0.001
EPOCHS = 50
VALIDATION_SPLIT = 0.01  # 1% 数据作为验证集
```

### 超参数调整建议

- **高性能GPU环境**：可以尝试增加 `BATCH_SIZE`（32-64）、`EMBED_SIZE`（128-256）、`NUM_LAYERS`（4-6）
- **内存受限环境**：减小 `BATCH_SIZE`（4-8）和 `BLOCK_SIZE`（16-32）
- **提高模型性能**：增加 `EMBED_SIZE`、`NUM_HEADS` 和 `NUM_LAYERS`
- **加快收敛速度**：调整 `LEARNING_RATE`（0.0001-0.001）


## 参考资料

- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- GPT模型原理与实现
