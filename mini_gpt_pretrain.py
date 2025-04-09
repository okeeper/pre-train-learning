import torch
import torch.nn as nn
import torch.optim as optim
import math
import jieba
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

# 设置随机种子，确保结果可复现
torch.manual_seed(42)

# 超参数设置（适合 Mac M1 的小规模参数）
VOCAB_SIZE = 1000  # 词汇表大小
EMBED_SIZE = 64    # 嵌入维度
NUM_HEADS = 4      # 多头注意力头数
BLOCK_SIZE = 32    # 最大序列长度
STRIDE = 4         # 滑动窗口步幅
BATCH_SIZE = 8     # 批次大小
NUM_LAYERS = 2     # Transformer 层数
DROPOUT = 0.1      # Dropout 比例
LEARNING_RATE = 0.001
EPOCHS = 50  # 增加 epoch 以观察趋势
VALIDATION_SPLIT = 0.01  # 20% 数据作为验证集

# 1. 构建词汇表
def build_vocab_from_file(file_path, max_vocab_size=VOCAB_SIZE):
    word_freq = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = jieba.cut(line.strip(), cut_all=False)
            word_freq.update(tokens)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_freq.most_common(max_vocab_size - 1))}
    vocab['<UNK>'] = 0
    return vocab

# 将词序列转换为索引序列
def tokens_to_indices(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

# 2. 数据集类
class TextDataset(Dataset):
    def __init__(self, file_path, block_size, stride, vocab):
        self.block_size = block_size
        self.stride = stride
        self.vocab = vocab
        self.data = []
        
        all_tokens = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = list(jieba.cut(line.strip(), cut_all=False))
                all_tokens.extend(tokens)
        
        indices = tokens_to_indices(all_tokens, vocab)
        for i in range(0, len(indices) - block_size, stride):
            seq = indices[i:i + block_size + 1]
            self.data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return (torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long))

# 3. Embedding 层
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, block_size):
        super().__init__()
        pe = torch.zeros(block_size, embed_size)
        for pos in range(block_size):
            for i in range(0, embed_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / embed_size)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / embed_size)))
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_enc = PositionalEncoding(embed_size, block_size)

    def forward(self, x):
        x = self.token_emb(x)
        return self.pos_enc(x)

# 4. Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, block_size, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.register_buffer("mask", torch.triu(torch.ones(block_size, block_size), diagonal=1).bool())

    def forward(self, x):
        seq_len = x.size(0)
        attn_output, _ = self.attn(x, x, x, attn_mask=self.mask[:seq_len, :seq_len])
        x = self.norm1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        return x

# 5. 完整模型
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size, num_heads, num_layers, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_size, block_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, block_size, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_f(x)
        x = x.transpose(0, 1)
        logits = self.head(x)
        return logits

# 初始化模型、损失函数和优化器
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MiniGPT(VOCAB_SIZE, EMBED_SIZE, BLOCK_SIZE, NUM_HEADS, NUM_LAYERS, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 测试生成
def generate(model, vocab, start_text, max_length=10):
    model.eval()
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    with torch.no_grad():
        start_tokens = list(jieba.cut(start_text, cut_all=False))
        indices = torch.tensor(tokens_to_indices(start_tokens, vocab), dtype=torch.long).unsqueeze(0).to(device)
        for _ in range(max_length):
            logits = model(indices)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            indices = torch.cat([indices, next_token.unsqueeze(0)], dim=1)
        return "".join([reverse_vocab.get(idx.item(), '<UNK>') for idx in indices[0]])

# 6 & 7. 训练循环（监控训练和验证损失）
def train(model, train_loader, epochs):
    train_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_count += 1
        
        avg_train_loss = train_loss / train_count
        train_losses.append(avg_train_loss)
     
        # 计算困惑度
        train_perplexity = math.exp(avg_train_loss)
        
        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}")
        print(generate(model, vocab, "人工智能是", max_length=10))
    
    # 绘制损失曲线
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

# 8. 运行训练
file_path = "train.txt"
print("Building vocabulary...")
vocab = build_vocab_from_file(file_path)
dataset = TextDataset(file_path, BLOCK_SIZE, STRIDE, vocab)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Training on {len(dataset)} samples")
print("Starting training...")
train(model, train_loader, EPOCHS)

# 保存模型
torch.save(model.state_dict(), "mini_gpt.pth")
print("Training complete! Model saved as 'mini_gpt.pth'")

print("Generated text:", generate(model, vocab, "人工智能是", max_length=10))