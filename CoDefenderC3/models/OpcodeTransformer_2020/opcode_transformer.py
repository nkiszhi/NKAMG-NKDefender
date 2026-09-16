"""
OpcodeTransformer — Ren et al. 2020
论文: "End-to-end Malware Detection for Android IoT Devices Using Deep Learning"
会议: IEEE Access
链接: https://doi.org/10.1109/ACCESS.2020.2993370

论文原始架构: Transformer Encoder on opcode sequences
  Embedding(vocab, 128) → 位置编码 → 4层 TransformerEncoder(nhead=4, ff=512)
  → 全局均值池化 → FC(128→64→2) → CrossEntropyLoss

输入: 操作码 token 序列 (LongTensor)
输出: (B, 2) 二分类 logits (用 CrossEntropyLoss 训练, softmax 后取 argmax 预测)
"""
import torch
import torch.nn as nn


class OpcodeTransformer(nn.Module):
    """Transformer 操作码恶意软件检测器 — 与论文 Section III-C 一致"""

    def __init__(self, vocab_size=512, d_model=128, nhead=4,
                 num_layers=4, max_seq_len=4096, input_dim=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # 位置编码 (Transformer 必需)
        self.pos_enc = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=512, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),  # 二分类 logits → CrossEntropyLoss
        )

    def forward(self, x):
        emb = self.embed(x.long())
        T = emb.size(1)
        emb = emb + self.pos_enc[:, :T, :]  # 加入位置编码
        encoded = self.transformer(emb)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)
