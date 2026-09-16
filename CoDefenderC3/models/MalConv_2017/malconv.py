"""
MalConv — Raff et al. 2018
"Malware Detection by Eating a Whole EXE"
AAAI Workshop on Artificial Intelligence for Cyber Security

Architecture:
  Input: raw byte sequence [0, 256], padded/truncated to max_len
  Embedding(257, 8) → Conv1d(8, 128, k=500, stride=500)
  Gating mechanism: main_path * sigmoid(gate_path)
  Global Max Pooling → FC(128 → 1) → Sigmoid

论文原始实现: 输入最长 200KB raw bytes, 无需特征工程
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Malconv(nn.Module):
    def __init__(self, max_len=200000, win_size=500, vocab_size=256,
                 embed_dim=8, num_filters=128):
        super(Malconv, self).__init__()
        self.max_len = max_len

        # 字节嵌入层: 0=padding, 1-256=byte values
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        # 门控卷积 (Gated Convolution)
        self.conv_main = nn.Conv1d(embed_dim, num_filters, win_size,
                                   stride=win_size, bias=True)
        self.conv_gate = nn.Conv1d(embed_dim, num_filters, win_size,
                                   stride=win_size, bias=True)

        # 分类头
        self.fc1 = nn.Linear(num_filters, num_filters)
        self.fc2 = nn.Linear(num_filters, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: LongTensor (B, L), byte values in [0, 256]
        Returns:
            Tensor (B, 1), malware probability
        """
        x = x.long()
        # Embedding
        emb = self.embedding(x)          # (B, L, E)
        emb = emb.permute(0, 2, 1)       # (B, E, L)

        # Gated convolution
        main_path = self.conv_main(emb)   # (B, C, L')
        gate_path = self.conv_gate(emb)   # (B, C, L')
        gated = main_path * torch.sigmoid(gate_path)

        # Global max pooling
        pooled = F.adaptive_max_pool1d(gated, 1).squeeze(-1)  # (B, C)

        # Classification
        out = F.relu(self.fc1(pooled))
        out = self.sigmoid(self.fc2(out))  # (B, 1)
        return out
