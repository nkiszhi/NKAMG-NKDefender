"""
OpcodeLSTM — McLaughlin et al. 2017
论文: "Deep Android Malware Detection"
会议: IEEE Transactions on Information Forensics and Security (TIFS)
链接: https://doi.org/10.1109/TIFS.2017.2713418

论文原始架构: BiLSTM on opcode sequences
  Embedding(vocab_size, 64) → 2层 BiLSTM(hidden=128)
  → 全局均值池化 → FC(256→64→2) → CrossEntropyLoss

输入: BinaryNinja 反汇编得到的操作码序列 (LongTensor)
输出: (B, 2) 二分类 logits (用 CrossEntropyLoss 训练, softmax 后取 argmax 预测)
"""
import torch.nn as nn


class OpcodeLSTM(nn.Module):
    """双向 LSTM 操作码序列恶意软件检测器 — 与论文 Section III-B 一致"""

    def __init__(self, vocab_size=512, embed_dim=64, hidden_dim=128,
                 num_layers=2, input_dim=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # BiLSTM 输出维度 × 2
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),  # 二分类 logits → CrossEntropyLoss
        )

    def forward(self, x):
        """
        参数:
            x: LongTensor(B, seq_len), 操作码索引
        返回:
            Tensor(B, 2), 二分类 logits
        """
        emb = self.embed(x.long())            # (B, L, embed_dim)
        lstm_out, _ = self.lstm(emb)           # (B, L, hidden*2)
        pooled = lstm_out.mean(dim=1)          # (B, hidden*2) 均值池化
        return self.classifier(pooled)
