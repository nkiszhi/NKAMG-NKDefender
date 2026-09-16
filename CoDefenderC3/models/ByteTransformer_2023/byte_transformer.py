"""
ByteTransformer — 基于 Transformer 的字节序列恶意软件检测
论文: 基于 Vaswani et al. 2017 "Attention Is All You Need"
链接: https://arxiv.org/abs/1706.03762

架构: 字节分块(64) → Embedding(257, 128) → 位置编码 →
      4层 TransformerEncoder(nhead=4, ff=512) → 全局均值池化 → FC(128→2)
输入: 原始 PE 字节序列 LongTensor(B, L)
输出: (B, 2) 二分类 logits → CrossEntropyLoss
"""
import torch
import torch.nn as nn


class ByteTransformer(nn.Module):
    """基于 Transformer 的字节序列恶意软件检测器"""

    def __init__(self, max_len=32768, d_model=128, nhead=4, dim_ff=512,
                 n_layers=4, chunk_size=64, num_classes=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.embed = nn.Embedding(257, d_model, padding_idx=0)
        self.pos_enc = nn.Parameter(
            torch.randn(1, max_len // chunk_size, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.3, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),  # logits → CrossEntropyLoss
        )

    def forward(self, x):
        """
        参数:
            x: LongTensor(B, L), 字节值 [0, 256]
        返回:
            Tensor(B, num_classes), 二分类 logits
        """
        B, L = x.shape
        x = x.long()
        cs = self.chunk_size
        L_trim = (L // cs) * cs
        if L_trim == 0:
            x = torch.nn.functional.pad(x, (0, cs - L))
            L_trim = cs
        x = x[:, :L_trim].reshape(B, -1, cs)
        e = self.embed(x).mean(dim=2)
        T = e.size(1)
        e = e + self.pos_enc[:, :T, :]
        z = self.encoder(e).mean(dim=1)
        return self.classifier(z)
