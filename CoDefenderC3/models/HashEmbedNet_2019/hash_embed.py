"""
HashEmbedNet — Pagani et al. 2019
论文: "Towards Interpretable and Robust ML-based Malware Detection"
会议: CCS Workshop on AI for Security
链接: https://doi.org/10.1145/3338466.3358907

架构:
  4种哈希 (MD5/SHA256/Imphash/SSDeep) → 各自 Embedding(vocab, 64)
  Concat(4×64=256d) → FC(256→128→64→1) → Sigmoid

当输入为预提取特征向量时, 跳过 Embedding 层直接进入 MLP

输入: LongTensor(B, 4) 哈希索引 或 FloatTensor(B, D) 预提取特征
输出: (B, 2) 二分类 logits (用 CrossEntropyLoss 训练, softmax 后取 argmax 预测)
"""
import torch
import torch.nn as nn


class HashEmbedNet(nn.Module):
    """基于哈希嵌入的恶意软件检测器 — 含4种独立哈希 Embedding"""

    def __init__(self, input_dim=512, num_hash_types=4,
                 embed_dim=64, hash_vocab_size=65536):
        """
        参数:
            input_dim: 预提取特征维度 (当不使用 Embedding 时)
            num_hash_types: 哈希类型数量 (MD5/SHA256/Imphash/SSDeep)
            embed_dim: 每种哈希的嵌入维度
            hash_vocab_size: 哈希值的词表大小 (取模后的桶数)
        """
        super().__init__()
        self.num_hash_types = num_hash_types
        self.embed_dim = embed_dim

        # 4种独立的哈希 Embedding 层
        self.hash_embeddings = nn.ModuleList([
            nn.Embedding(hash_vocab_size, embed_dim)
            for _ in range(num_hash_types)
        ])

        # 分类头: 始终接受 embed_total 维度输入
        embed_total = num_hash_types * embed_dim  # 4 × 64 = 256
        self.classifier = nn.Sequential(
            nn.Linear(embed_total, 128),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # 二分类 logits → CrossEntropyLoss
        )

        # 当输入维度与 Embedding 输出不匹配时, 用投影层适配到 embed_total
        if input_dim != embed_total:
            self.input_proj = nn.Linear(input_dim, embed_total)
        else:
            self.input_proj = None

    def forward(self, x):
        """
        前向传播

        参数:
            x: LongTensor(B, 4) 哈希索引 → 经过 Embedding
               或 FloatTensor(B, D) 预提取特征 → 投影后进入分类头
        """
        if x.dtype in (torch.long, torch.int):
            # 哈希索引输入: 经过4个独立 Embedding
            parts = [self.hash_embeddings[i](x[:, i]) for i in range(self.num_hash_types)]
            z = torch.cat(parts, dim=1)  # (B, 256)
        else:
            # 预提取特征输入: 投影到 Embedding 维度
            z = self.input_proj(x) if self.input_proj else x
        return self.classifier(z)
