"""
DL4MD — Hardy et al. 2016
论文: "DL4MD: A Deep Learning Framework for Intelligent Malware Detection"
会议: International Conference on Data Mining (DMIN)
链接: https://doi.org/10.1109/DMIN.2016.9

论文原始架构: Stacked Denoising Autoencoder (SDAE)
  预训练阶段 (逐层贪婪):
    Layer 1: input → 400, 加噪 → 重构 → 学习 W1
    Layer 2: 400 → 300, 加噪 → 重构 → 学习 W2
    Layer 3: 300 → 200, 加噪 → 重构 → 学习 W3
    Layer 4: 200 → 100, 加噪 → 重构 → 学习 W4
  微调阶段:
    编码器(W1-W4) + 分类头 → 端到端 BCE 训练

本实现包含:
  1. DenoisingAutoencoder: 单层去噪自编码器 (用于预训练)
  2. DL4MD: 完整的堆叠模型 (可选预训练 + 端到端微调)

输入: EMBER PE 结构特征 (327d 或 2381d)
输出: (B, 1) 恶意概率
"""
import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    """单层去噪自编码器 — 用于逐层预训练"""

    def __init__(self, input_dim, hidden_dim, noise_rate=0.2):
        """
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            noise_rate: 去噪率 (Dropout 概率)
        """
        super().__init__()
        self.noise = nn.Dropout(noise_rate)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """前向传播: 加噪 → 编码 → 解码"""
        noisy = self.noise(x)
        encoded = self.encoder(noisy)
        decoded = self.decoder(encoded)
        return decoded, encoded


class DL4MD(nn.Module):
    """
    DL4MD — Stacked Denoising Autoencoder 恶意软件检测器

    论文架构: input → 400 → 300 → 200 → 100 → classifier(1)
    每层使用 Dropout 去噪 (noise_rate=0.2)
    """

    def __init__(self, input_dim=327, layer_dims=None, noise_rate=0.2):
        """
        参数:
            input_dim: 输入特征维度
            layer_dims: 各层维度列表, 默认 [400, 300, 200, 100]
            noise_rate: 去噪率
        """
        super().__init__()
        if layer_dims is None:
            layer_dims = [400, 300, 200, 100]

        # 构建堆叠编码器
        dims = [input_dim] + layer_dims
        encoder_layers = []
        for i in range(len(layer_dims)):
            encoder_layers.extend([
                nn.Dropout(noise_rate),  # 去噪
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(inplace=True),
            ])
        self.encoder = nn.Sequential(*encoder_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(layer_dims[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 逐层预训练用的自编码器 (可选)
        self.daes = nn.ModuleList([
            DenoisingAutoencoder(dims[i], dims[i + 1], noise_rate)
            for i in range(len(layer_dims))
        ])

    def pretrain_layer(self, layer_idx, x, epochs=10, lr=1e-3):
        """
        预训练第 layer_idx 层的去噪自编码器，并将权重迁移到编码器

        参数:
            layer_idx: 层索引 (0-3)
            x: 该层的输入数据
            epochs: 预训练轮数
            lr: 学习率
        """
        dae = self.daes[layer_idx]
        optimizer = torch.optim.Adam(dae.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            decoded, encoded = dae(x)
            loss = criterion(decoded, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 将 DAE 编码器权重迁移到主编码器对应层
        # 编码器每层由 3 个子模块组成: Dropout, Linear, ReLU
        # 第 layer_idx 层的 Linear 在索引 layer_idx*3 + 1
        enc_linear_idx = layer_idx * 3 + 1
        with torch.no_grad():
            self.encoder[enc_linear_idx].weight.copy_(dae.encoder[0].weight)
            self.encoder[enc_linear_idx].bias.copy_(dae.encoder[0].bias)

        return encoded.detach()

    def forward(self, x):
        """
        前向传播: 堆叠编码 → 分类

        参数:
            x: Tensor(B, input_dim)
        返回:
            Tensor(B, 1), 恶意概率
        """
        z = self.encoder(x)
        return self.classifier(z)
