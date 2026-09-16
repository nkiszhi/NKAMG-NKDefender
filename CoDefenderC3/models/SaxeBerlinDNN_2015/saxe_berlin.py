"""
SaxeBerlinDNN — Saxe & Berlin 2015
论文: "Deep Neural Network Based Malware Detection Using Two-Dimensional Binary Visualization"
会议: IEEE Intelligence and Security Informatics (ISI)
链接: https://doi.org/10.1109/ISI.2015.7165944

论文原始架构: 4层全连接网络 + BatchNorm + Dropout
  输入 → FC(1024) → BN → ReLU → Dropout(0.5)
       → FC(512) → BN → ReLU → Dropout(0.5)
       → FC(256) → BN → ReLU → Dropout(0.3)
       → FC(1) → Sigmoid

本实现与原论文完全一致 — 论文本身就是深度神经网络
输入: EMBER 2381维特征向量 或 ByteHistogram+ByteEntropy 512维子集
输出: (B, 1) 恶意概率
"""
import torch.nn as nn


class SaxeBerlinDNN(nn.Module):
    """Saxe & Berlin 2015 深度恶意软件检测网络 — 与论文架构完全一致"""

    def __init__(self, input_dim=512):
        """
        参数:
            input_dim: 输入特征维度 (512 = ByteHist+ByteEntropy, 2381 = 全EMBER)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: Tensor(B, input_dim), EMBER 静态特征
        返回:
            Tensor(B, 1), 恶意概率 [0, 1]
        """
        return self.net(x)
