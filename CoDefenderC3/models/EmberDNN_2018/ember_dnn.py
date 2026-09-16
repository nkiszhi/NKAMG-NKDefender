"""
EmberDNN — Anderson & Roth 2018 EMBER 论文中的 DNN 基线
论文: "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models"
链接: https://arxiv.org/abs/1804.04637

论文原始架构: 2层全连接网络 (论文使用全部2381维输入)
  FC(300) → BN → ReLU → Dropout(0.5) → FC(300) → BN → ReLU → Dropout(0.5) → FC(1) → Sigmoid

本实现: 网络架构与论文一致, 输入维度按视角组调整
  V2视角 (ByteHist+ByteEntropy): input_dim=512
  V11视角 (全EMBER特征): input_dim=2381
输入: EMBER 2381维特征向量
输出: (B, 1) 恶意概率
"""
import torch.nn as nn


class EmberDNN(nn.Module):
    """EMBER DNN 基线模型 — 与论文 Section 4.2 一致"""

    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(300, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(300, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
