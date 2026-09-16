"""
ALOHANet — Aghakhani et al. 2020
论文: "When Malware is Packin' Heat; Limits of Machine Learning Classifiers
       Based on Static Analysis Features"
会议: USENIX Security 2020
链接: https://www.usenix.org/conference/usenixsecurity20/presentation/aghakhani

论文原始架构: MLP(512, 256, 128) + L2正则化 + BatchNorm
  论文核心贡献: 分析对抗性特征空间攻击下的检测鲁棒性
  模型本身是标准 MLP，但训练时使用 L2 weight_decay 增强鲁棒性

输入: Import 表特征 (ImportInfo 1280d from EMBER)
输出: (B, 1) 恶意概率
"""
import torch.nn as nn


class ALOHANet(nn.Module):
    """ALOHA 对抗鲁棒恶意软件检测器 — 与论文 Section 5 架构一致"""

    def __init__(self, input_dim=1280):
        """
        参数:
            input_dim: 输入特征维度 (1280 = EMBER ImportInfo)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
