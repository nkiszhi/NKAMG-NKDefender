"""
RaffFeatureNet — Raff et al. 2018 MalConv 论文中的特征工程基线
论文: "Malware Detection by Eating a Whole EXE"
链接: https://arxiv.org/abs/1710.09435

论文原始架构: 2层 MLP(256, 128)
  用于证明端到端 MalConv 优于传统特征工程方法
  输入是手工提取的字符串统计特征

输入: StringInfo 特征 (104d from EMBER)
输出: (B, 1) 恶意概率
"""
import torch.nn as nn


class RaffFeatureNet(nn.Module):
    """MalConv 论文中的特征基线 — 与论文 Section 4.1 一致"""

    def __init__(self, input_dim=104):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
