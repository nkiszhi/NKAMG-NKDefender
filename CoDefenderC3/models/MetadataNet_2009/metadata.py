"""
MetadataNet — 基于 Shafiq et al. 2009 (RAID)
PE 版本/资源/清单 元数据特征分类器

输入: EMBER ExportInfo(128d) + DataDirs(30d) = 158d
输出: (B, 1) 恶意概率
"""
import torch.nn as nn


class MetadataNet(nn.Module):
    def __init__(self, input_dim=158):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
