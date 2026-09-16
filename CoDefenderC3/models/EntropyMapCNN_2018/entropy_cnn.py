"""
EntropyMapCNN — Conti et al. 2018
论文: "Visual Reverse Engineering of Binary and Data Files"
会议: Visualization for Cyber Security (VizSec)
链接: https://doi.org/10.1007/978-3-319-73198-8_1

论文方法: PE 文件 → 滑动窗口 Shannon 熵 → 256×256 熵热图 → CNN

熵热图构建:
  将文件分为 256×256 个块，每块计算 Shannon 熵: H = -Σ p_i·log(p_i)
  熵值范围 [0, 8]，归一化到 [0, 1]

CNN 架构: 针对熵热图的平滑特性设计
  Conv(1→16, 7×7, stride=2) → BN → ReLU  (大核捕获熵变化趋势)
  Conv(16→32, 5×5) → BN → ReLU → MaxPool(2)
  Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)
  Conv(64→128, 3×3) → BN → ReLU
  GlobalAvgPool → FC(128→64→2)

特点: 前两层用大卷积核 (7×7, 5×5) 捕获熵的空间渐变模式

输入: 熵热图 Tensor(B, 1, 256, 256)
输出: (B, 2) 二分类 logits
"""
import torch.nn as nn


class EntropyMapCNN(nn.Module):
    """针对 256×256 Shannon 熵热图设计的 CNN"""

    def __init__(self, input_dim=None):
        super().__init__()
        self.features = nn.Sequential(
            # 第1层: 7×7 大核捕获熵的空间渐变趋势
            nn.Conv2d(1, 16, 7, stride=2, padding=3),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),

            # 第2层: 5×5 中核
            nn.Conv2d(16, 32, 5, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 第3层
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 第4层
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        if x.dim() == 2:
            H = int(x.shape[1] ** 0.5)
            x = x.reshape(x.size(0), 1, H, H)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x).flatten(1))
