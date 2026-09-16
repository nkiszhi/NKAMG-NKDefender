"""
MarkovCNN — Ni et al. 2018
论文: "Malware Identification Using Visualization Images and Deep Learning"
会议: Computers & Security
链接: https://doi.org/10.1016/j.cose.2018.09.001

论文方法: PE 字节 → 256×256 Markov 转移矩阵 → CNN

Markov 矩阵构建:
  M[i][j] = P(byte_t+1 = j | byte_t = i)
  统计所有相邻字节对的转移概率 → 256×256 矩阵

CNN 架构 (论文 Section 4.2):
  Conv(1→32, 5×5, stride=2) → BN → ReLU    (256→128)
  Conv(32→64, 3×3) → BN → ReLU → MaxPool(2) (128→64)
  Conv(64→128, 3×3) → BN → ReLU → MaxPool(2) (64→32)
  Conv(128→256, 3×3) → BN → ReLU
  GlobalAvgPool → FC(256→128→2)

特点: 第一层用 5×5 大卷积核捕获 Markov 矩阵的局部模式

输入: Markov 矩阵 Tensor(B, 1, 256, 256)
输出: (B, 2) 二分类 logits
"""
import torch.nn as nn


class MarkovCNN(nn.Module):
    """针对 256×256 Markov 转移矩阵设计的 CNN"""

    def __init__(self, input_dim=None):
        super().__init__()
        self.features = nn.Sequential(
            # 第1层: 5×5 大核捕获 Markov 局部模式
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            # 第2层: 3×3 标准卷积 + 池化
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 第3层: 进一步抽象
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 第4层: 高层特征
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 2),  # logits
        )

    def forward(self, x):
        if x.dim() == 2:
            H = int(x.shape[1] ** 0.5)
            x = x.reshape(x.size(0), 1, H, H)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.features(x).flatten(1)
        return self.classifier(z)
