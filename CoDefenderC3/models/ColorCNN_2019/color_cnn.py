"""
ColorCNN — Liu & Wang 2019
论文: "A New Method for PE Malware Detection Based on Deep Learning with Visualization"

论文方法: PE → RGB 彩色图像 → VGG-like CNN (3个卷积块, 每块2层conv)
  与 GrayscaleCNN 的区别: 3通道输入、独立架构、无 GIST

架构 (论文 Section 3.3):
  Block1: Conv(3→32, 3×3) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool(2)
  Block2: Conv(32→64, 3×3) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool(2)
  Block3: Conv(64→128, 3×3) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool(2)
  Global AvgPool → FC(128→64→2)

输入: RGB PE 图像 Tensor(B, 3, H, W)
输出: (B, 2) 二分类 logits
"""
import torch.nn as nn


class ColorCNN(nn.Module):
    """VGG-style RGB 恶意软件图像分类器 — 独立架构"""

    def __init__(self, in_channels=3, input_dim=None):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 2))

    def forward(self, x):
        if x.dim() == 2:
            H = int((x.shape[1] // 3) ** 0.5)
            x = x.reshape(x.size(0), H, H, 3).permute(0, 3, 1, 2)
        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)
        z = self.gap(z).flatten(1)
        return self.classifier(z)
