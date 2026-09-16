"""
GrayscaleCNN — Nataraj et al. 2011
论文: "Malware Images: Visualization and Automatic Classification"
会议: VizSec 2011
链接: https://doi.org/10.1145/2016904.2016908

论文原始方法: PE 字节 → 灰度图 → GIST 纹理描述子(320d) → kNN
本实现: PE 灰度图 → 可学习 GIST-like 多尺度 Gabor 滤波器 → 全局池化 → 分类器

GIST 描述子包含:
  - 4个尺度 (scale), 每个尺度 8 个方向的 Gabor 滤波
  - 每个方向在 4×4 空间网格上平均 → 4×8×16 = 512d (论文简化为320d)

本实现用可学习的 Gabor-like 卷积模拟 GIST:
  GaborBank: 4个尺度 × 8个方向 = 32 个滤波器
  SpatialPooling: 4×4 网格平均
  → 32×16 = 512d → FC(512→128→2) → logits

输入: 灰度图 Tensor(B, 1, H, W)
输出: (B, 2) 二分类 logits → CrossEntropyLoss
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaborConv2d(nn.Module):
    """
    Gabor 滤波器卷积层 — 模拟 GIST 描述子的多尺度多方向滤波

    Gabor 滤波器: g(x,y) = exp(-x'^2+γ²y'^2 / 2σ²) × cos(2π·x'/λ + ψ)
      x' = x·cosθ + y·sinθ
      y' = -x·sinθ + y·cosθ
    """

    def __init__(self, n_scales=4, n_orientations=8, kernel_size=31):
        """
        参数:
            n_scales: Gabor 尺度数 (论文使用 4)
            n_orientations: 每个尺度的方向数 (论文使用 8)
            kernel_size: 滤波器大小
        """
        super().__init__()
        self.n_filters = n_scales * n_orientations

        # 预生成 Gabor 滤波器组，然后作为可学习参数微调
        filters = []
        for s in range(n_scales):
            sigma = 2.0 * (s + 1)       # 尺度参数
            lambd = 4.0 * (s + 1)       # 波长
            for o in range(n_orientations):
                theta = o * math.pi / n_orientations  # 方向角

                # 生成 Gabor 核
                half = kernel_size // 2
                y, x = torch.meshgrid(
                    torch.arange(-half, half + 1, dtype=torch.float32),
                    torch.arange(-half, half + 1, dtype=torch.float32),
                    indexing="ij")
                x_theta = x * math.cos(theta) + y * math.sin(theta)
                y_theta = -x * math.sin(theta) + y * math.cos(theta)
                gb = torch.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2)) * \
                     torch.cos(2 * math.pi * x_theta / lambd)
                # 归一化
                gb = gb / (gb.norm() + 1e-8)
                filters.append(gb)

        # (n_filters, 1, K, K) — 可学习微调
        weight = torch.stack(filters).unsqueeze(1)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(self.n_filters))

    def forward(self, x):
        """
        参数:
            x: (B, 1, H, W) 灰度图
        返回:
            (B, n_filters, H', W') Gabor 响应图
        """
        return F.conv2d(x, self.weight, self.bias, padding=self.weight.shape[-1] // 2)


class GrayscaleCNN(nn.Module):
    """
    Nataraj 2011 GIST-like CNN 恶意软件可视化检测器

    架构: Gabor滤波器组(32) → ReLU → AdaptiveAvgPool(4×4)
         → Flatten(512) → FC(512→128→2)
    """

    def __init__(self, in_channels=1, n_scales=4, n_orientations=8, input_dim=None):
        super().__init__()
        n_filters = n_scales * n_orientations  # 32

        # GIST-like 多尺度多方向 Gabor 滤波
        self.gabor = GaborConv2d(n_scales, n_orientations, kernel_size=31)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(n_filters)

        # 空间网格池化 (论文: 4×4 grid → 每格平均)
        self.spatial_pool = nn.AdaptiveAvgPool2d(4)  # → (B, 32, 4, 4) = 512d

        # 分类头
        feat_dim = n_filters * 4 * 4  # 512
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),  # 二分类 logits → CrossEntropyLoss
        )

    def forward(self, x):
        """
        参数:
            x: Tensor(B, 1, H, W), (B, H, W), 或 flat (B, H*W) 灰度图
        返回:
            Tensor(B, 2) 二分类 logits
        """
        if x.dim() == 2:
            # flat (B, D) → (B, 1, H, W)
            H = int(x.shape[1] ** 0.5)
            x = x.reshape(x.size(0), 1, H, H)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.relu(self.bn(self.gabor(x)))   # Gabor 响应
        z = self.spatial_pool(z)                 # 4×4 空间网格池化
        z = z.flatten(1)                         # (B, 512)
        return self.classifier(z)
