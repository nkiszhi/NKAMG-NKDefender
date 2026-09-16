"""
InceptionV3 for Malware Classification — 2020
基于 Nataraj et al. 2011 的恶意软件可视化方法,
使用 InceptionV3 (Szegedy et al. 2016, CVPR) 预训练模型进行迁移学习

Architecture:
  Input: PE binary → grayscale image → 3-channel repeat → 299×299
  InceptionV3 (ImageNet pretrained)
  - 替换最后 FC 层: fc → Linear(2048, num_classes)
  - 去除 AuxLogits

参考论文:
  Nataraj et al. 2011 "Malware Images: Visualization and Automatic Classification"
  Szegedy et al. 2016 "Rethinking the Inception Architecture for Computer Vision"
"""
import torch
import torch.nn as nn


class InceptionV3Model(nn.Module):
    """
    InceptionV3 迁移学习模型
    加载 ImageNet 预训练权重，替换最后分类层
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(InceptionV3Model, self).__init__()

        try:
            import torchvision.models as models
            if pretrained:
                try:
                    weights = models.Inception_V3_Weights.IMAGENET1K_V1
                    self.inception_v3 = models.inception_v3(weights=weights)
                except AttributeError:
                    self.inception_v3 = models.inception_v3(pretrained=True)
            else:
                self.inception_v3 = models.inception_v3(pretrained=False)
        except ImportError:
            raise ImportError("InceptionV3Model requires torchvision")

        # 替换最后全连接层
        in_features = self.inception_v3.fc.in_features
        self.inception_v3.fc = nn.Linear(in_features, num_classes)

        # 去除辅助分类器 (可选, 在 run 脚本中控制)
        # self.inception_v3.AuxLogits = None

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) grayscale, (B, 3, H, W) color, or flat (B, H*W)
        Returns:
            logits (B, num_classes) or (logits, aux_logits) if training with AuxLogits
        """
        if x.dim() == 2:
            H = int(x.shape[1] ** 0.5)
            x = x.reshape(x.size(0), 1, H, H)
        # 灰度→3通道 (Nataraj-style: repeat grayscale to match ImageNet 3ch input)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        # InceptionV3 要求最小 299×299 输入
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return self.inception_v3(x)
