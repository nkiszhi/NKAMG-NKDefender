"""
IMCFN — Vasan et al. 2020
"IMCFN: Image-based Malware Classification using Fine-tuned
 Convolutional Neural Network Architecture"
Computer Networks, 171, 107138

Architecture:
  基于 VGG16 的迁移学习:
  - Block 1-4: 冻结 VGG16 ImageNet 预训练权重
  - Block 5: 微调 (小学习率)
  - Classifier: FC(25088→4096) → ReLU → Dropout →
                FC(4096→4096) → ReLU → Dropout → FC(4096→num_classes)

输入: PE 二进制 → 彩色可视化图 (jet colormap) → 224×224×3
     ImageNet normalize (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
"""
import torch
import torch.nn as nn


class IMCFN(nn.Module):
    """
    IMCFN: VGG16 架构 (完全手写，不依赖 torchvision 权重加载)
    支持 freeze_until_block 参数控制冻结层级
    """
    def __init__(self, num_classes=2, freeze_until_block=4):
        super(IMCFN, self).__init__()

        # VGG16 卷积块
        self.block1 = self._make_block(3, 64, 2)
        self.block2 = self._make_block(64, 128, 2)
        self.block3 = self._make_block(128, 256, 3)
        self.block4 = self._make_block(256, 512, 3)
        self.block5 = self._make_block(512, 512, 3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 分类头 (与 VGG16 classifier 对齐)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.classifier = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(0.5)

        # 冻结策略:
        # 若有预训练权重 (通过 load_state_dict 加载), 冻结 Block 1~freeze_until_block
        # 若从零训练 (无预训练权重), 不冻结任何层 (否则冻结随机权重无意义)
        # 调用者应在加载预训练权重后手动调用 freeze_blocks()
        self._freeze_until = freeze_until_block

    def freeze_blocks(self, until_block=None):
        """加载预训练权重后调用此方法冻结低层"""
        n = until_block if until_block is not None else self._freeze_until
        blocks = [self.block1, self.block2, self.block3, self.block4, self.block5]
        for i in range(min(n, 5)):
            for param in blocks[i].parameters():
                param.requires_grad = False

    @staticmethod
    def _make_block(in_ch, out_ch, n_convs):
        """VGG 卷积块: n_convs 个 Conv2d(3×3) + BN + ReLU"""
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) color image or flat (B, H*W*3)
        Returns:
            (B, num_classes) logits
        """
        if x.dim() == 2:
            # flat (B, D) → (B, 3, H, W)
            H = int((x.shape[1] // 3) ** 0.5)
            x = x.reshape(x.size(0), H, H, 3).permute(0, 3, 1, 2)  # (B,H,W,3)→(B,3,H,W)
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.pool(self.block4(x))
        x = self.pool(self.block5(x))

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)

        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.classifier(x)

        return x
