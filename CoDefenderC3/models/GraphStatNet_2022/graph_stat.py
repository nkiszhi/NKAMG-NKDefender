"""
GraphStatNet — 基于图统计特征的恶意软件分类器
用于 torch_geometric 不可用时作为 GNN 的降级替代

22 维图统计特征 (由 BinaryNinja 提取):
  CFG 特征 (11维):
    [0] 基本块数量          [1] 边数量            [2] 平均出度
    [3] 最大出度            [4] 连通分量数        [5] 图密度
    [6] 传递性 (clustering) [7] 平均路径长度      [8] 叶子节点数
    [9] 入度方差            [10] 出度方差

  CallGraph 特征 (11维):
    [11] 函数总数           [12] 外部调用数        [13] 内部调用数
    [14] 调用图密度         [15] 最大扇入          [16] 最大扇出
    [17] 平均扇出           [18] 递归函数数        [19] 孤立函数数
    [20] 外部调用比率       [21] 叶子函数比率

非论文模型, 作为工程降级方案。分类器使用 BatchNorm + 残差连接增强。

输入: Tensor(B, 22)
输出: (B, 2) 二分类 logits
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """带残差连接的 MLP 块"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + x)  # 残差连接


class GraphStatNet(nn.Module):
    """
    图统计特征分类器 — 带残差连接

    架构: Linear(22→128) → BN → ReLU
        → ResidualBlock(128) × 2
        → Linear(128→64) → ReLU → Dropout → Linear(64→2)
    """

    def __init__(self, input_dim=22):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        self.res_blocks = nn.Sequential(
            ResidualBlock(128), ResidualBlock(128))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 2))

    def forward(self, x):
        z = self.input_proj(x)
        z = self.res_blocks(z)
        return self.classifier(z)
