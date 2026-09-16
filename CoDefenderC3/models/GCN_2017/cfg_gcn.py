"""
CFGGCN — ICLR 2017
论文: "Semi-Supervised Classification with Graph Convolutional Networks"
链接: https://arxiv.org/abs/1609.02907

架构: 3层 GCN(64→32→16) → global_mean_pool → FC → Sigmoid

需要: torch_geometric (pip install torch-geometric)
当 torch_geometric 不可用时:
  使用 BinaryNinja 提取图统计特征 (22维) → MLP 分类
  这是降级方案，非论文原始架构

输入:
  torch_geometric 可用: PyG Batch (含 x, edge_index, batch)
  不可用: 图统计特征 Tensor(B, 22)
输出: (B, 2) 二分类 logits (CrossEntropyLoss)
"""
import torch
import torch.nn as nn

# 尝试导入 torch_geometric
_HAS_PYG = False
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.nn import global_mean_pool
    _HAS_PYG = True
except ImportError:
    pass


class CFGGCN(nn.Module):
    """
    3层 GCN(64→32→16) → global_mean_pool → FC → Sigmoid

    当 torch_geometric 不可用时回退到 MLP on 图统计特征
    """

    def __init__(self, input_dim=22, use_pyg=_HAS_PYG):
        """
        参数:
            input_dim: 图统计特征维度 (仅 MLP 模式使用)
            use_pyg: 是否使用 torch_geometric 完整 GNN
        """
        super().__init__()
        self.use_pyg = use_pyg and _HAS_PYG

        if self.use_pyg:
            # 完整 GNN 实现 (需要 torch_geometric)
            self.conv1 = GCNConv(11, 64)   # ACFG 节点特征 11维
            self.conv2 = GCNConv(64, 32)
            self.conv3 = GCNConv(32, 16)
            self.gnn_classifier = nn.Sequential(
                nn.Linear(16, 2))
        # MLP 回退 (图统计特征) — 始终创建, flat 输入时使用
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x, edge_index=None, batch=None):
        """
        前向传播

        GNN 模式:
            x: 节点特征 (N_total, 11)
            edge_index: 边索引 (2, E_total)
            batch: 节点-图映射 (N_total,)
        MLP 模式:
            x: 图统计特征 Tensor(B, 22)
        """
        if self.use_pyg and edge_index is not None:
            import torch.nn.functional as F
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))
            x = global_mean_pool(x, batch)
            return self.gnn_classifier(x)
        else:
            return self.mlp(x)
