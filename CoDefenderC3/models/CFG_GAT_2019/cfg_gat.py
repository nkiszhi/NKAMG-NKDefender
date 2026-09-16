"""
CFG_GAT — Yan et al. 2019
论文: "Classifying Malware Represented as Control Flow Graphs using Deep Graph CNN"
会议: IEEE DSC 2019
链接: https://doi.org/10.1109/DSC47296.2019

架构: 3层 GAT(heads=[8,8,1], hidden=[64,64,32]) → global_mean_pool → FC → Sigmoid
输入: BinaryNinja 提取的 CFG 图 (节点特征 11维)
输出: (B, 2) 二分类 logits (CrossEntropyLoss)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_PYG = False
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    _HAS_PYG = True
except ImportError:
    pass


class CFGGAT(nn.Module):
    """3层 Graph Attention Network — 与论文架构一致"""

    def __init__(self, input_dim=22, node_dim=11, use_pyg=_HAS_PYG):
        super().__init__()
        self.use_pyg = use_pyg and _HAS_PYG

        if self.use_pyg:
            # 论文架构: 3层 GAT, multi-head attention
            self.gat1 = GATConv(node_dim, 64, heads=8, concat=False, dropout=0.3)
            self.gat2 = GATConv(64, 64, heads=8, concat=False, dropout=0.3)
            self.gat3 = GATConv(64, 32, heads=1, concat=False, dropout=0.3)
            self.gnn_fc = nn.Sequential(nn.Linear(32, 2))  # logits → CrossEntropyLoss
        # MLP 回退 (图统计特征, 非论文原始算法) — 始终创建, flat 输入时使用
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x, edge_index=None, batch=None):
        if self.use_pyg and edge_index is not None:
            x = F.elu(self.gat1(x, edge_index))
            x = F.elu(self.gat2(x, edge_index))
            x = F.elu(self.gat3(x, edge_index))
            x = global_mean_pool(x, batch)
            return self.gnn_fc(x)
        return self.mlp(x)
