"""
DGCNN — Zhang et al. 2018
论文: "An End-to-End Deep Learning Architecture for Graph Classification"
会议: AAAI 2018
链接: https://doi.org/10.1609/aaai.v32i1.11782

架构: 4层 GCN → SortPooling(k=30) → 1D-CNN(16 filters) → MaxPool → FC → Sigmoid
关键创新: SortPooling 按节点特征最后一维排序, 截取 top-k 节点, 展平为固定长度向量

输入: CFG 图数据
输出: (B, 2) 二分类 logits (CrossEntropyLoss)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_PYG = False
try:
    from torch_geometric.nn import GCNConv, global_sort_pool
    _HAS_PYG = True
except ImportError:
    pass


class CFGDGCNN(nn.Module):
    """DGCNN — 含完整 SortPooling + 1D-CNN 管线"""

    def __init__(self, input_dim=22, node_dim=11, k=30, use_pyg=_HAS_PYG):
        super().__init__()
        self.use_pyg = use_pyg and _HAS_PYG
        self.k = k

        if self.use_pyg:
            # 4层 GCN 编码
            self.conv1 = GCNConv(node_dim, 32)
            self.conv2 = GCNConv(32, 32)
            self.conv3 = GCNConv(32, 32)
            self.conv4 = GCNConv(32, 1)  # 最后一层输出 1维, 用于 SortPooling 排序

            # 拼接所有层输出: 32+32+32+1 = 97 维
            total_dim = 32 * 3 + 1

            # 1D-CNN on sorted node features
            self.conv1d_1 = nn.Conv1d(1, 16, total_dim, stride=total_dim)
            self.conv1d_2 = nn.Conv1d(16, 32, 5, stride=1, padding=2)
            self.maxpool = nn.MaxPool1d(2)

            # 分类头
            fc_input = 32 * (k // 2)
            self.fc = nn.Sequential(
                nn.Linear(fc_input, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 2))
        else:
            pass  # no PyG layers
        # MLP 回退 — 始终创建, flat 输入时使用
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x, edge_index=None, batch=None):
        if self.use_pyg and edge_index is not None:
            # 4层 GCN, 收集每层输出
            x1 = torch.tanh(self.conv1(x, edge_index))
            x2 = torch.tanh(self.conv2(x1, edge_index))
            x3 = torch.tanh(self.conv3(x2, edge_index))
            x4 = torch.tanh(self.conv4(x3, edge_index))

            # 拼接所有层输出
            x_cat = torch.cat([x1, x2, x3, x4], dim=-1)  # (N, 97)

            # SortPooling: 按最后一维排序, 取 top-k
            x_sort = global_sort_pool(x_cat, batch, self.k)  # (B, k*97)

            # 1D-CNN
            x_sort = x_sort.unsqueeze(1)  # (B, 1, k*97)
            x_conv = F.relu(self.conv1d_1(x_sort))  # (B, 16, k)
            x_conv = self.maxpool(F.relu(self.conv1d_2(x_conv)))  # (B, 32, k//2)
            x_flat = x_conv.reshape(x_conv.size(0), -1)
            return self.fc(x_flat)
        return self.mlp(x)
