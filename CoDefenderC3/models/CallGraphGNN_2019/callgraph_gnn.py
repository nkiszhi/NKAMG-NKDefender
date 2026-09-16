"""
CallGraphGNN — 基于 Li et al. 2019 Graph Matching Networks
论文: "Graph Matching Networks for Learning the Similarity of Graph Structured Objects"
会议: ICML 2019
链接: https://arxiv.org/abs/1904.12787

论文核心: Graph Matching Network (GMN) 含交叉图注意力
  1. 图编码: GNN (GraphSAGE/GCN) 编码节点
  2. 交叉注意力: 两图节点间软对齐 (cross-graph attention)
  3. 图级表示: 注意力加权聚合
  4. 相似度: 图级向量距离 → 匹配分数

在恶意软件检测场景:
  不做图对匹配, 而是用带注意力聚合的 GNN 分类
  保留论文的 注意力聚合机制 作为核心组件

输入: BinaryNinja 函数调用图 (节点特征 + 边)
输出: (B, 2) 二分类 logits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_PYG = False
try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    _HAS_PYG = True
except ImportError:
    pass


class GraphAttentionAggregator(nn.Module):
    """
    图级注意力聚合 — 来自 GMN 论文 Equation (5)

    对每个节点计算注意力权重, 加权聚合为图级表示:
      α_i = softmax(w · tanh(W·h_i))
      h_G = Σ α_i · h_i
    """

    def __init__(self, node_dim, attn_dim=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(node_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1))

    def forward(self, x, batch):
        """
        参数:
            x: (N_total, node_dim) 所有节点特征
            batch: (N_total,) 节点-图映射
        返回:
            (B, node_dim) 图级表示
        """
        gate_scores = self.gate(x)  # (N, 1)

        # 按图分组 softmax
        from torch_geometric.utils import softmax as pyg_softmax
        attn = pyg_softmax(gate_scores, batch)  # (N, 1)

        # 加权聚合
        weighted = x * attn  # (N, D)
        from torch_geometric.nn import global_add_pool
        return global_add_pool(weighted, batch)


class CallGraphGNN(nn.Module):
    """
    带注意力聚合的 GraphSAGE 恶意软件检测器

    架构: SAGEConv(2层) → GraphAttentionAggregator → FC(2)
    """

    def __init__(self, input_dim=22, node_dim=11, use_pyg=_HAS_PYG):
        super().__init__()
        self.use_pyg = use_pyg and _HAS_PYG

        if self.use_pyg:
            self.sage1 = SAGEConv(node_dim, 64)
            self.sage2 = SAGEConv(64, 64)
            self.attn_pool = GraphAttentionAggregator(64)
            self.gnn_fc = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(32, 2))
        # MLP 回退 — 始终创建, flat 输入时使用
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x, edge_index=None, batch=None):
        if self.use_pyg and edge_index is not None:
            x = F.relu(self.sage1(x, edge_index))
            x = F.relu(self.sage2(x, edge_index))
            x = self.attn_pool(x, batch)  # 注意力聚合 (非简单 mean pool)
            return self.gnn_fc(x)
        return self.mlp(x)
