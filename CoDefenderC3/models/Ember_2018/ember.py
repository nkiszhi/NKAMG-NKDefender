"""
Ember — Anderson & Roth 2018
"EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models"
arXiv:1804.04637

原文模型为 LightGBM GBDT, 本实现使用 Neural Decision Forest:
  可微分软决策树集成 (Neural Decision Forest, Kontschieder et al. 2015)
  通过 PyTorch 实现端到端可训练

Architecture:
  Input: 2381-d EMBER static feature vector (9个特征组)
  Feature Embedding: Linear → BN → ReLU → Linear
  Soft Decision Trees × num_trees
  Ensemble averaging → Sigmoid → output

特征组 (总共 2381 维):
  ByteHistogram (256d), ByteEntropy (256d), StringInfo (104d),
  GeneralInfo (10d), HeaderInfo (62d), SectionInfo (255d),
  ImportInfo (1280d), ExportInfo (128d), DataDirectories (30d)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDecisionTree(nn.Module):
    """可微分软决策树 — 基于 Kontschieder et al. 2015"""
    def __init__(self, input_dim, depth=3):
        super(SoftDecisionTree, self).__init__()
        self.depth = depth
        self.num_internal = 2 ** depth - 1
        self.num_leaves = 2 ** depth

        # 内部节点的决策函数
        self.decision = nn.Linear(input_dim, self.num_internal)
        # 叶子节点的类别概率
        self.leaf_probs = nn.Parameter(torch.randn(self.num_leaves, 1))

    def forward(self, x):
        """
        Args:
            x: (B, D) feature vector
        Returns:
            (B, 1) prediction
        """
        batch_size = x.size(0)
        decisions = torch.sigmoid(self.decision(x))  # (B, num_internal)

        # 计算每个叶子节点的到达概率
        mu = torch.ones(batch_size, 1, device=x.device)  # (B, 1)
        begin_idx = 0
        end_idx = 1

        for d in range(self.depth):
            node_decisions = decisions[:, begin_idx:end_idx]  # (B, 2^d)
            mu_left = mu * node_decisions
            mu_right = mu * (1 - node_decisions)
            # 交错合并 left/right
            mu = torch.stack([mu_left, mu_right], dim=2).reshape(batch_size, -1)
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (d + 1)

        # mu: (B, num_leaves) — 到达概率
        leaf_values = torch.sigmoid(self.leaf_probs).squeeze(-1)  # (num_leaves,)
        out = (mu * leaf_values.unsqueeze(0)).sum(dim=1, keepdim=True)
        return out


class Ember(nn.Module):
    """
    Neural Decision Forest for EMBER features.
    模拟 LightGBM 的 GBDT 行为，但可端到端训练。
    """
    def __init__(self, input_dim=2381, num_trees=10, tree_depth=3, **kwargs):
        super(Ember, self).__init__()

        # 特征嵌入网络
        hidden_dim = 256
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 软决策树森林
        self.trees = nn.ModuleList([
            SoftDecisionTree(hidden_dim, depth=tree_depth)
            for _ in range(num_trees)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) EMBER feature vector
        Returns:
            (B, 1) malware probability
        """
        features = self.feature_net(x)  # (B, hidden_dim)

        # 集成所有树的预测
        tree_outputs = [tree(features) for tree in self.trees]
        ensemble = torch.stack(tree_outputs, dim=0).mean(dim=0)  # (B, 1)

        return ensemble
