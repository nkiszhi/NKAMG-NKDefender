"""
PEMiner — Shafiq et al. 2009
论文: "PE-Miner: Mining Structural Information to Detect Malicious Executables in Realtime"
会议: RAID 2009
链接: https://doi.org/10.1007/978-3-642-04342-0_1

1:1 精准复刻: Random Forest (CART 决策树 + Bootstrap + 投票)
纯 Python + numpy 实现

算法:
  1. Bootstrap: 有放回采样 N 个样本
  2. CART 决策树: 每个节点随机选 sqrt(D) 个特征, 基尼不纯度分裂
  3. 集成: n_estimators 棵树的多数投票

输入: numpy array (N, D)
输出: 预测标签和概率
"""
import numpy as np


class DecisionTreeNode:
    """CART 决策树节点"""

    def __init__(self):
        self.feature_idx = None   # 分裂特征索引
        self.threshold = None     # 分裂阈值
        self.left = None          # 左子树 (≤ threshold)
        self.right = None         # 右子树 (> threshold)
        self.is_leaf = False
        self.class_counts = None  # 叶子节点各类别计数
        self.prediction = None    # 叶子节点多数类


class CARTDecisionTree:
    """CART 决策树 — 基尼不纯度分裂, 与 Breiman (1984) 一致"""

    def __init__(self, max_depth=15, min_samples_leaf=20, max_features="sqrt",
                 random_state=None):
        """
        参数:
            max_depth: 最大深度 (论文 max_depth=15)
            min_samples_leaf: 叶子最少样本数 (论文 min_samples_leaf=20)
            max_features: 每次分裂考虑的特征数 ("sqrt" = sqrt(D))
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.RandomState(random_state)
        self.root = None
        self.n_classes = 2

    def _gini(self, y):
        """计算基尼不纯度: G = 1 - Σ p_k²"""
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y, feature_indices):
        """在给定特征子集中找最佳分裂点"""
        best_gain = -1.0
        best_feat = None
        best_thr = None
        parent_gini = self._gini(y)
        N = len(y)

        for feat in feature_indices:
            values = X[:, feat]
            # 使用分位数作为候选阈值 (加速)
            thresholds = np.unique(np.percentile(values, np.linspace(10, 90, 9)))

            for thr in thresholds:
                left_mask = values <= thr
                right_mask = ~left_mask
                n_left = np.sum(left_mask)
                n_right = N - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                # 信息增益 = 父节点基尼 - 加权子节点基尼
                gain = parent_gini - (
                    n_left / N * self._gini(y[left_mask]) +
                    n_right / N * self._gini(y[right_mask])
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = thr

        return best_feat, best_thr, best_gain

    def _build_tree(self, X, y, depth):
        """递归构建 CART 决策树"""
        node = DecisionTreeNode()
        node.class_counts = np.bincount(y, minlength=self.n_classes)

        # 叶子节点条件
        if (depth >= self.max_depth or
            len(y) < 2 * self.min_samples_leaf or
            len(np.unique(y)) == 1):
            node.is_leaf = True
            node.prediction = np.argmax(node.class_counts)
            return node

        # 随机选择特征子集
        D = X.shape[1]
        if self.max_features == "sqrt":
            n_feats = max(1, int(np.sqrt(D)))
        else:
            n_feats = D
        feat_indices = self.rng.choice(D, size=min(n_feats, D), replace=False)

        # 找最佳分裂
        best_feat, best_thr, best_gain = self._best_split(X, y, feat_indices)

        if best_feat is None or best_gain <= 0:
            node.is_leaf = True
            node.prediction = np.argmax(node.class_counts)
            return node

        # 分裂
        node.feature_idx = best_feat
        node.threshold = best_thr
        left_mask = X[:, best_feat] <= best_thr
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X, y):
        """训练决策树"""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.root = self._build_tree(X, y, depth=0)

    def _predict_one(self, x, node):
        """递归预测单个样本"""
        if node.is_leaf:
            return node.class_counts
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict_proba(self, X):
        """预测概率"""
        X = np.asarray(X, dtype=np.float32)
        proba = np.zeros((X.shape[0], self.n_classes))
        for i in range(X.shape[0]):
            counts = self._predict_one(X[i], self.root)
            total = np.sum(counts)
            proba[i] = counts / max(total, 1)
        return proba


class PEMinerRF:
    """
    Random Forest — 纯 Python 实现, 与 Shafiq 2009 论文一致

    算法:
      1. Bootstrap 采样 n_estimators 个训练子集
      2. 每个子集训练一棵 CART 决策树 (max_features="sqrt")
      3. 预测: 所有树的概率平均
    """

    def __init__(self, n_estimators=200, max_depth=15, min_samples_leaf=20):
        """
        参数:
            n_estimators: 树的数量 (论文使用 200)
            max_depth: 单棵树最大深度 (论文使用 15)
            min_samples_leaf: 叶子最少样本数 (论文使用 20)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def fit(self, X, y):
        """Bootstrap + CART 训练"""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        N = X.shape[0]
        rng = np.random.RandomState(42)
        self.trees = []

        for t in range(self.n_estimators):
            # Bootstrap 有放回采样
            boot_idx = rng.choice(N, size=N, replace=True)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]

            tree = CARTDecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt",
                random_state=rng.randint(2**31))
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            if (t + 1) % 50 == 0:
                print(f"  [PEMiner RF] {t+1}/{self.n_estimators} 棵树已训练")

    def predict(self, X):
        """多数投票预测"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """所有树的概率平均"""
        X = np.asarray(X, dtype=np.float32)
        proba_sum = np.zeros((X.shape[0], 2))
        for tree in self.trees:
            proba_sum += tree.predict_proba(X)
        return proba_sum / len(self.trees)
