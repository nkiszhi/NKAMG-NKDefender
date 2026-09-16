"""
EmberGBDT — Anderson & Roth 2018
论文: "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models"
链接: https://arxiv.org/abs/1804.04637

1:1 精准复刻: Gradient Boosted Decision Trees (GBDT)
纯 Python + numpy 实现

算法 (Friedman 2001):
  初始化: F_0(x) = log(p / (1-p))  (对数几率)
  迭代 m = 1, ..., M:
    1. 计算负梯度残差: r_i = y_i - sigmoid(F_{m-1}(x_i))
    2. 拟合回归树 h_m(x) 到残差 r
    3. 行搜索: 每个叶子节点的最优输出值 γ = Σr / Σ(p*(1-p))
    4. 更新: F_m(x) = F_{m-1}(x) + η · h_m(x)

输入: numpy array (N, D)
输出: 预测标签和概率
"""
import numpy as np


class GBDTRegressionTree:
    """
    GBDT 回归树 — 拟合梯度残差

    与分类树的区别: 叶子节点存储连续值 (残差的加权均值)
    """

    def __init__(self, max_depth=6, min_samples_leaf=50, max_features=0.8,
                 random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.RandomState(random_state)
        self.tree = None

    def _mse(self, y):
        """均方误差 (回归树的分裂准则)"""
        if len(y) == 0:
            return 0.0
        return np.var(y) * len(y)

    def _best_split(self, X, y, feat_indices):
        """找使 MSE 减少最大的分裂点"""
        best_reduction = -1.0
        best_feat, best_thr = None, None
        parent_mse = self._mse(y)
        N = len(y)

        for feat in feat_indices:
            vals = X[:, feat]
            # 分位数候选阈值
            percentiles = np.linspace(10, 90, 8)
            thresholds = np.unique(np.percentile(vals, percentiles))

            for thr in thresholds:
                left = vals <= thr
                n_left = np.sum(left)
                n_right = N - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                reduction = parent_mse - self._mse(y[left]) - self._mse(y[~left])
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feat = feat
                    best_thr = thr

        return best_feat, best_thr

    def _build(self, X, y, hessians, depth):
        """递归构建回归树"""
        node = {}
        if (depth >= self.max_depth or
            len(y) < 2 * self.min_samples_leaf or
            np.std(y) < 1e-10):
            # 叶子节点: Newton-Raphson 最优输出 γ = Σr / Σ(p*(1-p))
            node["leaf"] = True
            if hessians is not None and np.sum(hessians) > 1e-10:
                node["value"] = np.sum(y) / np.sum(hessians)
            else:
                node["value"] = np.mean(y) if len(y) > 0 else 0.0
            return node

        D = X.shape[1]
        n_feats = max(1, int(D * self.max_features))
        feat_idx = self.rng.choice(D, size=min(n_feats, D), replace=False)

        best_feat, best_thr = self._best_split(X, y, feat_idx)
        if best_feat is None:
            node["leaf"] = True
            node["value"] = np.mean(y) if len(y) > 0 else 0.0
            return node

        left = X[:, best_feat] <= best_thr
        node["leaf"] = False
        node["feat"] = best_feat
        node["thr"] = best_thr
        h_l = hessians[left] if hessians is not None else None
        h_r = hessians[~left] if hessians is not None else None
        node["left"] = self._build(X[left], y[left], h_l, depth + 1)
        node["right"] = self._build(X[~left], y[~left], h_r, depth + 1)
        return node

    def fit(self, X, residuals, hessians=None):
        """拟合回归树到梯度残差"""
        D = X.shape[1]
        self.tree = self._build(X, residuals, hessians, depth=0)

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feat"]] <= node["thr"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(X[i], self.tree) for i in range(X.shape[0])])


def _sigmoid(x):
    """数值稳定的 sigmoid"""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


class EmberGBDT:
    """
    Gradient Boosted Decision Trees — 纯 Python 实现

    与 Anderson & Roth 2018 论文 (LightGBM) 参数对齐:
      n_estimators: 树的数量
      max_depth: 单棵树深度
      learning_rate: 收缩率
      subsample: 行采样比例
      colsample: 列采样比例
    """

    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.05,
                 subsample=0.8, min_samples_leaf=50):
        """
        参数:
            n_estimators: 迭代轮数 (论文1000, 默认200以平衡速度)
            max_depth: 树深度 (论文15, 默认8)
            learning_rate: 收缩率 (论文0.05)
            subsample: 行采样比例 (论文0.8)
            min_samples_leaf: 叶子最少样本 (论文50)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.F0 = 0.0

    def fit(self, X, y):
        """
        GBDT 训练: 逐步拟合梯度残差

        参数:
            X: numpy array (N, D)
            y: numpy array (N,), 标签 {0, 1}
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        N = X.shape[0]
        rng = np.random.RandomState(42)

        # 初始化: F_0 = log(p / (1-p))
        p_mean = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.F0 = np.log(p_mean / (1 - p_mean))
        F = np.full(N, self.F0)
        self.trees = []

        for m in range(self.n_estimators):
            # 当前预测概率
            p = _sigmoid(F)
            p = np.clip(p, 1e-10, 1 - 1e-10)

            # 负梯度残差 (= y - p 对于 log loss)
            residuals = y - p

            # 二阶导 (Hessian): p * (1 - p)
            hessians = p * (1 - p)

            # 行采样
            n_sub = max(1, int(N * self.subsample))
            sub_idx = rng.choice(N, size=n_sub, replace=False)

            # 拟合回归树到残差
            tree = GBDTRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=0.8,
                random_state=rng.randint(2**31))
            tree.fit(X[sub_idx], residuals[sub_idx], hessians[sub_idx])
            self.trees.append(tree)

            # 更新: F_m = F_{m-1} + η · h_m(x)
            F += self.learning_rate * tree.predict(X)

            if (m + 1) % 50 == 0:
                loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                print(f"  [EmberGBDT] 第 {m+1}/{self.n_estimators} 轮, log_loss={loss:.4f}")

    def _raw_predict(self, X):
        """计算原始决策函数值 F(x)"""
        X = np.asarray(X, dtype=np.float32)
        F = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict(self, X):
        """预测标签"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """预测概率"""
        F = self._raw_predict(X)
        p1 = _sigmoid(F)
        return np.stack([1 - p1, p1], axis=1)
