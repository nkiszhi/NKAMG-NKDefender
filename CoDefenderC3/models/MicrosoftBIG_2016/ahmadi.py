"""
MicrosoftBIG — Ahmadi et al. 2016
论文: "Novel Feature Extraction, Selection and Fusion for Effective Malware Family Classification"
会议: ACM CODASPY 2016
链接: https://doi.org/10.1145/2857705.2857713

1:1 精准复刻:
  OpcodeStatNet: XGBoost on opcode n-gram + 指令类别分布
  ResourceNet:   Random Forest on PE 资源节特征

纯 Python + numpy 实现

XGBoost 算法 (Chen & Guestrin 2016):
  与 GBDT 类似, 区别:
    1. 二阶泰勒展开: 使用梯度 g 和 Hessian h
    2. 正则化: 叶子值 L2 正则 + 叶子数惩罚
    3. 叶子最优值: w_j = -G_j / (H_j + λ)
    4. 分裂增益: Gain = G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ) - γ

输入: numpy array (N, D)
输出: 预测标签和概率
"""
import numpy as np


def _sigmoid(x):
    """数值稳定的 sigmoid"""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# ═══════════════════════════════════════
#  XGBoost 回归树 (用于 OpcodeStatNet)
# ═══════════════════════════════════════

class XGBTree:
    """XGBoost 单棵正则化回归树"""

    def __init__(self, max_depth=6, min_child_weight=10, reg_lambda=1.0,
                 gamma=0.1, colsample=0.8, random_state=None):
        """
        参数:
            max_depth: 最大深度
            min_child_weight: 子节点最小 Hessian 和
            reg_lambda: L2 正则化系数 (λ)
            gamma: 分裂最小增益 (γ)
            colsample: 列采样比例
        """
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.colsample = colsample
        self.rng = np.random.RandomState(random_state)
        self.tree = None

    def _calc_gain(self, G_L, H_L, G_R, H_R):
        """XGBoost 分裂增益公式"""
        lam = self.reg_lambda
        gain = (G_L**2 / (H_L + lam) +
                G_R**2 / (H_R + lam) -
                (G_L + G_R)**2 / (H_L + H_R + lam)) / 2.0 - self.gamma
        return gain

    def _best_split(self, X, grads, hess, feat_indices):
        """找 XGBoost 增益最大的分裂"""
        best_gain = 0.0  # 必须 > 0 才分裂 (含 gamma 惩罚)
        best_feat, best_thr = None, None
        N = len(grads)

        for feat in feat_indices:
            vals = X[:, feat]
            sorted_idx = np.argsort(vals)
            s_vals = vals[sorted_idx]
            s_g = grads[sorted_idx]
            s_h = hess[sorted_idx]

            G_total, H_total = np.sum(s_g), np.sum(s_h)
            G_L, H_L = 0.0, 0.0

            for i in range(N - 1):
                G_L += s_g[i]
                H_L += s_h[i]
                G_R = G_total - G_L
                H_R = H_total - H_L

                # 跳过重复值
                if s_vals[i] == s_vals[i + 1]:
                    continue
                # 最小 Hessian 约束
                if H_L < self.min_child_weight or H_R < self.min_child_weight:
                    continue

                gain = self._calc_gain(G_L, H_L, G_R, H_R)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = (s_vals[i] + s_vals[i + 1]) / 2.0

        return best_feat, best_thr

    def _build(self, X, grads, hess, depth):
        """递归构建"""
        node = {}
        G, H = np.sum(grads), np.sum(hess)

        if depth >= self.max_depth or len(grads) < 2 or H < self.min_child_weight:
            node["leaf"] = True
            node["value"] = -G / (H + self.reg_lambda)  # XGBoost 叶子最优值
            return node

        D = X.shape[1]
        n_feats = max(1, int(D * self.colsample))
        feat_idx = self.rng.choice(D, size=min(n_feats, D), replace=False)

        best_feat, best_thr = self._best_split(X, grads, hess, feat_idx)
        if best_feat is None:
            node["leaf"] = True
            node["value"] = -G / (H + self.reg_lambda)
            return node

        left = X[:, best_feat] <= best_thr
        node["leaf"] = False
        node["feat"] = best_feat
        node["thr"] = best_thr
        node["left"] = self._build(X[left], grads[left], hess[left], depth + 1)
        node["right"] = self._build(X[~left], grads[~left], hess[~left], depth + 1)
        return node

    def fit(self, X, grads, hess):
        self.tree = self._build(X, grads, hess, 0)

    def _pred_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feat"]] <= node["thr"]:
            return self._pred_one(x, node["left"])
        return self._pred_one(x, node["right"])

    def predict(self, X):
        return np.array([self._pred_one(X[i], self.tree) for i in range(X.shape[0])])


class OpcodeStatNet:
    """
    XGBoost 分类器 — 纯 Python 实现

    论文中用于 opcode n-gram + 指令类别分布特征的分类
    算法: 二阶 GBDT (XGBoost, Chen & Guestrin 2016)
    """

    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1,
                 reg_lambda=1.0, gamma=0.1, subsample=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.trees = []
        self.F0 = 0.0

    def fit(self, X, y):
        """XGBoost 训练"""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        N = X.shape[0]
        rng = np.random.RandomState(42)

        p_mean = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.F0 = np.log(p_mean / (1 - p_mean))
        F = np.full(N, self.F0)
        self.trees = []

        for m in range(self.n_estimators):
            p = _sigmoid(F)
            p = np.clip(p, 1e-10, 1 - 1e-10)

            # 一阶梯度 g_i = p_i - y_i, 二阶 h_i = p_i * (1 - p_i)
            grads = p - y
            hess = p * (1 - p)

            # 行采样
            n_sub = max(1, int(N * self.subsample))
            idx = rng.choice(N, size=n_sub, replace=False)

            tree = XGBTree(
                max_depth=self.max_depth,
                min_child_weight=10,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                random_state=rng.randint(2**31))
            tree.fit(X[idx], grads[idx], hess[idx])
            self.trees.append(tree)

            F += self.learning_rate * tree.predict(X)

            if (m + 1) % 50 == 0:
                loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                print(f"  [OpcodeStatNet XGBoost] 第 {m+1}/{self.n_estimators} 轮, log_loss={loss:.4f}")

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        F = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        p1 = _sigmoid(F)
        return np.stack([1 - p1, p1], axis=1)


# ═══════════════════════════════════════
#  Random Forest (用于 ResourceNet)
# ═══════════════════════════════════════

class _RFTree:
    """CART 回归/分类树 (简化版, 复用 PEMiner 的逻辑)"""

    def __init__(self, max_depth=10, min_samples_leaf=10, random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.rng = np.random.RandomState(random_state)
        self.tree = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        c = np.bincount(y, minlength=2)
        p = c / len(y)
        return 1.0 - np.sum(p ** 2)

    def _build(self, X, y, depth):
        node = {"counts": np.bincount(y, minlength=2)}
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or len(np.unique(y)) == 1:
            node["leaf"] = True
            return node

        D = X.shape[1]
        n_f = max(1, int(np.sqrt(D)))
        feats = self.rng.choice(D, n_f, replace=False)

        best_gain, best_feat, best_thr = -1, None, None
        parent_g = self._gini(y)
        N = len(y)

        for f in feats:
            thrs = np.unique(np.percentile(X[:, f], np.linspace(10, 90, 8)))
            for t in thrs:
                left = X[:, f] <= t
                nl, nr = np.sum(left), N - np.sum(left)
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                gain = parent_g - nl/N * self._gini(y[left]) - nr/N * self._gini(y[~left])
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, f, t

        if best_feat is None:
            node["leaf"] = True
            return node

        left = X[:, best_feat] <= best_thr
        node["leaf"] = False
        node["feat"] = best_feat
        node["thr"] = best_thr
        node["left"] = self._build(X[left], y[left], depth + 1)
        node["right"] = self._build(X[~left], y[~left], depth + 1)
        return node

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)

    def _pred(self, x, node):
        if node["leaf"]:
            c = node["counts"]
            return c / max(np.sum(c), 1)
        if x[node["feat"]] <= node["thr"]:
            return self._pred(x, node["left"])
        return self._pred(x, node["right"])

    def predict_proba(self, X):
        return np.array([self._pred(X[i], self.tree) for i in range(X.shape[0])])


class ResourceNet:
    """
    Random Forest — 纯 Python 实现

    论文中用于 PE 资源节特征的分类
    """

    def __init__(self, n_estimators=100, max_depth=10, min_samples_leaf=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        N = X.shape[0]
        rng = np.random.RandomState(42)
        self.trees = []

        for t in range(self.n_estimators):
            idx = rng.choice(N, N, replace=True)
            tree = _RFTree(self.max_depth, self.min_samples_leaf, rng.randint(2**31))
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        s = np.zeros((X.shape[0], 2))
        for tree in self.trees:
            s += tree.predict_proba(X)
        return s / len(self.trees)
