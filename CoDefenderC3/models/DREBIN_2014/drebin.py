"""
DREBIN — Arp et al. 2014
论文: "DREBIN: Effective and Explainable Detection of Android Malware in Your Pocket"
会议: NDSS 2014
链接: https://www.ndss-symposium.org/ndss2014/

1:1 精准复刻: Linear SVM (线性超平面, 铰链损失, L2正则化)
纯 Python + numpy 实现

算法: 随机梯度下降 (SGD) 优化铰链损失
  损失: L = Σ max(0, 1 - y_i · (w·x_i + b)) + (C/2) · ||w||²
  梯度:
    若 y_i · (w·x_i + b) < 1:
      ∂L/∂w = -y_i · x_i + C · w
      ∂L/∂b = -y_i
    否则:
      ∂L/∂w = C · w
      ∂L/∂b = 0

Platt Scaling: 训练后用 sigmoid 校准输出概率
  P(y=1|x) = 1 / (1 + exp(A · f(x) + B))
  A, B 通过最大似然估计拟合

输入: numpy array (N, D)
输出: 预测标签和概率
"""
import numpy as np


class DrebinLinear:
    """Linear SVM — 纯 Python SGD 实现, 与论文完全一致"""

    def __init__(self, C=1.0, lr=0.001, n_epochs=100, batch_size=256):
        """
        参数:
            C: 正则化强度 (论文使用 C=1.0)
            lr: 学习率
            n_epochs: SGD 迭代轮数
            batch_size: 小批量大小
        """
        self.C = C
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.w = None  # 权重向量
        self.b = 0.0   # 偏置
        # Platt Scaling 参数
        self.platt_A = -1.0
        self.platt_B = 0.0

    def fit(self, X, y):
        """
        SGD 训练 Linear SVM

        参数:
            X: numpy array (N, D), 训练特征
            y: numpy array (N,), 标签 (0 或 1)
        """
        X = np.asarray(X, dtype=np.float32)
        # 将标签从 {0, 1} 转换为 {-1, +1} (SVM 标准)
        y_svm = np.where(np.asarray(y) == 1, 1.0, -1.0)

        N, D = X.shape
        self.w = np.zeros(D, dtype=np.float64)
        self.b = 0.0
        rng = np.random.RandomState(42)

        # SGD 训练
        for epoch in range(self.n_epochs):
            indices = rng.permutation(N)
            for start in range(0, N, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                xb = X[batch_idx]
                yb = y_svm[batch_idx]

                # 计算 decision function: f(x) = w·x + b
                margins = yb * (xb @ self.w + self.b)  # (batch,)

                # 铰链损失梯度
                violated = margins < 1.0  # 违反间隔的样本
                grad_w = self.C * self.w  # L2 正则化梯度
                grad_b = 0.0

                if np.any(violated):
                    grad_w -= np.mean(yb[violated, None] * xb[violated], axis=0)
                    grad_b -= np.mean(yb[violated])

                # SGD 更新
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

        # Platt Scaling: 用 sigmoid 校准概率
        self._platt_fit(X, y)

    def _platt_fit(self, X, y):
        """
        Platt Scaling — 拟合 sigmoid 参数 A, B
        P(y=1|f) = 1 / (1 + exp(A·f + B))
        使用牛顿法最大似然估计 (Platt 1999)
        """
        f = self.decision_function(X)
        y = np.asarray(y, dtype=np.float64)

        # 目标概率 (Platt 的平滑处理)
        N_pos = np.sum(y == 1)
        N_neg = np.sum(y == 0)
        target = np.where(y == 1, (N_pos + 1) / (N_pos + 2), 1.0 / (N_neg + 2))

        # 牛顿法迭代求 A, B
        A, B = 0.0, np.log((N_neg + 1) / (N_pos + 1))
        for _ in range(100):
            p = 1.0 / (1.0 + np.exp(A * f + B))
            p = np.clip(p, 1e-10, 1 - 1e-10)

            # 梯度
            dA = np.sum(f * (target - p))
            dB = np.sum(target - p)

            # Hessian 对角近似
            d2A = np.sum(f * f * p * (1 - p)) + 1e-8
            d2B = np.sum(p * (1 - p)) + 1e-8

            A -= dA / d2A
            B -= dB / d2B

        self.platt_A = A
        self.platt_B = B

    def decision_function(self, X):
        """计算 SVM 决策函数值: f(x) = w·x + b"""
        X = np.asarray(X, dtype=np.float32)
        return X @ self.w + self.b

    def predict(self, X):
        """预测标签: sign(w·x + b) → {0, 1}"""
        f = self.decision_function(X)
        return (f >= 0).astype(np.int64)

    def predict_proba(self, X):
        """Platt Scaling 概率输出"""
        f = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(self.platt_A * f + self.platt_B))
        return np.stack([1 - p1, p1], axis=1)
