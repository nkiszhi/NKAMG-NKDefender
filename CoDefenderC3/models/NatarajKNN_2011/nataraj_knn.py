"""
NatarajKNN — Nataraj et al. 2011
论文: "Malware Images: Visualization and Automatic Classification"
会议: VizSec 2011
链接: https://doi.org/10.1145/2016904.2016908

1:1 精准复刻: k-Nearest Neighbors (k=5, 欧氏距离, 距离加权)
纯 Python + numpy 实现, 不依赖 sklearn

输入: numpy array (N, D)
输出: 预测标签和概率
"""
import numpy as np


class NatarajKNN:
    """k-最近邻分类器 — 纯 Python 实现, 与论文完全一致"""

    def __init__(self, k=5, epsilon=1e-8, max_train=50000):
        self.k = k
        self.epsilon = epsilon
        self.max_train = max_train
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """存储训练样本 (lazy learner), 大数据集子采样防 OOM"""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if len(X) > self.max_train:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """欧氏距离 → k近邻 → 距离加权投票 (向量化, 避免OOM)"""
        X = np.asarray(X, dtype=np.float32)
        preds = np.zeros(X.shape[0], dtype=np.int64)
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b — 避免 (M,N,D) 中间张量
        train_sq = np.sum(self.X_train ** 2, axis=1)  # (N_train,)
        chunk = 500
        for start in range(0, X.shape[0], chunk):
            end = min(start + chunk, X.shape[0])
            Xc = X[start:end]
            test_sq = np.sum(Xc ** 2, axis=1, keepdims=True)  # (chunk, 1)
            cross = Xc @ self.X_train.T                        # (chunk, N_train)
            dist_sq = test_sq + train_sq[None, :] - 2 * cross  # (chunk, N_train)
            dists = np.sqrt(np.maximum(dist_sq, 0))
            k = min(self.k, dists.shape[1])
            nn_idx = np.argpartition(dists, k, axis=1)[:, :k]
            nn_dists = np.take_along_axis(dists, nn_idx, axis=1)
            nn_labels = self.y_train[nn_idx]
            weights = 1.0 / (nn_dists + self.epsilon)
            vote_1 = np.sum(weights * (nn_labels == 1), axis=1)
            vote_0 = np.sum(weights * (nn_labels == 0), axis=1)
            preds[start:end] = (vote_1 > vote_0).astype(np.int64)
        return preds

    def predict_proba(self, X):
        """返回 (M, 2) 概率矩阵 (向量化, 避免OOM)"""
        X = np.asarray(X, dtype=np.float32)
        proba = np.zeros((X.shape[0], 2))
        train_sq = np.sum(self.X_train ** 2, axis=1)
        chunk = 500
        for start in range(0, X.shape[0], chunk):
            end = min(start + chunk, X.shape[0])
            Xc = X[start:end]
            test_sq = np.sum(Xc ** 2, axis=1, keepdims=True)
            cross = Xc @ self.X_train.T
            dist_sq = test_sq + train_sq[None, :] - 2 * cross
            dists = np.sqrt(np.maximum(dist_sq, 0))
            k = min(self.k, dists.shape[1])
            nn_idx = np.argpartition(dists, k, axis=1)[:, :k]
            nn_dists = np.take_along_axis(dists, nn_idx, axis=1)
            nn_labels = self.y_train[nn_idx]
            weights = 1.0 / (nn_dists + self.epsilon)
            total = weights.sum(axis=1, keepdims=True)
            proba[start:end, 0] = np.sum(weights * (nn_labels == 0), axis=1) / total.squeeze()
            proba[start:end, 1] = np.sum(weights * (nn_labels == 1), axis=1) / total.squeeze()
        return proba
