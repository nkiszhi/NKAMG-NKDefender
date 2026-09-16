"""
models/ — 所有恶意软件检测模型 + 集成桥接层
================================================
31个模型目录 (模型名_年份/), 每个对应一篇论文
ensemble.py — MultiModelEnsemble 桥接层, 连接模型到 SDD 引擎

实现类型:
  26个 nn.Module (论文本身就是神经网络)
  5个纯 Python+numpy (kNN/SVM/RF/GBDT/XGBoost, 1:1精准复刻)
  6个用户原始代码 (MalConv/MalConv2/Ember/IMCFN/InceptionV3/MalGraph)
"""
from .ensemble import MultiModelEnsemble
