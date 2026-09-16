"""
feature_extraction/ — 统一特征提取模块
=========================================
所有模型的特征由 unified.py 统一管理

使用方法:
    from feature_extraction.unified import FeatureType, extract_feature, UnifiedDataset

    # 直接提取
    tensor = extract_feature("sample.exe", FeatureType.EMBER_FULL)

    # 在 Dataset 中使用
    dataset = UnifiedDataset(samples, feature_type=FeatureType.EMBER_BYTEHIST)
"""
from .unified import FeatureType, extract_feature, UnifiedDataset, get_feature_dim, set_cache_dir
