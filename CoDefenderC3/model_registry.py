"""
模型注册表 — 所有模型到特征类型的中心映射
=============================================
每个模型在此注册:
  - 特征类型 (FeatureType)
  - 特征提取参数 (max_len, img_size 等)
  - 模型构造参数
  - 损失函数类型 (bce / ce)
  - 是否为纯 Python 模型 (不使用 PyTorch)

所有 run 脚本通过此注册表获取配置, 不再各自硬编码
"""
from feature_extraction.unified import FeatureType


# 模型类型: pytorch (nn.Module) 或 numpy (纯Python)
PYTORCH = "pytorch"
NUMPY = "numpy"


REGISTRY = {
    # ═══════════════════════════════
    #  V1: 字节级模型 (raw_bytes)
    # ═══════════════════════════════
    "MalConv": {
        "module": "models.MalConv_2017.malconv",
        "class": "Malconv",
        "feature_type": FeatureType.RAW_BYTES,
        "feature_kwargs": {"max_len": 200000},
        "model_kwargs": {},
        "loss": "bce",       # (B,1) Sigmoid → BCELoss
        "engine": PYTORCH,
        "save_path": "../models/MalConv_2017/saved/malconv_best.pth",
    },
    "MalConv2": {
        "module": "models.MalConv2_2021.malconv2",
        "class": "MalConvGCT",
        "feature_type": FeatureType.RAW_BYTES,
        "feature_kwargs": {"max_len": 16000000},
        "model_kwargs": {"out_size": 2, "channels": 128, "window_size": 512,
                         "stride": 64, "embd_size": 8, "low_mem": True},
        "loss": "ce",        # (B,2) logits → CrossEntropyLoss, 返回三元组
        "engine": PYTORCH,
        "save_path": "../models/MalConv2_2021/saved/malconv_gct_best.pth",
    },
    "ByteTransformer": {
        "module": "models.ByteTransformer_2023.byte_transformer",
        "class": "ByteTransformer",
        "feature_type": FeatureType.RAW_BYTES,
        "feature_kwargs": {"max_len": 32768},
        "model_kwargs": {},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/ByteTransformer_2023/saved/bytetransformer_best.pth",
    },

    # ═══════════════════════════════
    #  V2: 字节统计模型 (ember_bytehist)
    # ═══════════════════════════════
    "SaxeBerlinDNN": {
        "module": "models.SaxeBerlinDNN_2015.saxe_berlin",
        "class": "SaxeBerlinDNN",
        "feature_type": FeatureType.EMBER_BYTEHIST,
        "feature_kwargs": {},
        "model_kwargs": {"input_dim": 512},
        "loss": "bce",
        "engine": PYTORCH,
        "save_path": "../models/SaxeBerlinDNN_2015/saved/saxeberlin_best.pth",
    },
    "EmberDNN": {
        "module": "models.EmberDNN_2018.ember_dnn",
        "class": "EmberDNN",
        "feature_type": FeatureType.EMBER_BYTEHIST,
        "feature_kwargs": {},
        "model_kwargs": {"input_dim": 512},
        "loss": "bce",
        "engine": PYTORCH,
        "save_path": "../models/EmberDNN_2018/saved/emberdnn_best.pth",
    },
    "NatarajKNN": {
        "module": "models.NatarajKNN_2011.nataraj_knn",
        "class": "NatarajKNN",
        "feature_type": FeatureType.EMBER_BYTEHIST,
        "feature_kwargs": {},
        "model_kwargs": {"k": 5},
        "loss": "none",      # 纯 Python kNN, 无损失函数
        "engine": NUMPY,
        "save_path": "../models/NatarajKNN_2011/saved/natarajknn.npz",
    },

    # ═══════════════════════════════
    #  V3: PE 结构模型 (ember_pe_struct)
    # ═══════════════════════════════
    "DrebinSVM": {
        "module": "models.DREBIN_2014.drebin",
        "class": "DrebinLinear",
        "feature_type": FeatureType.EMBER_PE_STRUCT,
        "feature_kwargs": {},
        "model_kwargs": {"C": 1.0},
        "loss": "none",
        "engine": NUMPY,
        "save_path": "../models/DREBIN_2014/saved/drebin.npz",
    },
    "DL4MD": {
        "module": "models.DL4MD_2016.dl4md",
        "class": "DL4MD",
        "feature_type": FeatureType.EMBER_PE_STRUCT,
        "feature_kwargs": {},
        "model_kwargs": {"input_dim": 327},
        "loss": "bce",
        "engine": PYTORCH,
        "save_path": "../models/DL4MD_2016/saved/dl4md_best.pth",
    },
    "PEMinerRF": {
        "module": "models.PEMiner_2009.peminer",
        "class": "PEMinerRF",
        "feature_type": FeatureType.EMBER_PE_STRUCT,
        "feature_kwargs": {},
        "model_kwargs": {"n_estimators": 30, "max_depth": 10},
        "loss": "none",
        "engine": NUMPY,
        "save_path": "../models/PEMiner_2009/saved/peminer.npz",
    },

    # ═══════════════════════════════
    #  V4: 导入表模型 (ember_import)
    # ═══════════════════════════════
    "ALOHANet": {
        "module": "models.ALOHANet_2020.aloha",
        "class": "ALOHANet",
        "feature_type": FeatureType.EMBER_IMPORT,
        "feature_kwargs": {},
        "model_kwargs": {"input_dim": 1280},
        "loss": "bce",
        "engine": PYTORCH,
        "save_path": "../models/ALOHANet_2020/saved/aloha_best.pth",
    },
    "DrebinImport": {
        "module": "models.DREBIN_2014.drebin",
        "class": "DrebinLinear",
        "feature_type": FeatureType.EMBER_IMPORT,
        "feature_kwargs": {},
        "model_kwargs": {"C": 1.0},
        "loss": "none",
        "engine": NUMPY,
        "save_path": "../models/DREBIN_2014/saved/drebin_import.npz",
    },
    "EmberGBDT_Import": {
        "module": "models.EmberGBDT_2018.ember_gbdt",
        "class": "EmberGBDT",
        "feature_type": FeatureType.EMBER_IMPORT,
        "feature_kwargs": {},
        "model_kwargs": {"n_estimators": 30, "max_depth": 6, "learning_rate": 0.1},
        "loss": "none",
        "engine": NUMPY,
        "save_path": "../models/EmberGBDT_2018/saved/embergbdt_import.npz",
    },

    # ═══════════════════════════════
    #  V5: 字符串模型 (ember_string)
    # ═══════════════════════════════
    "RaffFeatureNet": {
        "module": "models.RaffFeatureNet_2018.raff_feature",
        "class": "RaffFeatureNet",
        "feature_type": FeatureType.EMBER_STRING,
        "feature_kwargs": {},
        "model_kwargs": {"input_dim": 104},
        "loss": "bce",
        "engine": PYTORCH,
        "save_path": "../models/RaffFeatureNet_2018/saved/rafffeature_best.pth",
    },
    "DrebinString": {
        "module": "models.DREBIN_2014.drebin",
        "class": "DrebinLinear",
        "feature_type": FeatureType.EMBER_STRING,
        "feature_kwargs": {},
        "model_kwargs": {"C": 1.0},
        "loss": "none",
        "engine": NUMPY,
        "save_path": "../models/DREBIN_2014/saved/drebin_string.npz",
    },

    # ═══════════════════════════════
    #  V6: 操作码模型 (opcode_seq)
    # ═══════════════════════════════
    "OpcodeLSTM": {
        "module": "models.OpcodeLSTM_2017.opcode_lstm",
        "class": "OpcodeLSTM",
        "feature_type": FeatureType.OPCODE_SEQ,
        "feature_kwargs": {"seq_len": 4096},
        "model_kwargs": {},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/OpcodeLSTM_2017/saved/opcodelstm_best.pth",
    },
    "OpcodeTransformer": {
        "module": "models.OpcodeTransformer_2020.opcode_transformer",
        "class": "OpcodeTransformer",
        "feature_type": FeatureType.OPCODE_SEQ,
        "feature_kwargs": {"seq_len": 512},
        "model_kwargs": {},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/OpcodeTransformer_2020/saved/opcodetransformer_best.pth",
    },
    "OpcodeStatNet": {
        "module": "models.MicrosoftBIG_2016.ahmadi",
        "class": "OpcodeStatNet",
        "feature_type": FeatureType.EMBER_BYTEHIST,
        "feature_kwargs": {},
        "model_kwargs": {"n_estimators": 30},
        "loss": "none",
        "engine": NUMPY,
        "save_path": "../models/MicrosoftBIG_2016/saved/opcodestatnet.npz",
    },
    "AsmEmbedNet": {
        "module": "models.Asm2Vec_2019.asm_embed",
        "class": "AsmEmbedNet",
        "feature_type": FeatureType.FUNC_EMBEDDINGS,
        "feature_kwargs": {"n_funcs": 48, "embed_dim": 128},
        "model_kwargs": {},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/Asm2Vec_2019/saved/asm2vec_best.pth",
    },

    # ═══════════════════════════════
    #  V7: 图模型 (graph_stat / pyg_graph)
    # ═══════════════════════════════
    "CFGGAT": {
        "module": "models.CFG_GAT_2019.cfg_gat",
        "class": "CFGGAT",
        "feature_type": FeatureType.GRAPH_STAT,
        "feature_kwargs": {},
        "model_kwargs": {},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/CFG_GAT_2019/saved/cfg_gat_best.pth",
    },
    "CFGGCN": {
        "module": "models.GCN_2017.cfg_gcn",
        "class": "CFGGCN",
        "feature_type": FeatureType.GRAPH_STAT,
        "feature_kwargs": {},
        "model_kwargs": {},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/GCN_2017/saved/gcn_best.pth",
    },
    "CFGDGCNN": {
        "module": "models.DGCNN_2018.cfg_dgcnn",
        "class": "CFGDGCNN",
        "feature_type": FeatureType.GRAPH_STAT,
        "feature_kwargs": {},
        "model_kwargs": {},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/DGCNN_2018/saved/dgcnn_best.pth",
    },
    "CallGraphGNN": {
        "module": "models.CallGraphGNN_2019.callgraph_gnn",
        "class": "CallGraphGNN",
        "feature_type": FeatureType.GRAPH_STAT,
        "feature_kwargs": {},
        "model_kwargs": {},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/CallGraphGNN_2019/saved/callgraphgnn_best.pth",
    },
    "GraphStatNet": {
        "module": "models.GraphStatNet_2022.graph_stat",
        "class": "GraphStatNet",
        "feature_type": FeatureType.GRAPH_STAT,
        "feature_kwargs": {},
        "model_kwargs": {},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/GraphStatNet_2022/saved/graphstat_best.pth",
    },

    # ═══════════════════════════════
    #  V8: 视觉模型 (image)
    # ═══════════════════════════════
    "IMCFN": {
        "module": "models.IMCFN_2020.Imcfn",
        "class": "IMCFN",
        "feature_type": FeatureType.COLOR_IMAGE,
        "feature_kwargs": {"img_size": 224},
        "model_kwargs": {"num_classes": 2, "freeze_until_block": 4},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/IMCFN_2020/saved/imcfn_best.pth",
    },
    "InceptionV3": {
        "module": "models.InceptionV3_2020.InceptionV3",
        "class": "InceptionV3Model",
        "feature_type": FeatureType.GRAY_IMAGE,
        "feature_kwargs": {"img_size": 299},
        "model_kwargs": {"pretrained": True},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/InceptionV3_2020/saved/inceptionv3_best.pth",
    },
    "GrayscaleCNN": {
        "module": "models.GrayscaleCNN_2011.grayscale_cnn",
        "class": "GrayscaleCNN",
        "feature_type": FeatureType.GRAY_IMAGE,
        "feature_kwargs": {"img_size": 256},
        "model_kwargs": {},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/GrayscaleCNN_2011/saved/grayscalecnn_best.pth",
    },
    "ColorCNN": {
        "module": "models.ColorCNN_2019.color_cnn",
        "class": "ColorCNN",
        "feature_type": FeatureType.COLOR_IMAGE,
        "feature_kwargs": {"img_size": 224},
        "model_kwargs": {},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/ColorCNN_2019/saved/colorcnn_best.pth",
    },
    "MarkovCNN": {
        "module": "models.MarkovCNN_2018.markov_cnn",
        "class": "MarkovCNN",
        "feature_type": FeatureType.MARKOV_IMAGE,
        "feature_kwargs": {},
        "model_kwargs": {},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/MarkovCNN_2018/saved/markovcnn_best.pth",
    },
    "EntropyMapCNN": {
        "module": "models.EntropyMapCNN_2018.entropy_cnn",
        "class": "EntropyMapCNN",
        "feature_type": FeatureType.ENTROPY_IMAGE,
        "feature_kwargs": {},
        "model_kwargs": {},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/EntropyMapCNN_2018/saved/entropymapcnn_best.pth",
    },

    # ═══════════════════════════════
    #  V9: 哈希模型
    # ═══════════════════════════════
    "HashEmbedNet": {
        "module": "models.HashEmbedNet_2019.hash_embed",
        "class": "HashEmbedNet",
        "feature_type": FeatureType.EMBER_BYTEHIST,
        "feature_kwargs": {},
        "model_kwargs": {"input_dim": 512},
        "loss": "ce",
        "engine": PYTORCH,
        "save_path": "../models/HashEmbedNet_2019/saved/hashembed_best.pth",
    },

    # ═══════════════════════════════
    #  V10: 元数据模型
    # ═══════════════════════════════
    "MetadataNet": {
        "module": "models.MetadataNet_2009.metadata",
        "class": "MetadataNet",
        "feature_type": FeatureType.EMBER_METADATA,
        "feature_kwargs": {},
        "model_kwargs": {"input_dim": 158},
        "loss": "bce",
        "engine": PYTORCH,
        "save_path": "../models/MetadataNet_2009/saved/metadatanet_best.pth",
    },
    "ResourceNet": {
        "module": "models.MicrosoftBIG_2016.ahmadi",
        "class": "ResourceNet",
        "feature_type": FeatureType.EMBER_METADATA,
        "feature_kwargs": {},
        "model_kwargs": {"n_estimators": 30},
        "loss": "none",
        "engine": NUMPY,
        "save_path": "../models/MicrosoftBIG_2016/saved/resourcenet.npz",
    },

    # ═══════════════════════════════
    #  V11: 全特征集成
    # ═══════════════════════════════
    "Ember_NDF": {
        "module": "models.Ember_2018.ember",
        "class": "Ember",
        "feature_type": FeatureType.EMBER_FULL,
        "feature_kwargs": {},
        "model_kwargs": {"num_trees": 10, "tree_depth": 3},
        "loss": "bce",
        "engine": PYTORCH,
        "save_path": "../models/Ember_2018/saved/ember_best.pth",
    },
    "EmberGBDT": {
        "module": "models.EmberGBDT_2018.ember_gbdt",
        "class": "EmberGBDT",
        "feature_type": FeatureType.EMBER_FULL,
        "feature_kwargs": {},
        "model_kwargs": {"n_estimators": 30, "max_depth": 6, "learning_rate": 0.1},
        "loss": "none",
        "engine": NUMPY,
        "save_path": "../models/EmberGBDT_2018/saved/embergbdt.npz",
    },
}


def get_model_config(model_name):
    """获取模型配置"""
    if model_name not in REGISTRY:
        raise KeyError(f"未注册的模型: {model_name}, 可用: {list(REGISTRY.keys())}")
    return REGISTRY[model_name]


def list_models():
    """列出所有已注册模型"""
    for name, cfg in REGISTRY.items():
        ft = cfg["feature_type"].name
        eng = cfg["engine"]
        print(f"  {name:25s} {ft:20s} {eng:8s} {cfg['loss']:4s}")
