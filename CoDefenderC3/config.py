"""
CoDefenderC3 — 配置文件
========================
所有配置通过此文件或环境变量指定，无启发式推断。
"""
import os

# ── 路径配置 (通过环境变量或直接修改) ──
DATA_DIR   = os.environ.get("CODEFENDER_DATA_DIR",
                            os.path.join(r"E:\nkproject\test_samples", "CoDefenderC3_data"))
ROOT       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "results")
MODEL_PATH = os.environ.get("CODEFENDER_MODEL_PATH",
                            os.path.join(ROOT, "ensemble.pkl"))
CACHE_DIR  = os.path.join(ROOT, ".cache")

# ── 实验参数 ──
RUN_ABLATION = True
SAVE_MODEL   = True
N_WORKERS    = int(os.environ.get("CODEFENDER_N_WORKERS",
                                  min(max(os.cpu_count() or 8, 8), 16)))
ALPHA, DELTA_THR = 0.05, 0.50
WS, WR = 0.6, 0.4
BASELINE_MONTHS = 3
DACP_ALPHA = 0.10
SEED = 42

# ── EMBER 特征范围 (由数据集 metadata.json 的 feature_ranges 字段决定) ──
EMBER_RANGES_V2 = {
    "ByteHistogram": (0, 256),     "ByteEntropy":   (256, 512),
    "StringInfo":    (512, 616),   "GeneralInfo":   (616, 626),
    "HeaderInfo":    (626, 688),   "SectionInfo":   (688, 943),
    "ImportInfo":    (943, 2223),  "ExportInfo":    (2223, 2351),
    "DataDirs":      (2351, 2381),
}

EMBER_RANGES_V3 = {
    "ByteHistogram": (0, 256),     "ByteEntropy":   (256, 512),
    "StringInfo":    (512, 616),   "GeneralInfo":   (616, 636),
    "HeaderInfo":    (636, 738),   "SectionInfo":   (738, 1043),
    "ImportInfo":    (1043, 2323), "ExportInfo":    (2323, 2451),
    "DataDirs":      (2451, 2491), "RichHeader":    (2491, 2531),
    "Authenticode":  (2531, 2551), "ParseWarnings": (2551, 2568),
}

# 兼容旧引用
EMBER_RANGES = EMBER_RANGES_V2

def get_ember_ranges(ndim=2381):
    """根据特征维度返回对应的 EMBER 特征范围映射。
    注意: 优先使用 metadata.json 中的 feature_ranges 字段。"""
    if ndim >= 2568:
        return EMBER_RANGES_V3
    if ndim < 512:
        # 非 EMBER 数据集 (如 MalMem 55d): 全量映射到单视角
        return {"ByteHistogram": (0, ndim)}
    return EMBER_RANGES_V2


# ═══════════════════════════════════════════════════════════════
# 33 Models × 16 View Groups — 完整架构定义
# ═══════════════════════════════════════════════════════════════
#
# 运行时, ensemble.py 根据数据加载器提供的 views 选择可用的视角组。
# 不存在 "ember 模式" vs "full 模式" — 可用数据决定可用模型。
#
# 每个 view group 包含:
#   - ember_ranges (可选): EMBER 预提取特征的范围名列表
#   - binja_features (可选): BinaryNinja 提取的特征 ID 列表
#   - models: 该视角下的模型列表

VIEW_GROUPS = {
    "V1_byte": {
        "label": "Byte-Level", "attck": "T1027.002",
        "binja_features": ["B01"],
        "models": [
            ("MalConv",         "Raff et al. 2018 (AAAI-WS)",
             "models/MalConv_2017 | Embedding\u2192GatedConv\u2192GlobalMaxPool\u2192FC"),
            ("MalConv2",        "Raff et al. 2021 (MalConvGCT)",
             "models/MalConv2_2021 | MalConvML+GCT: GLU\u2192GlobalContext\u2192LowMemConv"),
            ("ByteTransformer", "Vaswani+Custom 2023",
             "models/ByteTransformer_2023 | 4-layer Transformer: d=128,nhead=4"),
        ],
    },
    "V2_byte_stat": {
        "label": "ByteStat", "attck": "T1027.002",
        "binja_features": ["B02", "B03", "B04", "B05"],
        "ember_ranges": ["ByteHistogram", "ByteEntropy"],
        "models": [
            ("SaxeBerlinDNN", "Saxe & Berlin 2015 (IEEE ISI)",
             "models/SaxeBerlinDNN_2015 | MLP(1024,512,256)+BN+Dropout"),
            ("EmberDNN",      "Anderson & Roth 2018 (EMBER)",
             "models/EmberDNN_2018 | MLP(300,300)+BN+Dropout"),
            ("NatarajKNN",    "Nataraj et al. 2011 (VizSec)",
             "models/NatarajKNN_2011 | kNN(k=5,distance-weighted)"),
        ],
    },
    "V3_pe_struct": {
        "label": "PE-Struct", "attck": "T1036.005",
        "binja_features": ["A01", "A02", "A03", "A04", "A05"],
        "ember_ranges": ["GeneralInfo", "HeaderInfo", "SectionInfo"],
        "models": [
            ("DrebinSVM",  "Arp et al. 2014 (NDSS)",
             "models/DREBIN_2014 | LinearSVM+Platt on PE structural features"),
            ("DL4MD",      "Hardy et al. 2016 (DMIN)",
             "models/DL4MD_2016 | StackedDenoisingAE(400,300,200,100)\u2192FC"),
            ("PEMinerRF",  "Shafiq et al. 2009 (RAID)",
             "models/PEMiner_2009 | RandomForest(200 trees,depth=15)"),
        ],
    },
    "V4_import": {
        "label": "Import", "attck": "T1027.007",
        "binja_features": ["A06", "A07", "A08_A15"],
        "ember_ranges": ["ImportInfo"],
        "models": [
            ("DrebinImport", "Arp et al. 2014 (NDSS)",
             "models/DREBIN_2014 | LinearSVM+Platt on import features"),
            ("ALOHANet",     "Aghakhani et al. 2020 (USENIX Sec)",
             "models/ALOHANet_2020 | MLP(512,256,128)+BN+L2-reg"),
            ("EmberGBDT_Import", "Anderson & Roth 2018 (EMBER)",
             "models/EmberGBDT_2018 | GBDT on import features"),
        ],
    },
    "V5_string": {
        "label": "String", "attck": "T1027",
        "binja_features": ["C01", "C02", "C03", "C04"],
        "ember_ranges": ["StringInfo"],
        "models": [
            ("RaffFeatureNet", "Raff et al. 2018 (AAAI-WS)",
             "models/RaffFeatureNet_2018 | MLP(256,128) feature baseline"),
            ("DrebinString",   "Arp et al. 2014 (NDSS)",
             "models/DREBIN_2014 | LinearSVM+Platt on string features"),
        ],
    },
    "V6_opcode_seq": {
        "label": "OpcodeSeq", "attck": "T1027.004",
        "binja_features": ["D01"],
        "models": [
            ("OpcodeLSTM",        "McLaughlin et al. 2017 (IEEE TIFS)",
             "models/OpcodeLSTM_2017 | BiLSTM(128\u00d72)\u2192FC"),
            ("OpcodeTransformer", "Ren et al. 2020 (IEEE Access)",
             "models/OpcodeTransformer_2020 | 4-layer Transformer on opcodes"),
        ],
    },
    "V6_opcode_stat": {
        "label": "OpcodeStat", "attck": "T1027.004",
        "binja_features": ["D02", "D03", "D04", "D05", "D06"],
        "models": [
            ("OpcodeStatNet",     "Ahmadi et al. 2016 (MS BIG)",
             "models/MicrosoftBIG_2016 | MLP(256,128) on n-gram+category dist"),
        ],
    },
    "V6_func_embed": {
        "label": "FuncEmbed", "attck": "T1027.004",
        "binja_features": ["D07"],
        "models": [
            ("AsmEmbedNet",       "Ding et al. 2019 (IEEE S&P, Asm2Vec)",
             "models/Asm2Vec_2019 | Embedding+LSTM on assembly functions"),
        ],
    },
    "V7_graph": {
        "label": "Graph", "attck": "T1027.004",
        "binja_features": ["E01", "E02", "E03", "E04"],
        "models": [
            ("MalGraph",     "Chen et al. 2022 (MalGraph)",
             "models/MalGraph_2022 | HierarchicalGNN [SPECIAL_PIPELINE]"),
            ("CFGGAT",       "Yan et al. 2019 (IEEE DSC)",
             "models/CFG_GAT_2019 | 3-layer GAT: heads=[8,8,1]"),
            ("CFGGCN",       "Kipf & Welling 2017 (ICLR)",
             "models/GCN_2017 | 3-layer GCN(64\u219232\u219216)\u2192ReadOut"),
            ("CFGDGCNN",     "Zhang et al. 2018 (AAAI)",
             "models/DGCNN_2018 | DGCNN: SortPool(k=30)\u21921D-CNN"),
            ("CallGraphGNN", "Li et al. 2019 (ICSE)",
             "models/CallGraphGNN_2019 | GraphSAGE on function call graph"),
            ("GraphStatNet", "Hybrid (MLP on graph stats)",
             "models/GraphStatNet_2022 | MLP(128,64) on CFG+CallGraph statistics"),
        ],
    },
    "V8_gray": {
        "label": "GrayVis", "attck": "T1027",
        "binja_features": ["F01"],
        "models": [
            ("GrayscaleCNN", "Nataraj et al. 2011 (VizSec)",
             "models/GrayscaleCNN_2011 | ResNet-style CNN on grayscale image"),
            ("InceptionV3",  "Nataraj-style 2020 (InceptionV3)",
             "models/InceptionV3_2020 | InceptionV3(pretrained)\u2192FC(2048\u21922)"),
        ],
    },
    "V8_color": {
        "label": "ColorVis", "attck": "T1027",
        "binja_features": ["F02"],
        "models": [
            ("IMCFN",        "Vasan et al. 2020 (Computer Networks)",
             "models/IMCFN_2020 | VGG16(pretrained)+FineTune"),
            ("ColorCNN",     "Liu & Wang 2019",
             "models/ColorCNN_2019 | Same CNN architecture, RGB input"),
        ],
    },
    "V8_markov": {
        "label": "MarkovVis", "attck": "T1027",
        "binja_features": ["F03"],
        "models": [
            ("MarkovCNN",    "Ni et al. 2018 (Computers & Security)",
             "models/MarkovCNN_2018 | CNN-3 on 256\u00d7256 Markov transition image"),
        ],
    },
    "V8_entropy": {
        "label": "EntropyVis", "attck": "T1027",
        "binja_features": ["F04"],
        "models": [
            ("EntropyMapCNN","Conti et al. 2018 (VizSec)",
             "models/EntropyMapCNN_2018 | CNN-3 on 256\u00d7256 entropy heatmap"),
        ],
    },
    "V9_hash": {
        "label": "Hash", "attck": "T1036",
        "binja_features": ["G01", "G02", "G03", "G04"],
        "models": [
            ("HashEmbedNet", "Pagani et al. 2019 (CCS Workshop)",
             "models/HashEmbedNet_2019 | Embedding(hash\u219264)\u00d74\u2192MLP"),
        ],
    },
    "V10_metadata": {
        "label": "Metadata", "attck": "T1036",
        "binja_features": ["H01", "H02", "H03"],
        # V2: ExportInfo+DataDirs=158d; V3额外含 RichHeader/Authenticode/ParseWarnings
        "ember_ranges": ["ExportInfo", "DataDirs", "RichHeader", "Authenticode", "ParseWarnings"],
        "models": [
            ("MetadataNet", "Shafiq et al. 2009 (RAID)",
             "models/MetadataNet_2009 | MLP(128,64) on version/resource/manifest"),
            ("ResourceNet", "Ahmadi et al. 2016 (MS BIG)",
             "models/MicrosoftBIG_2016 | MLP(64,32) on resource features"),
        ],
    },
    "V11_ensemble": {
        "label": "Full", "attck": "Multiple",
        "binja_features": [],
        "ember_ranges": ["__ALL__"],
        "models": [
            ("Ember_NDF",      "Anderson & Roth 2018 (EMBER)",
             "models/Ember_2018 | NeuralDecisionForest(SoftTrees\u00d710)"),
            ("EmberGBDT",      "Anderson & Roth 2018 (EMBER)",
             "models/EmberGBDT_2018 | LightGBM(leaves=2048,depth=15,n=1000)"),
        ],
    },
}

# 兼容旧引用
VIEW_GROUPS_FULL = VIEW_GROUPS
VIEW_GROUPS_EMBER = {k: v for k, v in VIEW_GROUPS.items() if "ember_ranges" in v}

# ── 视角可用性策略 (PE_RAW 训练) ──────────────────────────────────
# V6_opcode_* / V6_func_embed / V7_graph 依赖 BinaryNinja 反汇编 (D/E 特征)。
# 若 BN 不可用 (例如 Personal 许可不支持 headless), 这些视角永远提取不出来。
#
#   VIEWS_REQUIRE_ALL = 1 (严格/旧行为)
#       缺任何一个视角 → 该样本被整个排除。
#       在 BN 不可用的机器上会导致"所有样本都被排除" → train_months 全空 → 无法训练。
#   VIEWS_REQUIRE_ALL = 0 (默认/自适应)
#       构建月度缓存时先探测 .npz 实际含有哪些 feature_id, 把本数据集根本取不到的
#       视角从 active_views 中剔除 (对应模型不参与集成), 样本**不丢**。
#       仍然对"保留下来的视角"严格要求: 任一缺失 → 该样本排除 (防止半残样本污染训练)。
#
# 环境变量覆盖: CODEFENDER_VIEWS_REQUIRE_ALL=1
VIEWS_REQUIRE_ALL = os.environ.get("CODEFENDER_VIEWS_REQUIRE_ALL", "0") != "0"

# ═══════════════════════════════════════════════════════════════
# nkrepo 哈希库（SHA256 + MD5 预检）
# ═══════════════════════════════════════════════════════════════
# 说明：构造参数必须与建库时一致（layout=hex + fp_rate=0.01 + max_open_shards=16），
# 否则 HashSignatureDB._sync_layout 会触发全量重分片（62M 条约 311s）。
HASH_DB_ENABLED = os.environ.get("CODEFENDER_HASH_DB_ENABLED", "1") != "0"
HASH_DB_DIR     = os.environ.get("CODEFENDER_HASH_DB_DIR", os.path.join(ROOT, "signatures"))
HASH_DB_LAYOUT  = os.environ.get("CODEFENDER_HASH_DB_LAYOUT", "hex")
HASH_DB_SHARDS  = int(os.environ.get("CODEFENDER_HASH_DB_SHARDS", "4"))   # hex 布局下内部固定 256 片
HASH_DB_FP_RATE = float(os.environ.get("CODEFENDER_HASH_DB_FP_RATE", "0.01"))
HASH_DB_MAX_OPEN = int(os.environ.get("CODEFENDER_HASH_DB_MAX_OPEN", "16"))

# ── fuzzy 模糊哈希库（第三级预检: sha256 → md5 → fuzzy）─────────────
# ssdeep 为纯 Python 实现 (ppdeep), 耗时随输入近线性 (1MB≈24s):
# 预检是同步路径, 默认只对 ≤256KB 的文件算 ssdeep; 超限只算
# PE 类哈希 (imphash/authentihash, pefile 解析毫秒级)。
FUZZY_DB_ENABLED  = os.environ.get("CODEFENDER_FUZZY_DB_ENABLED", "1") != "0"
FUZZY_DB_MAX_OPEN = int(os.environ.get("CODEFENDER_FUZZY_DB_MAX_OPEN", "16"))
FUZZY_DB_MAX_BYTES = int(os.environ.get("CODEFENDER_FUZZY_DB_MAX_BYTES", str(256 * 1024)))
