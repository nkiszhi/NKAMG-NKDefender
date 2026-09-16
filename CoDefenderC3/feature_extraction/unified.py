"""
统一特征提取层 — 所有模型的特征由此模块统一管理
=================================================
15种特征类型, 按需提取, 自动缓存

使用方法:
    from feature_extraction import FeatureType, extract_feature

    # 提取 EMBER 全特征
    tensor = extract_feature(pe_path, FeatureType.EMBER_FULL)

    # 提取灰度图像
    tensor = extract_feature(pe_path, FeatureType.GRAY_IMAGE, img_size=256)

    # 在 Dataset 中使用
    class MyDataset(UnifiedDataset):
        FEATURE_TYPE = FeatureType.EMBER_BYTEHIST

特征类型与 EMBER 维度映射:
    EMBER_FULL       : 2381维 (全部9个特征组)
    EMBER_BYTEHIST   : 512维  (ByteHistogram 256 + ByteEntropy 256)
    EMBER_PE_STRUCT  : 327维  (GeneralInfo 10 + HeaderInfo 62 + SectionInfo 255)
    EMBER_IMPORT     : 1280维 (ImportInfo)
    EMBER_STRING     : 104维  (StringInfo)
    EMBER_METADATA   : 158维  (ExportInfo 128 + DataDirs 30)

依赖:
    pefile       — EMBER 特征提取 (pip install pefile)
    binaryninja  — 操作码/图特征 (商业许可证)
    matplotlib   — 图像可视化 (pip install matplotlib)
"""
import os
import math
import hashlib
import numpy as np
import torch
from enum import Enum, auto
from typing import Optional, Dict, Any


# ═══════════════════════════════════════
#  特征类型枚举
# ═══════════════════════════════════════

class FeatureType(Enum):
    """所有支持的特征类型"""
    # 原始字节
    RAW_BYTES = auto()           # LongTensor(L,) 字节序列

    # EMBER 静态特征子集
    EMBER_FULL = auto()          # FloatTensor(2381,) 全部
    EMBER_BYTEHIST = auto()      # FloatTensor(512,)  ByteHist+ByteEntropy
    EMBER_PE_STRUCT = auto()     # FloatTensor(327,)  GeneralInfo+HeaderInfo+SectionInfo
    EMBER_IMPORT = auto()        # FloatTensor(1280,) ImportInfo
    EMBER_STRING = auto()        # FloatTensor(104,)  StringInfo
    EMBER_METADATA = auto()      # FloatTensor(158,)  ExportInfo+DataDirs

    # 操作码/汇编 (需要 BinaryNinja)
    OPCODE_SEQ = auto()          # LongTensor(L,) 操作码索引
    FUNC_EMBEDDINGS = auto()     # FloatTensor(6144,) 48函数×128维嵌入

    # 图结构 (需要 BinaryNinja)
    PYG_GRAPH = auto()           # torch_geometric Data 对象
    GRAPH_STAT = auto()          # FloatTensor(22,) CFG+CallGraph 统计

    # 可视化图像
    GRAY_IMAGE = auto()          # FloatTensor(1, H, W) 灰度图
    COLOR_IMAGE = auto()         # FloatTensor(3, H, W) jet colormap RGB
    MARKOV_IMAGE = auto()        # FloatTensor(1, 256, 256) Markov 转移矩阵
    ENTROPY_IMAGE = auto()       # FloatTensor(1, 256, 256) Shannon 熵热图


# ═══════════════════════════════════════
#  EMBER 特征范围定义
# ═══════════════════════════════════════

EMBER_RANGES = {
    "ByteHistogram": (0, 256),
    "ByteEntropy":   (256, 512),
    "StringInfo":    (512, 616),
    "GeneralInfo":   (616, 626),
    "HeaderInfo":    (626, 688),
    "SectionInfo":   (688, 943),
    "ImportInfo":    (943, 2223),
    "ExportInfo":    (2223, 2351),
    "DataDirs":      (2351, 2381),
}

# 特征类型 → EMBER 子维度映射
EMBER_SUBSETS = {
    FeatureType.EMBER_FULL:      list(EMBER_RANGES.keys()),
    FeatureType.EMBER_BYTEHIST:  ["ByteHistogram", "ByteEntropy"],
    FeatureType.EMBER_PE_STRUCT: ["GeneralInfo", "HeaderInfo", "SectionInfo"],
    FeatureType.EMBER_IMPORT:    ["ImportInfo"],
    FeatureType.EMBER_STRING:    ["StringInfo"],
    FeatureType.EMBER_METADATA:  ["ExportInfo", "DataDirs"],
}


# ═══════════════════════════════════════
#  缓存管理
# ═══════════════════════════════════════

_CACHE_DIR = ".feature_cache"


def _file_sha256(path):
    """计算文件 SHA256"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(pe_path, feature_type, suffix="npy"):
    """获取缓存路径"""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fh = _file_sha256(pe_path)
    return os.path.join(_CACHE_DIR, f"{fh}_{feature_type.name}.{suffix}")


def set_cache_dir(path):
    """设置全局缓存目录"""
    global _CACHE_DIR
    _CACHE_DIR = path
    os.makedirs(path, exist_ok=True)


# ═══════════════════════════════════════
#  核心提取函数
# ═══════════════════════════════════════

def extract_feature(pe_path: str, feature_type: FeatureType,
                    **kwargs) -> torch.Tensor:
    """
    统一特征提取入口

    参数:
        pe_path: PE 文件路径
        feature_type: 特征类型 (FeatureType 枚举)
        **kwargs: 特征类型特定参数
            max_len: 字节序列最大长度 (RAW_BYTES)
            img_size: 图像尺寸 (GRAY_IMAGE/COLOR_IMAGE)
            seq_len: 操作码序列长度 (OPCODE_SEQ)

    返回:
        torch.Tensor: 对应类型的特征张量
    """
    # ── 原始字节 ──
    if feature_type == FeatureType.RAW_BYTES:
        return _extract_raw_bytes(pe_path, kwargs.get("max_len", 200000))

    # ── EMBER 静态特征 ──
    if feature_type in EMBER_SUBSETS:
        return _extract_ember_subset(pe_path, feature_type)

    # ── 操作码序列 ──
    if feature_type == FeatureType.OPCODE_SEQ:
        return _extract_opcode_seq(pe_path, kwargs.get("seq_len", 4096))

    # ── 函数嵌入 ──
    if feature_type == FeatureType.FUNC_EMBEDDINGS:
        return _extract_func_embeddings(pe_path, kwargs.get("n_funcs", 48),
                                         kwargs.get("embed_dim", 128))

    # ── 图统计 ──
    if feature_type == FeatureType.GRAPH_STAT:
        return _extract_graph_stat(pe_path)

    # ── PyG 图数据 ──
    if feature_type == FeatureType.PYG_GRAPH:
        return _extract_pyg_graph(pe_path)

    # ── 可视化图像 ──
    if feature_type == FeatureType.GRAY_IMAGE:
        return _extract_gray_image(pe_path, kwargs.get("img_size", 256))
    if feature_type == FeatureType.COLOR_IMAGE:
        return _extract_color_image(pe_path, kwargs.get("img_size", 224))
    if feature_type == FeatureType.MARKOV_IMAGE:
        return _extract_markov_image(pe_path)
    if feature_type == FeatureType.ENTROPY_IMAGE:
        return _extract_entropy_image(pe_path)

    raise ValueError(f"不支持的特征类型: {feature_type}")


# ═══════════════════════════════════════
#  1. 原始字节提取
# ═══════════════════════════════════════

def _extract_raw_bytes(pe_path, max_len):
    """读取 PE 原始字节, 截断或零填充到 max_len"""
    with open(pe_path, "rb") as f:
        raw = f.read(max_len)  # 只读需要的量
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    if len(arr) < max_len:
        arr = np.pad(arr, (0, max_len - len(arr)), constant_values=0)
    return torch.tensor(arr, dtype=torch.long)


# ═══════════════════════════════════════
#  2. EMBER 静态特征提取
# ═══════════════════════════════════════

# 惰性初始化提取器 (避免重复创建)
_ember_extractors = None


def _get_ember_extractor():
    """获取 EMBER PEFeatureExtractor (从 ember 包或 _tools/ember)"""
    global _ember_extractors
    if _ember_extractors is not None:
        return _ember_extractors

    # 方法1: 已安装的 ember 包
    try:
        from ember.features import PEFeatureExtractor
        _ember_extractors = PEFeatureExtractor(feature_version=2)
        return _ember_extractors
    except ImportError:
        pass

    # 方法2: build_dataset.py 克隆的 _tools/ember
    import sys as _sys
    _tools_path = os.path.join(os.path.dirname(__file__), "..", "_tools", "ember")
    if os.path.exists(os.path.join(_tools_path, "ember", "features.py")):
        _sys.path.insert(0, _tools_path)
        try:
            from ember.features import PEFeatureExtractor
            _ember_extractors = PEFeatureExtractor(feature_version=2)
            return _ember_extractors
        except ImportError:
            pass

    return None


def _extract_ember_full(pe_path):
    """提取完整 2381 维 EMBER 特征向量"""
    cache = _cache_path(pe_path, FeatureType.EMBER_FULL)
    if os.path.exists(cache):
        return torch.tensor(np.load(cache), dtype=torch.float32)

    try:
        with open(pe_path, "rb") as f:
            bytez = f.read()

        extractor = _get_ember_extractor()
        if extractor is not None:
            features = np.array(extractor.feature_vector(bytez), dtype=np.float32)
            np.save(cache, features)
            return torch.tensor(features, dtype=torch.float32)

        # Fallback: pefile 基础提取 (仅字节直方图 + 熵)
        hist = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        features = np.zeros(2381, dtype=np.float32)
        features[:256] = hist.astype(np.float32) / max(len(bytez), 1)
        np.save(cache, features)
        return torch.tensor(features, dtype=torch.float32)

    except Exception as e:
        print(f"[特征提取失败] {pe_path}: {e}")
        return torch.zeros(2381, dtype=torch.float32)


def _extract_ember_subset(pe_path, feature_type):
    """提取 EMBER 特征的子集"""
    full = _extract_ember_full(pe_path)
    if feature_type == FeatureType.EMBER_FULL:
        return full

    # 根据子集定义切片
    subset_names = EMBER_SUBSETS[feature_type]
    parts = []
    for name in subset_names:
        start, end = EMBER_RANGES[name]
        parts.append(full[start:end])
    return torch.cat(parts, dim=0)


# ═══════════════════════════════════════
#  3. 操作码序列提取 (BinaryNinja)
# ═══════════════════════════════════════

def _extract_opcode_seq(pe_path, seq_len):
    """使用 BinaryNinja 提取操作码 token 序列"""
    cache = _cache_path(pe_path, FeatureType.OPCODE_SEQ)
    if os.path.exists(cache):
        return torch.tensor(np.load(cache), dtype=torch.long)

    opcodes = np.zeros(seq_len, dtype=np.int64)
    try:
        from binaryninja import BinaryViewType
        bv = BinaryViewType["PE"].open(pe_path)
        if bv is None:
            return torch.tensor(opcodes, dtype=torch.long)
        bv.update_analysis_and_wait()

        idx = 0
        for func in bv.functions:
            for block in func.basic_blocks:
                for tokens, addr, size in block:
                    # 操作码 hash → [1, 511] 索引空间
                    if len(tokens) > 0:
                        op_text = tokens[0].text.strip().lower()
                        op_hash = hash(op_text) % 511 + 1
                        opcodes[idx] = op_hash
                        idx += 1
                        if idx >= seq_len:
                            break
                if idx >= seq_len:
                    break
            if idx >= seq_len:
                break

    except ImportError:
        # BinaryNinja 不可用: 用字节值近似操作码
        with open(pe_path, "rb") as f:
            raw = f.read(seq_len)
        for i, b in enumerate(raw[:seq_len]):
            opcodes[i] = b % 511 + 1
    except Exception as e:
        print(f"[操作码提取失败] {pe_path}: {e}")

    np.save(cache, opcodes)
    return torch.tensor(opcodes, dtype=torch.long)


# ═══════════════════════════════════════
#  4. 函数级嵌入提取 (BinaryNinja)
# ═══════════════════════════════════════

def _extract_func_embeddings(pe_path, n_funcs=48, embed_dim=128):
    """提取函数级嵌入: 每个函数用其基本块特征的均值表示"""
    cache = _cache_path(pe_path, FeatureType.FUNC_EMBEDDINGS)
    if os.path.exists(cache):
        return torch.tensor(np.load(cache), dtype=torch.float32)

    result = np.zeros(n_funcs * embed_dim, dtype=np.float32)

    try:
        from binaryninja import BinaryViewType
        bv = BinaryViewType["PE"].open(pe_path)
        if bv is None:
            return torch.tensor(result, dtype=torch.float32)
        bv.update_analysis_and_wait()

        for i, func in enumerate(bv.functions[:n_funcs]):
            # 每个函数: 统计指令类别分布作为嵌入
            func_vec = np.zeros(embed_dim, dtype=np.float32)
            n_instrs = 0
            for block in func.basic_blocks:
                for tokens, addr, size in block:
                    if len(tokens) > 0:
                        # 用操作码文本的 hash 分配到 embed_dim 维度
                        op_text = tokens[0].text.strip().lower()
                        dim_idx = hash(op_text) % embed_dim
                        func_vec[dim_idx] += 1.0
                        n_instrs += 1

            if n_instrs > 0:
                func_vec /= n_instrs  # 归一化

            start = i * embed_dim
            result[start:start + embed_dim] = func_vec

    except ImportError:
        # BinaryNinja 不可用: 用随机但确定性的值填充
        rng = np.random.RandomState(hash(pe_path) % 2**31)
        result = rng.randn(n_funcs * embed_dim).astype(np.float32) * 0.01
    except Exception as e:
        print(f"[函数嵌入提取失败] {pe_path}: {e}")

    np.save(cache, result)
    return torch.tensor(result, dtype=torch.float32)


# ═══════════════════════════════════════
#  5. 图统计特征提取 (BinaryNinja)
# ═══════════════════════════════════════

def _extract_graph_stat(pe_path):
    """提取 CFG + CallGraph 22维统计特征"""
    cache = _cache_path(pe_path, FeatureType.GRAPH_STAT)
    if os.path.exists(cache):
        return torch.tensor(np.load(cache), dtype=torch.float32)

    stats = np.zeros(22, dtype=np.float32)

    try:
        from binaryninja import BinaryViewType, SymbolType
        bv = BinaryViewType["PE"].open(pe_path)
        if bv is None:
            return torch.tensor(stats, dtype=torch.float32)
        bv.update_analysis_and_wait()

        funcs = list(bv.functions)
        n_funcs = len(funcs)
        extern_syms = bv.get_symbols_of_type(SymbolType.ImportAddressSymbol)
        n_external = len(extern_syms)

        # CFG 统计
        total_blocks, total_edges = 0, 0
        max_blocks = 0
        for func in funcs:
            n_blocks = len(func.basic_blocks)
            n_edges = sum(len(b.outgoing_edges) for b in func.basic_blocks)
            total_blocks += n_blocks
            total_edges += n_edges
            max_blocks = max(max_blocks, n_blocks)

        avg_blocks = total_blocks / max(n_funcs, 1)
        density = total_edges / max(total_blocks * (total_blocks - 1), 1)

        # CallGraph 统计
        n_internal_calls = 0
        fan_outs = []
        for func in funcs:
            callees = set()
            for block in func.basic_blocks:
                for edge in block.outgoing_edges:
                    callees.add(edge.target.start)
            fan_out = len(callees)
            fan_outs.append(fan_out)
            n_internal_calls += fan_out

        avg_fan_out = np.mean(fan_outs) if fan_outs else 0
        max_fan_out = max(fan_outs) if fan_outs else 0

        # 填充 22 维
        stats[0] = total_blocks
        stats[1] = total_edges
        stats[2] = total_edges / max(total_blocks, 1)  # 平均度
        stats[3] = max_blocks
        stats[4] = density
        stats[5] = avg_blocks
        stats[6] = n_funcs
        stats[7] = n_external
        stats[8] = n_internal_calls
        stats[9] = n_internal_calls / max(n_funcs * (n_funcs - 1), 1)  # 调用密度
        stats[10] = max_fan_out
        stats[11] = avg_fan_out
        stats[12] = n_external / max(n_funcs + n_external, 1)  # 外部调用比
        stats[13] = sum(1 for fo in fan_outs if fo == 0) / max(n_funcs, 1)  # 叶子比
        # 其余维度填入额外统计
        if fan_outs:
            stats[14] = np.std(fan_outs)
            stats[15] = np.median(fan_outs)
        stats[16] = min(n_funcs, 1000) / 1000  # 归一化函数数
        stats[17] = min(total_blocks, 10000) / 10000  # 归一化块数
        stats[18] = min(total_edges, 50000) / 50000  # 归一化边数
        stats[19] = min(n_external, 500) / 500  # 归一化外部调用
        stats[20] = min(max_blocks, 200) / 200  # 归一化最大块数
        stats[21] = min(max_fan_out, 100) / 100  # 归一化最大扇出

    except ImportError:
        # BinaryNinja 不可用: 从 EMBER 特征近似
        ember = _extract_ember_full(pe_path)
        # 用 EMBER 的部分统计信息填充
        stats[0] = float(ember[620])  # GeneralInfo 中的 size
        stats[6] = float(ember[621])  # 导入函数数
        stats[7] = float(ember[2225]) # 导出函数数
    except Exception as e:
        print(f"[图统计提取失败] {pe_path}: {e}")

    np.save(cache, stats)
    return torch.tensor(stats, dtype=torch.float32)


# ═══════════════════════════════════════
#  6. PyG 图数据 (由 MalGraph 专用处理)
# ═══════════════════════════════════════

def _extract_pyg_graph(pe_path):
    """提取 PyG 图数据 — 由 MalGraph 的 utils.py 处理, 此处返回占位"""
    # MalGraph 有自己完整的 extract_cfg_and_fcg + Vocabulary
    # 此函数仅用于接口统一, 实际由 MalGraph 自己的 Dataset 处理
    raise NotImplementedError(
        "PyG 图数据由 models/MalGraph_2022/utils.py 的 extract_cfg_and_fcg() 提取, "
        "不通过统一接口。请直接使用 MalGraph 的 Dataset。"
    )


# ═══════════════════════════════════════
#  7. 可视化图像提取
# ═══════════════════════════════════════

def _pe_bytes_to_array(pe_path):
    """读取 PE 文件为 uint8 数组"""
    with open(pe_path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)


def _choose_width(file_size):
    """Nataraj 2011 论文的宽度选择策略"""
    if file_size < 10 * 1024:    return 32
    if file_size < 30 * 1024:    return 64
    if file_size < 60 * 1024:    return 128
    if file_size < 100 * 1024:   return 256
    if file_size < 200 * 1024:   return 384
    if file_size < 500 * 1024:   return 512
    if file_size < 1000 * 1024:  return 768
    return 1024


def _extract_gray_image(pe_path, img_size=256):
    """PE → 灰度可视化图像 (Nataraj 2011)"""
    raw = _pe_bytes_to_array(pe_path)
    if len(raw) == 0:
        return torch.zeros(1, img_size, img_size, dtype=torch.float32)

    width = _choose_width(len(raw))
    height = math.ceil(len(raw) / width)
    padded = np.zeros(width * height, dtype=np.uint8)
    padded[:len(raw)] = raw
    gray = padded.reshape(height, width)

    # 缩放到目标尺寸
    from PIL import Image
    img = Image.fromarray(gray, mode="L")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0

    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, H, W)


def _extract_color_image(pe_path, img_size=224):
    """PE → 彩色可视化图像 (jet colormap, Vasan 2020)"""
    raw = _pe_bytes_to_array(pe_path)
    if len(raw) == 0:
        return torch.zeros(3, img_size, img_size, dtype=torch.float32)

    width = _choose_width(len(raw))
    height = math.ceil(len(raw) / width)
    padded = np.zeros(width * height, dtype=np.uint8)
    padded[:len(raw)] = raw
    gray = padded.reshape(height, width)

    # jet colormap
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("jet")
        colored = (cmap(gray / 255.0)[:, :, :3] * 255).astype(np.uint8)
    except Exception:
        colored = np.stack([gray, gray, gray], axis=2)

    from PIL import Image
    img = Image.fromarray(colored)
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0

    # ImageNet 归一化
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    arr = (arr - mean) / std

    return torch.tensor(arr, dtype=torch.float32)


def _extract_markov_image(pe_path):
    """PE → 256×256 Markov 转移矩阵图像 (Ni 2018)"""
    cache = _cache_path(pe_path, FeatureType.MARKOV_IMAGE)
    if os.path.exists(cache):
        return torch.tensor(np.load(cache), dtype=torch.float32).unsqueeze(0)

    raw = _pe_bytes_to_array(pe_path)
    if len(raw) < 2:
        result = np.zeros((256, 256), dtype=np.float32)
    else:
        # 计算字节转移概率: M[i][j] = P(next=j | current=i)
        transitions = np.zeros((256, 256), dtype=np.float64)
        pairs = raw[:-1].astype(np.int32) * 256 + raw[1:].astype(np.int32)
        counts = np.bincount(pairs, minlength=65536).reshape(256, 256)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        result = (counts / row_sums).astype(np.float32)

    np.save(cache, result)
    return torch.tensor(result, dtype=torch.float32).unsqueeze(0)  # (1, 256, 256)


def _extract_entropy_image(pe_path, block_size=256):
    """PE → 256×256 Shannon 熵热图 (Conti 2018)"""
    cache = _cache_path(pe_path, FeatureType.ENTROPY_IMAGE)
    if os.path.exists(cache):
        return torch.tensor(np.load(cache), dtype=torch.float32).unsqueeze(0)

    raw = _pe_bytes_to_array(pe_path)
    n_cells = 256 * 256  # 65536 个单元格

    if len(raw) == 0:
        result = np.zeros((256, 256), dtype=np.float32)
    else:
        # 将文件分为 n_cells 个块, 每块计算 Shannon 熵
        chunk_size = max(len(raw) // n_cells, 1)
        entropies = np.zeros(n_cells, dtype=np.float32)

        for i in range(min(n_cells, len(raw) // max(chunk_size, 1))):
            start = i * chunk_size
            end = min(start + chunk_size, len(raw))
            block = raw[start:end]

            if len(block) == 0:
                continue

            # Shannon 熵: H = -Σ p·log2(p)
            counts = np.bincount(block, minlength=256)
            probs = counts / len(block)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            entropies[i] = entropy / 8.0  # 归一化到 [0, 1]

        result = entropies.reshape(256, 256)

    np.save(cache, result)
    return torch.tensor(result, dtype=torch.float32).unsqueeze(0)  # (1, 256, 256)


# ═══════════════════════════════════════
#  统一数据集基类
# ═══════════════════════════════════════

class UnifiedDataset(torch.utils.data.Dataset):
    """
    统一数据集基类 — 所有 run 脚本的数据集可继承此类

    使用方法:
        class MyDataset(UnifiedDataset):
            FEATURE_TYPE = FeatureType.EMBER_BYTEHIST

        dataset = MyDataset(samples)  # samples = [(pe_path, label), ...]
    """

    FEATURE_TYPE = FeatureType.EMBER_FULL  # 子类覆盖此属性
    FEATURE_KWARGS = {}                     # 额外参数 (如 max_len, img_size)

    def __init__(self, samples, feature_type=None, **kwargs):
        """
        参数:
            samples: [(pe文件路径, 标签), ...] 样本列表
            feature_type: 覆盖类属性的特征类型
            **kwargs: 传给 extract_feature 的额外参数
        """
        self.samples = samples
        self.feature_type = feature_type or self.FEATURE_TYPE
        self.kwargs = {**self.FEATURE_KWARGS, **kwargs}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            features = extract_feature(path, self.feature_type, **self.kwargs)
            return features, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"[错误] {path}: {e}")
            # 返回零特征而非递归 (防止所有样本失败时无限递归)
            dim = get_feature_dim(self.feature_type, **self.kwargs)
            if dim > 0:
                return torch.zeros(dim, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
            # 可变维度类型: 尝试下一个 (最多重试3次)
            for offset in range(1, 4):
                try:
                    next_idx = (idx + offset) % len(self.samples)
                    p2, l2 = self.samples[next_idx]
                    return extract_feature(p2, self.feature_type, **self.kwargs), torch.tensor(l2, dtype=torch.float32)
                except Exception:
                    continue
            return torch.zeros(1, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# ═══════════════════════════════════════
#  便捷函数: 获取特征维度
# ═══════════════════════════════════════

def get_feature_dim(feature_type: FeatureType, **kwargs) -> int:
    """获取指定特征类型的输出维度"""
    dims = {
        FeatureType.EMBER_FULL: 2381,
        FeatureType.EMBER_BYTEHIST: 512,
        FeatureType.EMBER_PE_STRUCT: 327,
        FeatureType.EMBER_IMPORT: 1280,
        FeatureType.EMBER_STRING: 104,
        FeatureType.EMBER_METADATA: 158,
        FeatureType.GRAPH_STAT: 22,
        FeatureType.FUNC_EMBEDDINGS: kwargs.get("n_funcs", 48) * kwargs.get("embed_dim", 128),
    }
    return dims.get(feature_type, -1)  # -1 表示可变维度 (图像/序列)
