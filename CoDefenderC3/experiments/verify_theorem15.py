"""
Experiment: Theorem 1.5 验证 — 真实 PE 文件上的特征泄露测量
==============================================================

核心预测: 对 Type-P 变换 (代码段内等长修改),
  结构特征 (section/imports/header/exports/datadirs) 精确不变 (ε_P = 0)
  表层特征 (histogram/byteentropy/strings) 发生有界变化 (≤ 2m/n)

验证方法:
  1. 收集真实 PE 文件 (良性+恶意)
  2. 对每个文件施加 Type-P 变换:
     - NOP 填充 (将 .text 段 padding 区域替换为 0x90)
     - 等价指令替换 (将 MOV reg,0 替换为 XOR reg,reg 等)
     - 随机代码段字节翻转 (最极端的 Type-P)
  3. 提取变换前后的 EMBER 2381d 特征
  4. 逐特征组计算 L1 距离
  5. 统计: 结构特征组的距离是否为精确零

用法:
  python -m experiments.verify_theorem15 --pe_dir /path/to/pe/files
  python -m experiments.verify_theorem15 --pe_dir F:\\malware_samples\\
"""
import os
import sys
import json
import struct
import argparse
import numpy as np

# ── EMBER 特征提取 ──
def _get_extractor():
    """获取 EMBER PEFeatureExtractor"""
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "_tools", "ember"),
        "/tmp/ember",
        os.path.join(os.environ.get("APPDATA", ""), "ember"),
    ]
    for p in paths:
        feat_path = os.path.join(p, "ember", "features.py")
        if os.path.exists(feat_path):
            sys.path.insert(0, p)
            break

    # 直接导入 features 模块, 绕过 lightgbm 依赖
    import importlib.util
    for p in paths:
        feat_path = os.path.join(p, "ember", "features.py")
        if os.path.exists(feat_path):
            spec = importlib.util.spec_from_file_location("ember_features", feat_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.PEFeatureExtractor(2)
    raise ImportError("找不到 ember features.py, 请 git clone https://github.com/elastic/ember.git")


# ── EMBER 特征分组 (与论文对应) ──
FEATURE_GROUPS = {
    # 表层 (Surface): 读取所有字节, 对代码段修改敏感
    "histogram":      (0, 256),      # ByteHistogram
    "byteentropy":    (256, 512),    # ByteEntropyHistogram
    "strings":        (512, 616),    # StringExtractor
    # 结构 (Structure): 仅读取 PE 结构化字段, 对代码段修改应不变
    "general":        (616, 626),    # GeneralInfo (PE header)
    "header":         (626, 688),    # HeaderInfo (COFF/Optional header)
    "section":        (688, 943),    # SectionInfo (节表)
    "imports":        (943, 2223),   # ImportsInfo (导入表)
    "exports":        (2223, 2351),  # ExportsInfo (导出表)
    "datadirectories": (2351, 2381), # DataDirectories
}

SURFACE_GROUPS = ["histogram", "byteentropy", "strings"]
STRUCTURE_GROUPS = ["general", "header", "section", "imports", "exports", "datadirectories"]


# ── Type-P 变换: 代码段内修改 ──

def _find_text_section(pe_bytes):
    """定位 .text 节在文件中的偏移和大小"""
    try:
        import lief
        pe = lief.parse(pe_bytes)
        if pe is None:
            return None, None
        for section in pe.sections:
            if ".text" in section.name or section.has_characteristic(
                    lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE):
                offset = section.offset
                size = section.size
                if offset > 0 and size > 0:
                    return offset, size
    except Exception:
        pass
    return None, None


def transform_nop_fill(pe_bytes, ratio=0.05):
    """
    Type-P 变换: NOP 填充
    将 .text 节末尾的 ratio 比例字节替换为 NOP (0x90)
    满足条件: (a) 仅修改代码段 (b) 不改 PE 头 (c) 等长替换
    """
    offset, size = _find_text_section(pe_bytes)
    if offset is None:
        return None, 0
    # 防止节声明大小超出文件实际长度
    actual_end = min(offset + size, len(pe_bytes))
    actual_size = actual_end - offset
    if actual_size <= 0:
        return None, 0
    data = bytearray(pe_bytes)
    n_modify = max(1, int(actual_size * ratio))
    n_modify = min(n_modify, actual_size)  # 不能超过实际可用大小
    start = offset + actual_size - n_modify
    for i in range(n_modify):
        data[start + i] = 0x90
    return bytes(data), n_modify


def transform_byte_flip(pe_bytes, ratio=0.05):
    """
    Type-P 变换: 随机字节翻转 (最极端的代码段内修改)
    满足条件: (a) 仅修改代码段 (b) 不改 PE 头 (c) 等长替换
    """
    offset, size = _find_text_section(pe_bytes)
    if offset is None:
        return None, 0
    actual_end = min(offset + size, len(pe_bytes))
    actual_size = actual_end - offset
    if actual_size <= 0:
        return None, 0
    data = bytearray(pe_bytes)
    rng = np.random.RandomState(42)
    n_modify = max(1, min(int(actual_size * ratio), actual_size))
    positions = rng.choice(actual_size, n_modify, replace=False) + offset
    for pos in positions:
        data[pos] = (data[pos] + rng.randint(1, 256)) % 256
    return bytes(data), n_modify


def transform_header_modify(pe_bytes):
    """
    Type-S 变换 (对照组): 修改 PE 头中的 TimeDateStamp
    这是结构型修改, 应导致结构特征变化
    """
    try:
        import lief
        pe = lief.parse(pe_bytes)
        if pe is None:
            return None
        pe.header.time_date_stamps = 0x12345678
        builder = lief.PE.Builder(pe)
        builder.build()
        return bytes(builder.get_build())
    except Exception:
        return None


# ── 主实验 ──

def run_verification(pe_dir, max_files=100):
    """
    在真实 PE 文件上验证 Theorem 1.5

    对每个 PE 文件:
      1. 提取原始 EMBER 特征
      2. 施加 Type-P 变换, 提取变换后特征
      3. 计算每个特征组的 L1 距离
      4. 统计: 结构特征组距离是否为零
    """
    extractor = _get_extractor()

    print("=" * 70)
    print("  Theorem 1.5 验证: 真实 PE 文件上的特征泄露测量")
    print("=" * 70)
    print(f"  PE 目录: {pe_dir}")
    print(f"  EMBER 特征维度: {extractor.dim}")

    # 收集 PE 文件
    pe_files = []
    for root, dirs, files in os.walk(pe_dir):
        for f in files:
            fpath = os.path.join(root, f)
            if os.path.getsize(fpath) > 1024:  # 至少 1KB
                pe_files.append(fpath)
            if len(pe_files) >= max_files:
                break
        if len(pe_files) >= max_files:
            break
    print(f"  找到 {len(pe_files)} 个文件")

    transforms = {
        "NOP_fill_1%":   lambda b: transform_nop_fill(b, 0.01),
        "NOP_fill_5%":   lambda b: transform_nop_fill(b, 0.05),
        "NOP_fill_10%":  lambda b: transform_nop_fill(b, 0.10),
        "ByteFlip_1%":   lambda b: transform_byte_flip(b, 0.01),
        "ByteFlip_5%":   lambda b: transform_byte_flip(b, 0.05),
        "ByteFlip_10%":  lambda b: transform_byte_flip(b, 0.10),
    }

    results = {tn: {gn: [] for gn in FEATURE_GROUPS} for tn in transforms}
    results["HeaderModify"] = {gn: [] for gn in FEATURE_GROUPS}
    file_count = {tn: 0 for tn in list(transforms.keys()) + ["HeaderModify"]}

    for fi, fpath in enumerate(pe_files):
        try:
            with open(fpath, "rb") as f:
                pe_bytes = f.read()
        except Exception:
            continue

        # 原始特征
        try:
            feat_orig = np.array(extractor.feature_vector(pe_bytes), dtype=np.float64)
        except Exception:
            continue

        if len(feat_orig) != extractor.dim:
            continue

        # Type-P 变换
        for tn, tfn in transforms.items():
            try:
                modified, n_mod = tfn(pe_bytes)
                if modified is None:
                    continue
                feat_mod = np.array(extractor.feature_vector(modified), dtype=np.float64)
                if len(feat_mod) != extractor.dim:
                    continue

                for gn, (lo, hi) in FEATURE_GROUPS.items():
                    dist = float(np.sum(np.abs(feat_mod[lo:hi] - feat_orig[lo:hi])))
                    results[tn][gn].append(dist)
                file_count[tn] += 1
            except Exception:
                continue

        # Type-S 对照
        try:
            modified_s = transform_header_modify(pe_bytes)
            if modified_s is not None:
                feat_s = np.array(extractor.feature_vector(modified_s), dtype=np.float64)
                if len(feat_s) == extractor.dim:
                    for gn, (lo, hi) in FEATURE_GROUPS.items():
                        dist = float(np.sum(np.abs(feat_s[lo:hi] - feat_orig[lo:hi])))
                        results["HeaderModify"][gn].append(dist)
                    file_count["HeaderModify"] += 1
        except Exception:
            pass

        if (fi + 1) % 10 == 0:
            print(f"\r  处理: {fi+1}/{len(pe_files)}", end="", flush=True)

    print(f"\n\n{'=' * 70}")
    print("  结果")
    print("=" * 70)

    # ── 统计汇总 ──
    print("\n  ── Type-P 变换 (应满足 ε_P(struct) = 0) ──\n")
    print(f"  {'变换':<20s} {'特征组':<18s} {'层级':<10s} "
          f"{'均值':<12s} {'标准差':<12s} {'最大值':<12s} {'零率':>8s} {'N':>5s}")
    print("  " + "─" * 100)

    theorem_holds = True
    for tn in transforms:
        if file_count[tn] == 0:
            continue
        for gn in FEATURE_GROUPS:
            vals = results[tn][gn]
            if not vals:
                continue
            level = "surface" if gn in SURFACE_GROUPS else "structure"
            mean = np.mean(vals)
            std = np.std(vals)
            mx = np.max(vals)
            zero_rate = np.mean(np.array(vals) == 0.0) * 100

            marker = ""
            if level == "structure":
                if zero_rate < 100:
                    marker = " ⚠ THEOREM VIOLATION"
                    theorem_holds = False
                else:
                    marker = " ✓"

            print(f"  {tn:<20s} {gn:<18s} {level:<10s} "
                  f"{mean:<12.6f} {std:<12.6f} {mx:<12.6f} {zero_rate:>7.1f}% {len(vals):>5d}{marker}")

    # Type-S 对照
    print(f"\n  ── Type-S 对照 (结构特征应变化) ──\n")
    if file_count["HeaderModify"] > 0:
        for gn in FEATURE_GROUPS:
            vals = results["HeaderModify"][gn]
            if not vals:
                continue
            level = "surface" if gn in SURFACE_GROUPS else "structure"
            mean = np.mean(vals)
            nonzero = np.mean(np.array(vals) > 0) * 100
            print(f"  HeaderModify       {gn:<18s} {level:<10s} "
                  f"mean={mean:<12.6f} nonzero={nonzero:.1f}%")

    # ── 结论 ──
    print(f"\n{'=' * 70}")
    if theorem_holds:
        print("  ✓ Theorem 1.5 验证通过:")
        print("    所有 Type-P 变换的结构特征距离 = 精确零")
        print("    ε_P(struct) = 0 在所有测试文件上成立")
    else:
        print("  ⚠ Theorem 1.5 部分违反:")
        print("    存在 Type-P 变换导致结构特征变化的情况")
        print("    需要检查: 变换是否违反了条件 (a)(b)(c)")
    print("=" * 70)

    # 保存结果
    out = {
        "theorem_holds": theorem_holds,
        "file_counts": file_count,
        "summary": {}
    }
    for tn in list(transforms.keys()) + ["HeaderModify"]:
        out["summary"][tn] = {}
        for gn in FEATURE_GROUPS:
            vals = results[tn][gn]
            if vals:
                out["summary"][tn][gn] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "max": float(np.max(vals)),
                    "zero_rate": float(np.mean(np.array(vals) == 0)),
                    "n": len(vals),
                }
    return out


def main():
    parser = argparse.ArgumentParser(description="Theorem 1.5 验证")
    parser.add_argument("--pe_dir", required=True, help="PE 文件目录")
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--output", default=None, help="结果 JSON 输出路径")
    args = parser.parse_args()

    out = run_verification(args.pe_dir, args.max_files)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  结果保存: {args.output}")


if __name__ == "__main__":
    main()
