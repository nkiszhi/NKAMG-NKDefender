#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_minimal_dataset.py — 构建"最小可训练 PE_RAW 数据集"，与 ensemble 模型的视角严格对齐

为什么要这个脚本
----------------
`run_all.py` 的 PE_RAW 管线要求数据目录里同时有:
    metadata.json    (feature_type=PE_RAW + train_months)
    metadata.jsonl   (每行 {path,label,month,family})
缺任何一个，`create_loader()` 要么直接抛 FileNotFoundError，要么
`discover_samples()` 拿不到 month → `assign_months()` 按 mtime 排序切月，
出现"某个月份全是同一类样本"的静默崩塌（train_months 里 0 个恶意样本，不报错）。

产出结构:
    <out>/
      malicious/<sha256>          ← 真实样本（复制，不硬链接，兼容 VMware 共享目录）
      benign/<sha256>
      metadata.json               ← feature_type=PE_RAW + 视角期望 + 口径说明
      metadata.jsonl              ← 逐样本标注（缺省写绝对路径，见下方"路径陷阱"）
      manifest.json               ← 构建指纹（源目录、样本哈希、视角维度、月度配比）
      .cache/per_month/*.npy      ← 单月合并特征（N 视角 + y）
      .cache/per_sample/*.npz     ← 单样本特征（--drop-per-sample 可删）

⚠ 路径陷阱（本脚本第一版就踩了，务必理解）
------------------------------------------
`data_loader.discover_samples()` 对相对路径做 `os.path.join(data_dir, rel)`。
若 rel 用 POSIX 风格（"benign/xxx"），Windows 上得到的是**混合分隔符**路径
    "D:\\VM_Share\\...\\minimal_dataset\\benign/xxx"
而 `PELoader._sample_cache_path()` 直接对这个字符串取 md5 —— 没有 normpath。
于是"用 normpath 算出来的键"和"加载器实际查找的键"**不是同一个**：
提取写进了 A，加载器去找 B，B 找不到 → 判定"无缓存"→ 重新提取，
而且 `_drop_unobtainable_views` 会在 D/E 特征真正落地**之前**探测 npz，
把 D/E 视角判为"取不到"并剔除 —— 于是刚提取好的 D/E 特征被**静默忽略**。

修法：metadata.jsonl 默认写**绝对规范路径**（`--path-style abs`），
让 `isabs()` 短路掉 join，键就稳定了。

视角口径（关键）
----------------
根 `CoDefenderC3/ensemble.pkl` 是 K=33 / V=15。本脚本按 `--views` 决定提取到哪一档:

    --views full  (默认) 快速路径 11 视角 + BinaryNinja D/E 4 视角 = **15 视角**
                         → 与基线完全一致，33 个模型全部可训练
    --views fast         只提快速路径 = 11 视角
                         → 23 个模型可训练，10 个 D/E 模型需从基线冻结继承

D/E 视角提取较慢（实测 ~40s bndb + ~16s 特征 / 8 个样本），且会重试。
若 D/E 覆盖率不是 100%，脚本会**在构建 per_month 之前中止**，避免
"_build_month_cache 严格模式把缺 D/E 的样本整个丢掉"这种静默样本损失。

用法
----
  # 1) 预演：看会提取哪些视角、怎么分月（不写文件）
  python CoDefenderC3/train/build_minimal_dataset.py --source "E:\\nkproject\\training_data0915" --dry-run

  # 2) 完整构建（15 视角，与基线对齐）
  python CoDefenderC3/train/build_minimal_dataset.py \\
      --source "E:\\nkproject\\training_data0915" \\
      --out "D:\\nkproject\\pilot500" --per-class 250 --months 4 --views full

  # 3) 只要快速路径（秒级，11 视角）
  python CoDefenderC3/train/build_minimal_dataset.py --source "..." --views fast --drop-per-sample

  # 4) 扩大到全量（5000 恶 / 5000 良）
  python CoDefenderC3/train/build_minimal_dataset.py \\
      --source "E:\\nkproject\\training_data0915" --out "D:\\nkproject\\data_10k" \\
      --per-class 5000 --months 4 --views full

源目录要求
----------
平铺一层，良性/恶性各一个子目录。目录名只认下列集合（**大小写敏感**）：
    良性: benign / clean / negative / ben / goodware / 0
    恶性: malicious / malware / positive / mal / 1
⚠ 注意 `malicous`（少一个 i 的拼写变体）**不在集合内**，会被静默忽略 → 请先改名。

批量提取前建议设置环境变量（详见 data_loader / extract_feature 的门控）：
    CODEFENDER_KEEP_BNDB=1     不自动删除 bndb（避免 safe-delete 钩子杀进程）
    CODEFENDER_N_WORKERS=8     降低 BN 并行度（500+ 规模下 16 并行易崩）
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time

# ── 路径引导（2026-09-16 目录整理：本脚本自项目根移入 CoDefenderC3/train/）──
#   HERE           = <项目>/CoDefenderC3/train
#   CODEFENDER_DIR = <项目>/CoDefenderC3        ← 仓库根，config/models 等在此
#   PROJECT_ROOT   = <项目>                     ← 生产产物 minimal_dataset/ 等在此
HERE = os.path.dirname(os.path.abspath(__file__))
CODEFENDER_DIR = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(CODEFENDER_DIR)
for _p in (CODEFENDER_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

PE_EXTENSIONS = {".exe", ".dll", ".bin", ".sys", ".scr", ".pe", ".drv",
                 ".cpl", ".ocx", ".msi", ""}

BENIGN_DIRS = {"benign", "clean", "negative", "ben", "goodware", "0"}
MALICIOUS_DIRS = {"malicious", "malware", "positive", "mal", "1"}

# 兜底维度表（仅当既没有参考缓存、也读不到基线时使用）
FALLBACK_VIEW_DIMS = {
    "V1_byte": 32768, "V2_byte_stat": 776, "V3_pe_struct": 264, "V4_import": 339,
    "V5_string": 553, "V6_opcode_seq": 4096, "V6_opcode_stat": 549,
    "V6_func_embed": 6144, "V7_graph": 22, "V8_gray": 65536, "V8_color": 196608,
    "V8_markov": 65536, "V8_entropy": 65536, "V9_hash": 512, "V10_metadata": 60,
}

DEFAULT_REFERENCE_CACHE = os.path.join(
    "E:/nkproject/test_samples/CoDefenderC3_data", ".cache", "per_month")
DEFAULT_BASELINE = os.path.join(CODEFENDER_DIR, "ensemble.pkl")


def hdr(t):
    print("\n" + "=" * 74)
    print("  " + t)
    print("=" * 74)


def sha256_of_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def head_is_mz(path):
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"MZ"
    except Exception:
        return False


def scan_labeled_files(root, max_depth=2):
    """递归扫描，按路径目录名判定 label。结果按路径排序 → 与 os.walk 顺序无关。"""
    out = []
    root = os.path.normpath(os.path.abspath(root))
    for dp, dirs, files in os.walk(root):
        depth = dp[len(root):].count(os.sep)
        if depth >= max_depth:
            dirs.clear()
            continue
        dirs[:] = [d for d in dirs if d != ".cache"]
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in PE_EXTENSIONS:
                continue
            fp = os.path.join(dp, fn)
            if not os.path.isfile(fp):
                continue
            parts = [p.lower() for p in
                     os.path.relpath(fp, root).replace("\\", "/").split("/")[:-1]]
            label = -1
            for p in parts:
                if p in MALICIOUS_DIRS:
                    label = 1
                    break
                if p in BENIGN_DIRS:
                    label = 0
                    break
            out.append((fp, label))
    out.sort(key=lambda t: t[0])
    return out


def group_key(path):
    """分组键 = imphash（导入表指纹），失败退化为内容前缀散列。

    用途是"同组样本不跨 train/eval"，不要求组内标签纯。
    """
    try:
        from scaner import staticinfo as si
        with open(path, "rb") as f:
            data = f.read()
        imp = si.compute_imphash(data)
        if imp:
            return "imp:" + imp
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            return "raw:" + hashlib.md5(f.read(1 << 16)).hexdigest()[:16]
    except Exception:
        return "none:" + os.path.basename(path)


def read_reference_dims(ref_cache_dir):
    if not ref_cache_dir or not os.path.isdir(ref_cache_dir):
        return {}
    out = {}
    for fp in sorted(glob.glob(os.path.join(ref_cache_dir, "m*_V*.npy"))):
        m = re.match(r"m\d+_(.+)\.npy", os.path.basename(fp))
        if not m:
            continue
        try:
            out.setdefault(m.group(1), np.load(fp, mmap_mode="r").shape[1])
        except Exception:
            pass
    return out


def read_baseline_dims(baseline_path):
    """从基线 ensemble.pkl 读出视角维度与结构信息。

    返回 (exact_dims, lower_dims, info)：
      exact_dims  — 能精确反推的视角维度（pytorch input_dim / 线性 w / KNN X_train）
      lower_dims  — 只能给下界的（纯 NumPy 树模型）
      info        — (K, V, 视角清单) 或 None
    """
    if not os.path.isfile(baseline_path):
        return {}, {}, None
    from models.ensemble import MultiModelEnsemble
    ens = MultiModelEnsemble.load(baseline_path)
    exact, lower = baseline_instance_dims(ens)
    return exact, lower, (ens.K, ens.V, sorted(ens.view_groups.keys()))


def expected_dims_from_npz(npz_path):
    """**权威口径**：按 config.VIEW_GROUPS 的 binja_features 定义，从 npz 实测形状推导视角列数。

    返回 (dims, partial)：
      dims    — 视角 → 列数
      partial — 视角 → 该视角定义里但 npz 中缺失的 fid 列表（如本批样本无 E01/E02）

    这正是 `_build_month_cache` 实际拼接的逻辑：对每个 fid，**存在才 append**，
    最后 concatenate；所以维度 = 实际存在的 fid 形状之积的和，而不是全量定义。
    """
    import config
    dims, partial = {}, {}
    if not npz_path or not os.path.exists(npz_path):
        return dims, partial
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            files = set(z.files)
            for vn, info in config.VIEW_GROUPS.items():
                if vn == "V1_byte":
                    dims[vn] = 32768          # data_loader.RAW_LEN
                    continue
                fids = list(info.get("binja_features") or [])
                present = [f for f in fids if f in files]
                if not present:
                    continue
                dims[vn] = int(sum(int(np.prod(z[f].shape)) for f in present))
                gone = [f for f in fids if f not in files]
                if gone:
                    partial[vn] = gone
    except Exception:
        pass
    return dims, partial


def _tree_max_feature(model):
    """树模型：返回 max(feature_idx)+1 —— 训练特征维度的**下界**估计（非精确值）。

    覆盖三种实现：
      - PEMiner 的 DecisionTreeNode（对象属性 feature_idx）
      - ahmadi 的 XGBTree / _RFTree（节点是 dict，键名 "feat"）
      - 其它把树根放在 .root / .tree 下的实现
    """
    trees = getattr(model, "trees", None)
    if not trees:
        return None
    mx = [-1]
    feat_keys = ("feature_idx", "feat", "feature", "f")

    def node_feat(node):
        if isinstance(node, dict):
            for k in feat_keys:
                if node.get(k) is not None and not isinstance(node[k], (dict, list)):
                    return int(node[k])
            return None
        for k in feat_keys:
            v = getattr(node, k, None)
            if v is not None and not isinstance(v, (dict, list)):
                return int(v)
        return None

    def walk(node, depth=0):
        if node is None or depth > 64:
            return
        fi = node_feat(node)
        if fi is not None:
            mx[0] = max(mx[0], fi)
        if isinstance(node, dict):
            for k, v in node.items():
                if k in feat_keys:
                    continue
                if isinstance(v, (dict, list, tuple)):
                    for item in (v if isinstance(v, (list, tuple)) else [v]):
                        walk(item, depth + 1)
        else:
            for attr in ("left", "right", "root", "tree", "child", "nodes"):
                walk(getattr(node, attr, None), depth + 1)

    try:
        for t in trees:
            walk(t)
    except Exception:
        return None
    return mx[0] + 1 if mx[0] >= 0 else None


def baseline_instance_dims(ens):
    """从基线**已加载的模型实例**反推 (视角 -> 维度)。

    精确可判（exact）：
      - pytorch     : model_kwargs['input_dim']
      - 线性 numpy  : w.shape[0] / coef_.shape[0]
      - KNN numpy   : X_train.shape[1]
    只能给下界（lower）：纯 NumPy 树模型（PEMinerRF / EmberGBDT / ResourceNet /
    OpcodeStatNet）不保存训练维度，只能用树节点里出现过的最大 feature_idx 估下界。
    """
    exact, lower = {}, {}
    for vname, idxs in ens._view_indices.items():
        for i in idxs:
            mname = ens._model_names[i]
            kw = ens._actual_kwargs.get(mname) or {}
            if "input_dim" in kw and vname not in exact:
                exact[vname] = int(kw["input_dim"])
            m = ens._models.get(mname)
            if m is None or vname in exact:
                continue
            for attr in ("w", "coef_"):
                v = getattr(m, attr, None)
                if v is not None and hasattr(v, "shape") and getattr(v, "ndim", 0) == 1:
                    exact[vname] = int(v.shape[0])
                    break
            if vname in exact:
                continue
            xt = getattr(m, "X_train", None)
            if xt is not None and hasattr(xt, "shape") and len(xt.shape) == 2:
                exact[vname] = int(xt.shape[1])
                continue
            est = _tree_max_feature(m)
            if est is not None:
                lower[vname] = est
    return exact, lower


def has_de_keys(loader, path):
    """单样本 npz 里是否已有 D/E 特征（D01 是关键判据）。"""
    cp = loader._sample_cache_path(path)
    if not os.path.exists(cp):
        return False
    try:
        with np.load(cp, allow_pickle=True) as z:
            return any(k.startswith("D") for k in z.files)
    except Exception:
        return False


def assign_months_grouped(records, n_months, seed=42):
    """按 (label, 组) 分层，**整组**分配到月份。

    与 prepare_rawpe_dataset.stratified_months 的差别：以"组"而非"样本"轮询，
    避免同家族变体被切到不同月份（泄漏）。组内样本跟随组长。
    """
    import random
    rng = random.Random(seed)
    by_label = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)
    for label in sorted(by_label):
        groups = {}
        for r in by_label[label]:
            groups.setdefault(r["group"], []).append(r)
        gkeys = sorted(groups.keys())
        rng.shuffle(gkeys)
        for i, gk in enumerate(gkeys):
            month = (i % n_months) + 1
            for r in groups[gk]:
                r["month"] = month
    records.sort(key=lambda r: (r["month"], r["path"]))
    return records


def main():
    ap = argparse.ArgumentParser(
        description="构建与 ensemble 视角对齐的最小可训练 PE_RAW 数据集")
    ap.add_argument("--source", required=True, help="裸 PE 源目录（含 malicious/ benign/）")
    # 默认输出到项目根（与既有 minimal_dataset/ 保持一致，勿随脚本位置漂移）
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "minimal_dataset"))
    ap.add_argument("--per-class", type=int, default=0, help="每类最多取多少（0=全取）")
    ap.add_argument("--months", type=int, default=4)
    ap.add_argument("--train-months", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--views", choices=["full", "fast"], default="full",
                    help="full=11快速+4个BN视角(15)；fast=只提快速路径(11)")
    ap.add_argument("--allow-partial-de", action="store_true",
                    help="允许 D/E 覆盖率<100%% 仍继续（会丢样本，慎用）")
    ap.add_argument("--path-style", choices=["abs", "native"], default="abs",
                    help="metadata.jsonl 的 path 写法；abs=绝对规范路径(推荐, 避免缓存键漂移)")
    ap.add_argument("--reference-month-cache", default=DEFAULT_REFERENCE_CACHE)
    ap.add_argument("--baseline", default=DEFAULT_BASELINE,
                    help="基线 ensemble.pkl（用于校验视角维度对齐）")
    ap.add_argument("--drop-per-sample", action="store_true",
                    help="构建后删除 per_sample/*.npz（per_month 已足够训练）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    src = os.path.normpath(os.path.abspath(args.source))
    out = os.path.normpath(os.path.abspath(args.out))
    if not os.path.isdir(src):
        print("[X] 源目录不存在: %s" % src)
        return 1

    # ── 0. 视角口径 ──
    hdr("0. 视角口径确认")
    print("  读取基线模型: %s" % args.baseline)
    base_dims, base_lower, base_info = read_baseline_dims(args.baseline)
    if base_info:
        print("  基线: K=%d, V=%d" % (base_info[0], base_info[1]))
        print("  基线可**精确**反推的视角维度 (%d 个，pytorch input_dim / 线性 w / KNN X_train):"
              % len(base_dims))
        for vn in sorted(base_dims):
            print("      %-16s %d" % (vn, base_dims[vn]))
        if base_lower:
            print("  基线只能给**下界**的视角 (纯 NumPy 树模型不存训练维度):")
            for vn in sorted(base_lower):
                print("      %-16s ≥ %d" % (vn, base_lower[vn]))
        base_views = set(base_info[2])
    else:
        print("  [警告] 读不到基线模型，改用参考缓存/兜底表。")
        base_views = set()

    ref_dims = read_reference_dims(args.reference_month_cache)
    if ref_dims:
        print("  参考 per_month 缓存: %s → %d 个视角"
              % (args.reference_month_cache, len(ref_dims)))
    # 优先序：npz 实测(在 5b 计算) > 基线精确值 > 参考缓存 > 兜底表
    expected = dict(FALLBACK_VIEW_DIMS)
    expected.update(ref_dims)
    expected.update(base_dims)
    for vn, lb in base_lower.items():
        expected.setdefault(vn, lb)

    fast_views = sorted(v for v in expected if not v.startswith(("V6_", "V7_")))
    de_views = sorted(set(expected) - set(fast_views))
    print("\n  快速路径视角 (%d): %s" % (len(fast_views), ", ".join(fast_views)))
    print("  BinaryNinja 视角 (%d): %s" % (len(de_views), ", ".join(de_views) or "(无)"))
    print("  本次 --views = %s" % args.views)

    # ── 1. 扫描 ──
    hdr("1. 扫描源目录并筛选样本")
    files = scan_labeled_files(src)
    print("  按扩展名扫描到 %d 个候选" % len(files))
    seen, records, dup, bad_mz, unlabeled = {}, [], [], [], []
    for fp, label in files:
        if not head_is_mz(fp):
            bad_mz.append(fp); continue
        if label < 0:
            unlabeled.append(fp); continue
        h = sha256_of_file(fp)
        if h in seen:
            dup.append(fp); continue
        seen[h] = fp
        records.append({"path": fp, "sha256": h, "label": label})
    print("    非 PE(MZ 失败) : %d" % len(bad_mz))
    print("    标签无法判定   : %d" % len(unlabeled))
    print("    内容重复       : %d" % len(dup))

    mal = [r for r in records if r["label"] == 1]
    ben = [r for r in records if r["label"] == 0]
    print("    可用样本       : %d (恶意 %d / 良性 %d)" % (len(records), len(mal), len(ben)))
    if args.per_class > 0:
        mal, ben = mal[:args.per_class], ben[:args.per_class]
        print("    每类上限 %d → 恶意 %d / 良性 %d" % (args.per_class, len(mal), len(ben)))
    n = min(len(mal), len(ben))
    if n == 0:
        print("[X] 至少一类为空，无法构成 1:1。")
        return 1
    if len(mal) != len(ben):
        print("    类别不均衡 → 各自截断到 %d，得到严格 1:1" % n)
    picked = mal[:n] + ben[:n]

    # ── 2. 分组键 ──
    hdr("2. 计算分组键（imphash）")
    for r in picked:
        r["group"] = group_key(r["path"])
    n_groups = len(set(r["group"] for r in picked))
    print("    %d 个样本 → %d 个分组" % (len(picked), n_groups))
    if n_groups == len(picked):
        print("    （无近重复变体 → 分组退化为逐样本，机制保留、暂不生效）")

    # ── 3. 月份分配 ──
    plan = assign_months_grouped([dict(r) for r in picked], args.months, args.seed)
    hdr("3. 月份分配（分组感知，月份与标签解耦）")
    print("    %-6s %-8s %-8s %-8s" % ("月份", "总数", "良性", "恶意"))
    for m in range(1, args.months + 1):
        rs = [r for r in plan if r["month"] == m]
        b = sum(1 for r in rs if r["label"] == 0)
        print("    %-6d %-8d %-8d %-8d" % (m, len(rs), b, len(rs) - b))
    tr = [r for r in plan if r["month"] <= args.train_months]
    ev = [r for r in plan if r["month"] > args.train_months]
    print("    训练集: 恶 %d / 良 %d     评估集: 恶 %d / 良 %d" % (
        sum(1 for r in tr if r["label"] == 1), sum(1 for r in tr if r["label"] == 0),
        sum(1 for r in ev if r["label"] == 1), sum(1 for r in ev if r["label"] == 0)))
    if args.dry_run:
        print("\n[dry-run] 未写入任何文件。")
        return 0

    # ── 4. 复制样本 ──
    hdr("4. 复制样本")
    if os.path.isdir(out) and os.listdir(out) and not args.force:
        print("[X] 输出目录非空: %s（加 --force 或换 --out）" % out)
        return 1
    for sub in ("malicious", "benign"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    dst_records = []
    for r in plan:
        sub = "malicious" if r["label"] == 1 else "benign"
        dst = os.path.join(out, sub, r["sha256"])
        shutil.copy2(r["path"], dst)
        rel_native = os.path.join(sub, r["sha256"])
        dst_records.append({
            "abs": os.path.normpath(os.path.abspath(dst)),
            "rel": rel_native,
            "label": r["label"], "month": r["month"],
            "group": r["group"], "sha256": r["sha256"],
        })
    print("    已复制 %d 个样本 → %s" % (len(dst_records), out))

    # ── 5. 特征提取 ──
    hdr("5. 特征提取 (--views %s)" % args.views)
    from data_loader import PELoader
    cache_dir = os.path.join(out, ".cache")
    loader = PELoader(data_dir=out)
    per_sample = loader._sample_cache          # 用加载器自己的路径规则，避免键漂移

    # per_month 一律先清空：残留的旧缓存会被 _has_month_cache 误判为有效
    month_cache = loader._month_cache
    if os.path.isdir(month_cache):
        stale = month_cache + ".stale-%s" % time.strftime("%Y%m%d-%H%M%S")
        os.replace(month_cache, stale)
        print("    旧 per_month 已移走 → %s" % os.path.basename(stale))
    os.makedirs(month_cache, exist_ok=True)
    # 跳过日志按 basename 记录，会永久跳过样本
    for stale_log in ("sample_skip.log",):
        p = os.path.join(cache_dir, stale_log)
        if os.path.exists(p):
            os.remove(p)
            print("    已删除陈旧 %s" % stale_log)

    all_paths = [r["abs"] for r in dst_records]
    need_fast = [p for p in all_paths if not loader._has_sample_cache(p)]
    if need_fast:
        loader._extract_fast_parallel(need_fast)
    else:
        print("    快速特征已全部就绪 (%d 个)，跳过" % len(all_paths))

    if args.views == "full":
        # 只在**缺 D/E** 的样本上调 extract_features：BinaryNinja 的清理阶段会
        # 删除临时 worker/残留 bndb，批量删除可能被 safe-delete 钩子拦截并中断进程。
        need_de = [p for p in all_paths if not has_de_keys(loader, p)]
        if need_de:
            print("    D/E 待补 %d/%d 个（BinaryNinja，较慢且会重试）"
                  % (len(need_de), len(all_paths)))
            loader.extract_features(need_de)
        else:
            print("    D/E 特征已全部就绪 (%d 个)，跳过（不触发 BinaryNinja）"
                  % len(all_paths))

    # ── 5b. D/E 覆盖率检查（必须在构建 per_month 之前）──
    def has_de(p):
        return has_de_keys(loader, p)

    ok_fast, ok_de = [], []
    for r in dst_records:
        cp = loader._sample_cache_path(r["abs"])
        if os.path.exists(cp):
            ok_fast.append(r)
            if has_de(r["abs"]):
                ok_de.append(r)
    print("\n    快速特征: %d/%d    D/E 特征: %d/%d"
          % (len(ok_fast), len(dst_records), len(ok_de), len(dst_records)))

    if args.views == "full":
        cov = len(ok_de) / max(len(dst_records), 1)
        if cov == 0.0:
            print("    [自动降级] D/E 覆盖率为 0（BinaryNinja 不可用）→ 本次按 --views fast 继续，"
                  "数据集为 11 视角。")
            args.views = "fast"
        elif cov < 1.0 and not args.allow_partial_de:
            print("\n  [X] D/E 覆盖率 %.1f%% 不是 100%%。若继续，"
                  "_build_month_cache 的严格模式会把缺 D/E 的样本整个丢掉（静默样本损失）。" % (cov * 100))
            print("      选择其一：")
            print("        a) 重跑本命令（bndb/npz 已缓存，只会补缺失的）")
            print("        b) python %s --source \"%s\" --views fast ...   # 退回 11 视角"
                  % (os.path.basename(__file__), src))
            print("        c) 加 --allow-partial-de 强制继续（不推荐）")
            return 1

    keep = ok_de if args.views == "full" else ok_fast
    dropped = [r for r in dst_records if r not in keep]
    if dropped:
        for r in dropped:
            try:
                os.remove(r["abs"])
            except OSError:
                pass
        print("    已从数据集移除 %d 个特征不完整的样本" % len(dropped))
    dst_records = keep
    if not dst_records:
        print("[X] 没有样本提取成功。")
        return 1

    # ── 5c. 权威期望维度：按 config.VIEW_GROUPS 从 npz 实测形状推导 ──
    probe_npz = loader._sample_cache_path(dst_records[0]["abs"])
    npz_dims, npz_partial = expected_dims_from_npz(probe_npz)
    if args.views == "fast":
        npz_dims = {k: v for k, v in npz_dims.items()
                    if not k.startswith(("V6_", "V7_"))}
        npz_partial = {k: v for k, v in npz_partial.items()
                       if not k.startswith(("V6_", "V7_"))}
    if npz_dims:
        expected.update(npz_dims)
        print("\n    权威期望维度（config.VIEW_GROUPS × npz 实测形状，%d 个视角）:"
              % len(npz_dims))
        for vn in sorted(npz_dims):
            extra = ("   ← 本批样本缺 %s，按「存在才拼接」只算已有部分"
                     % ",".join(npz_partial[vn])) if vn in npz_partial else ""
            print("      %-16s %-8d%s" % (vn, npz_dims[vn], extra))
        if npz_partial:
            print("    说明：缺失 fid 不代表数据有问题——"
                  "本批样本的 D/E 特征只覆盖了部分 BinaryNinja 子特征。")

    # ── 6. metadata ──
    hdr("6. 写出 metadata.json / metadata.jsonl")
    n_mal = sum(1 for r in dst_records if r["label"] == 1)
    n_ben = sum(1 for r in dst_records if r["label"] == 0)
    seen_months = sorted(set(r["month"] for r in dst_records))
    train_months = [m for m in seen_months if m <= args.train_months]
    eval_months = [m for m in seen_months if m > args.train_months]
    if not train_months:
        print("[X] 训练月份为空（样本太少，把 --train-months 设小）。")
        return 1

    meta = {
        "feature_type": "PE_RAW",
        "n_samples": len(dst_records),
        "n_benign": n_ben, "n_malicious": n_mal,
        "class_ratio": "1:%.2f" % ((n_mal / n_ben) if n_ben else 0),
        "train_months": train_months, "eval_months": eval_months,
        "n_months": args.months,
        "month_assignment": "group-aware-stratified(seed=%d, key=imphash)" % args.seed,
        "views_mode": args.views,
        "expected_views": {vn: expected[vn] for vn in sorted(expected)},
        "baseline": os.path.basename(args.baseline),
        "baseline_views": sorted(base_views) if base_views else [],
        "source": "build_minimal_dataset.py",
        "note": ("最小可训练数据集：视角与基线 ensemble.pkl 对齐。"
                 "月份为分组感知合成伪时间轴（原始样本无真实采集时间），"
                 "月份之间不可解释为真实时间漂移。"),
    }
    md_json, md_jsonl = os.path.join(out, "metadata.json"), os.path.join(out, "metadata.jsonl")
    for p in (md_json, md_jsonl):
        if os.path.exists(p):
            os.replace(p, p + ".bak")
    with open(md_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(md_jsonl, "w", encoding="utf-8") as f:
        for r in sorted(dst_records, key=lambda x: (x["month"], x["rel"])):
            p = r["abs"] if args.path_style == "abs" else r["rel"]
            f.write(json.dumps({"path": p, "label": r["label"], "month": r["month"],
                                "family": "unknown", "group": r["group"]},
                               ensure_ascii=False) + "\n")
    print("    %s" % md_json)
    print("    %s (%d 行, path-style=%s)" % (md_jsonl, len(dst_records), args.path_style))

    # ── 7. per_month ──
    hdr("7. 构建 per_month 缓存")
    loader.load()
    for m in seen_months:
        loader._build_month_cache(m)

    # ── 8. 校验 ──
    hdr("8. 校验：视角 / 维度 / 配比 / 样本保全")
    produced = {}
    for m in seen_months:
        for fn in os.listdir(month_cache):
            if fn.startswith("m%d_V" % m) and fn.endswith(".npy"):
                vn = fn[len("m%d_" % m):-4]
                try:
                    produced.setdefault(vn, set()).add(
                        np.load(os.path.join(month_cache, fn), mmap_mode="r").shape[1])
                except Exception:
                    pass

    problems = []
    print("    %-16s %-9s %-9s %-9s %-9s %s"
          % ("视角", "数据集", "管线定义", "基线精确", "基线下界", "结果"))
    for vn in sorted(set(list(produced.keys()) + list(npz_dims.keys()))):
        got = sorted(produced.get(vn, []))
        w_npz = npz_dims.get(vn)
        w_base = base_dims.get(vn)
        w_low = base_lower.get(vn)
        d = got[0] if len(got) == 1 else (got or None)
        if not got:
            status = "缺失 ✗"; problems.append("%s 未产出" % vn)
        elif len(got) > 1:
            status = "月间不一致 ✗"; problems.append("%s 月间维度不一致 %s" % (vn, got))
        elif w_npz is not None and d != w_npz:
            status = "vs 管线不符 ✗"
            problems.append("%s 维度 %s ≠ 管线定义 %s" % (vn, d, w_npz))
        elif w_base is not None and d != w_base:
            status = "vs 基线不符 ✗"
            problems.append("%s 维度 %s ≠ 基线精确值 %s" % (vn, d, w_base))
        elif w_low is not None and d < w_low:
            status = "低于基线下界 ✗"
            problems.append("%s 维度 %s < 基线下界 %s（基线该模型吃不下）" % (vn, d, w_low))
        else:
            status = "OK ✓"
        print("    %-16s %-9s %-9s %-9s %-9s %s"
              % (vn, d, w_npz, w_base if w_base is not None else "-",
                 (">=%d" % w_low) if w_low is not None else "-", status))

    if base_views and set(produced) != set(base_views):
        d1, d2 = sorted(set(produced) - set(base_views)), sorted(set(base_views) - set(produced))
        if d1:
            problems.append("数据集多出基线没有的视角: %s" % d1)
        if d2:
            print("\n    ⚠ 基线有但本数据集没有的视角: %s" % d2)
            print("      → 对应模型将在训练时从基线冻结继承（不删除、不置 0）。")

    print("\n    逐月样本量与配比:")
    print("    %-6s %-8s %-8s %-8s %-10s %s" % ("月份", "元数据", "缓存", "良性", "恶意", "角色"))
    for m in seen_months:
        y = np.load(os.path.join(month_cache, "m%d_y.npy" % m))
        exp_n = sum(1 for r in dst_records if r["month"] == m)
        flag = "" if len(y) == exp_n else "  ✗ 丢了 %d 个样本" % (exp_n - len(y))
        if len(y) != exp_n:
            problems.append("月 %d 缓存 %d 条 ≠ 元数据 %d 条" % (m, len(y), exp_n))
        print("    %-6d %-8d %-8d %-8d %-10d %s%s" % (
            m, exp_n, len(y), int((y == 0).sum()), int((y == 1).sum()),
            "训练" if m in train_months else "评估", flag))

    # ── 9. manifest ──
    hdr("9. 写出 manifest.json")
    by_month = {}
    for r in dst_records:
        by_month.setdefault(str(r["month"]), []).append(r["sha256"])
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": src, "dataset_dir": out,
        "views_mode": args.views,
        "n_samples": len(dst_records),
        "by_label": {str(r["label"]): sum(1 for x in dst_records if x["label"] == r["label"])
                     for r in dst_records},
        "by_month": {k: sorted(v) for k, v in by_month.items()},
        "views": {vn: (sorted(s)[0] if len(s) == 1 else sorted(s))
                  for vn, s in sorted(produced.items())},
        "pipeline_view_dims": npz_dims,
        "pipeline_missing_fids": npz_partial,
        "baseline_exact_dims": base_dims,
        "baseline_lower_dims": base_lower,
        "baseline_views": sorted(base_views) if base_views else [],
        "missing_vs_baseline": sorted(set(base_views) - set(produced)) if base_views else [],
        "train_months": train_months, "eval_months": eval_months,
        "path_style": args.path_style,
        "metadata_json_sha256": sha256_of_file(md_json),
        "metadata_jsonl_sha256": sha256_of_file(md_jsonl),
        "problems": problems,
        "build_seconds": round(time.time() - t_start, 1),
    }
    man_path = os.path.join(out, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("    %s" % man_path)

    if args.drop_per_sample:
        cnt = len(os.listdir(per_sample))
        for fn in os.listdir(per_sample):
            try:
                os.remove(os.path.join(per_sample, fn))
            except OSError:
                pass
        print("    已删除 %d 个 per_sample/*.npz" % cnt)

    hdr("完成" if not problems else "完成（有告警）")
    full_match = base_views and set(produced) == set(base_views)
    print("  视角数: %d / 基线 %d  %s"
          % (len(produced), len(base_views) or len(expected),
             "→ 完全对齐 ✓" if full_match else "→ 部分对齐（训练时其余模型冻结继承）"))
    for p in problems:
        print("  ⚠ %s" % p)
    print("  数据集: %s" % out)
    print("  耗时  : %.1fs" % (time.time() - t_start))
    print("\n  下一步:")
    print("    python CoDefenderC3/train_minimal.py --dataset \"%s\"" % out)
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
