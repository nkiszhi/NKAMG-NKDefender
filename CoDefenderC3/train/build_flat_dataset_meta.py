#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为「平铺一层目录」的 PE 数据集生成 metadata.json + metadata.jsonl。

为什么需要这个脚本
------------------
`data_loader.discover_samples()` 走「子目录结构」分支时，只按目录名推断 label
（malicious/ → 1，benign/ → 0），**month 一律置 0、family 一律 unknown**。
随后 `assign_months()` 把 month==0 的样本**按文件 mtime 排序后均匀分月**。

这带来一个静默且致命的后果：若两类样本的 mtime 分布不同（实测常见，
良性常是批量落盘的同一时间戳，恶意是逐步采集的），月份就会与标签强相关。
例如实测某数据集 mtime 排序后标签序列为 "BBBB...BBBBMMMM...MMMM"，
在默认 train_months=[1,2,3] 下训练集会**一个恶意样本都没有**，且不报任何错。

本脚本显式写出 month，绕开 mtime 依赖；并按「伪家族组」整体分配月份，
保证同组样本不跨月（降低同家族变体跨 train/eval 的泄漏）。

伪家族组（无 AVClass 标签时的替代方案）
---------------------------------------
  imphash        ← 导入表指纹，同一工具链/打包器会碰撞。默认分组键。
  rich_header    ← 微软工具链版本指纹（本脚本未启用，数据源里见 fuzzy 库）
  authentihash   ← 去掉签名后的整文件哈希，只能做精确去重，做不了分组
  ssdeep / tlsh  ← 相似度聚类，但 scaner 里的 TLSH 是「只算不比」的实现
                   （Tlsh 类无 diff()），ssdeep 受 FUZZY_DB_MAX_BYTES 限流
用法
----
  # 先看效果，不写任何文件
  python build_flat_dataset_meta.py --data-dir <dir> --dry-run

  # 正式生成（metadata.jsonl 会先备份）
  python build_flat_dataset_meta.py --data-dir <dir> --months 4 --train-months 2

  # 同时把陈旧的 per_month 缓存改名归档（不会直接删除）
  python build_flat_dataset_meta.py --data-dir <dir> --purge-per-month
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict

# ── 路径引导（2026-09-16 目录整理：本脚本自项目根移入 CoDefenderC3/train/）──
_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(_TRAIN_DIR)          # CoDefenderC3（仓库根）
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, os.path.join(CODE_ROOT, "scaner"))
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

# 与 data_loader.PE_EXTENSIONS 保持一致（含 "" —— 项目里样本多为无扩展名的 sha256 命名）
PE_EXTENSIONS = {".exe", ".dll", ".bin", ".sys", ".scr", ".pe", ".drv",
                 ".cpl", ".ocx", ".msi", ""}

LABEL_DIRS = [("malicious", 1), ("benign", 0)]
MAX_DEPTH = 2  # 与 data_loader._scan_pe_files 默认值一致


def scan(data_dir):
    """平铺扫描。max_depth 与 data_loader 一致：子目录 <=1 层才会被扫到。"""
    out = []
    for sub, label in LABEL_DIRS:
        d = os.path.join(data_dir, sub)
        if not os.path.isdir(d):
            print("  [warn] 目录不存在，跳过: %s" % d)
            continue
        n = 0
        for root, dirs, files in os.walk(d):
            if root[len(d):].count(os.sep) >= MAX_DEPTH:
                dirs.clear()
                continue
            for f in files:
                if os.path.splitext(f)[1].lower() not in PE_EXTENSIONS:
                    continue
                fp = os.path.join(root, f)
                out.append({
                    "path": fp,
                    "rel": os.path.relpath(fp, data_dir).replace("\\", "/"),
                    "label": label,
                    "size": os.path.getsize(fp),
                    "mtime": os.path.getmtime(fp),
                })
                n += 1
        print("  %-10s %4d 个" % (sub + "/", n))
    return out


def path_cache_key(p):
    """复刻 PELoader 的缓存键: md5(normpath(abspath(样本路径)))"""
    return hashlib.md5(os.path.normpath(os.path.abspath(p)).encode()).hexdigest()


def compute_group_keys(samples, with_fuzzy=False):
    """给每个样本算伪家族分组键。失败自动降级，不中断。"""
    try:
        from scaner import staticinfo as si
    except Exception as e:
        print("  [warn] scaner.staticinfo 不可用 (%s)，退化为 sha256 分组" % e)
        si = None

    for i, s in enumerate(samples):
        with open(s["path"], "rb") as fh:
            blob = fh.read()
        s["sha256"] = hashlib.sha256(blob).hexdigest()
        s["imphash"] = s["authentihash"] = None
        if si is not None:
            try:
                s["imphash"] = si.compute_imphash(blob)
            except Exception:
                pass
            try:
                s["authentihash"] = si.compute_authentihash(blob)
            except Exception:
                pass
            if with_fuzzy:
                try:
                    s["ssdeep"] = si.compute_ssdeep(blob)
                except Exception:
                    s["ssdeep"] = None
        if (i + 1) % 50 == 0:
            print("    已处理 %d/%d" % (i + 1, len(samples)))
    return samples


def group_key_of(rec, mode):
    if mode == "imphash" and rec.get("imphash"):
        return "imp:" + rec["imphash"]
    if mode == "authentihash" and rec.get("authentihash"):
        return "auth:" + rec["authentihash"]
    if mode == "sha256":
        return "sha:" + rec["sha256"]
    return "sha:" + rec["sha256"]


def assign_grouped_months(records, n_months, seed, mode):
    """按 (label, 组) 为单位把样本分配到月份。

    - 同一个组整体进同一个月（组不拆分 → 同族变体不跨月）
    - 每个月的正负样本数尽量均衡（分层）
    贪心：组按大小降序，每次放入「该 label 装载最少」的月。
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[(r["label"], group_key_of(r, mode))].append(i)

    load = {m: defaultdict(int) for m in range(1, n_months + 1)}
    total = {m: 0 for m in range(1, n_months + 1)}
    items = list(groups.values())
    rng.shuffle(items)
    items.sort(key=len, reverse=True)

    for g in items:
        lab = records[g[0]]["label"]
        m = min(range(1, n_months + 1),
                key=lambda x: (load[x][lab], total[x], x))
        for i in g:
            records[i]["month"] = m
        load[m][lab] += len(g)
        total[m] += len(g)

    for r in records:
        r["family"] = group_key_of(r, mode)
    return records, len(groups)


def contingency(records, n_months):
    """标签 × 月份 列联表 —— 用来肉眼确认月份没有与标签绑定。"""
    tbl = [[0, 0] for _ in range(n_months + 1)]
    for r in records:
        tbl[r["month"]][0 if r["label"] == 1 else 1] += 1
    return tbl


def print_report(records, n_months, train_months, n_groups, mode, key_hit, npz_n):
    print()
    print("=== 标签 x 月份 列联表（恶 / 良）===")
    tbl = contingency(records, n_months)
    print("   月份    恶意   良性    合计")
    for m in range(1, n_months + 1):
        mal, ben = tbl[m]
        flag = "  <- 训练" if m <= train_months else ""
        print("   m%-6d %5d  %5d   %5d%s" % (m, mal, ben, mal + ben, flag))
    tr = [m for m in range(1, train_months + 1)]
    ev = [m for m in range(train_months + 1, n_months + 1)]
    t_mal = sum(tbl[m][0] for m in tr)
    t_ben = sum(tbl[m][1] for m in tr)
    e_mal = sum(tbl[m][0] for m in ev)
    e_ben = sum(tbl[m][1] for m in ev)
    print()
    print("   训练集 m1..m%d: 恶 %d / 良 %d" % (train_months, t_mal, t_ben))
    print("   评估集 m%d..m%d: 恶 %d / 良 %d" % (train_months + 1, n_months, e_mal, e_ben))
    if t_mal == 0 or t_ben == 0:
        print("   [!!] 训练集某一类为 0 —— 该划分不可用，请调整 --months / --train-months")
    if n_groups:
        print()
        print("   伪家族组数: %d（来自 %s；组数越接近样本数说明分组信号越弱）" % (n_groups, mode))
    print("   per_sample 缓存键命中: %d/%d" % (key_hit, len(records)))
    if npz_n and key_hit < len(records):
        print("   -> 未命中样本需重新提取特征（每个 npz 约 18MB）")


def check_cache(data_dir, records, purge):
    """per_month 缓存冲突检查：这是 C5 静默忽略的触发点。"""
    pm = os.path.join(data_dir, ".cache", "per_month")
    pc = os.path.join(data_dir, ".cache", "per_sample")
    npz_n, hit = 0, 0
    if os.path.isdir(pc):
        keys = set(os.path.splitext(f)[0] for f in os.listdir(pc))
        npz_n = len(keys)
        hit = sum(1 for r in records if path_cache_key(r["path"]) in keys)

    if not os.path.isdir(pm):
        print("\n[缓存] per_month 不存在 —— 首次构建，无冲突。")
        return npz_n, hit

    import numpy as np
    cached = {}
    for f in sorted(os.listdir(pm)):
        if f.endswith("_y.npy"):
            m = int(f[1:].split("_")[0])
            cached[m] = len(np.load(os.path.join(pm, f)))
    print()
    print("[缓存] per_month 已存在: 月份 %s，缓存内样本合计 %d"
          % (",".join("m%d" % m for m in sorted(cached)), sum(cached.values())))
    print("[缓存] 本次待写样本 %d" % len(records))
    print("       注意: _has_month_cache() 只检查文件是否存在，**不校验内容/行数**。")
    print("       若沿用旧 per_month，新样本会被静默忽略（训练『跑了但没效果』）。")

    if purge:
        dst = pm + ".stale-" + time.strftime("%Y%m%d-%H%M%S")
        os.rename(pm, dst)
        print("       已归档 -> %s" % dst)
    else:
        print("       建议: 加 --purge-per-month 归档，或手工 mv 该目录后重建。")
    return npz_n, hit


def main():
    ap = argparse.ArgumentParser(description="为平铺 PE 数据集生成 metadata")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--months", type=int, default=4, help="合成月份数（默认 4）")
    ap.add_argument("--train-months", type=int, default=2, help="前 N 个月作训练（默认 2）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-key", default="imphash",
                    choices=["imphash", "authentihash", "sha256"])
    ap.add_argument("--with-fuzzy", action="store_true", help="额外计算 ssdeep（较慢）")
    ap.add_argument("--purge-per-month", action="store_true",
                    help="把陈旧 per_month 改名归档（不删除）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    data_dir = os.path.normpath(os.path.abspath(args.data_dir))
    print("数据集: %s" % data_dir)
    print("\n[1] 扫描平铺目录")
    samples = scan(data_dir)
    if not samples:
        print("未发现样本，退出。"); return 1
    n_mal = sum(1 for s in samples if s["label"] == 1)
    n_ben = len(samples) - n_mal
    print("  合计 %d（恶意 %d / 良性 %d，比例 %.2f:1）"
          % (len(samples), n_mal, n_ben, n_mal / max(1, n_ben)))

    print("\n[2] 计算分组键（%s）" % args.group_key)
    samples = compute_group_keys(samples, with_fuzzy=args.with_fuzzy)
    n_ok = sum(1 for s in samples if s.get("imphash"))
    print("  imphash 成功 %d/%d" % (n_ok, len(samples)))

    print("\n[3] 按 组 分层分配月份")
    records, n_groups = assign_grouped_months(samples, args.months, args.seed, args.group_key)
    print("  月份数 %d，训练月 %s" % (args.months, list(range(1, args.train_months + 1))))

    npz_n, hit = check_cache(data_dir, records, purge=args.purge_per_month and not args.dry_run)
    print_report(records, args.months, args.train_months, n_groups, args.group_key, hit, npz_n)

    meta = {
        "feature_type": "PE_RAW",
        "dataset_name": os.path.basename(data_dir),
        "n_samples": len(records),
        "n_malicious": n_mal,
        "n_benign": n_ben,
        "n_groups": n_groups,
        "family_source": args.group_key,
        "months": args.months,
        "train_months": list(range(1, args.train_months + 1)),
        "eval_months": list(range(args.train_months + 1, args.months + 1)),
        "month_assignment": "group-stratified(seed=%d,key=%s)" % (args.seed, args.group_key),
        "generated_by": "build_flat_dataset_meta.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": ("月份为合成划分，不含真实时间信息；family 为伪家族分组键，"
                  "非 AVClass 家族标签。若需时间外推评估，请另行补充 capture_date 并改用季度划分。"),
    }

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件。将写入:")
        print("   %s/metadata.json  (train_months=%s)" % (data_dir, meta["train_months"]))
        print("   %s/metadata.jsonl (%d 行)" % (data_dir, len(records)))
        print("\n   样例行:")
        for r in records[:3]:
            print("     " + json.dumps({
                "path": r["rel"], "label": r["label"], "month": r["month"],
                "family": r["family"][:20],
                "capture_date": time.strftime("%Y-%m-%d", time.localtime(r["mtime"])),
            }, ensure_ascii=False))
        return 0

    jl = os.path.join(data_dir, "metadata.jsonl")
    if os.path.exists(jl):
        bak = jl + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
        with open(jl, "rb") as src, open(bak, "wb") as dst:
            dst.write(src.read())
        print("\n[写入] 已备份原 metadata.jsonl -> %s" % os.path.basename(bak))

    tmp = jl + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "path": r["rel"],
                "label": r["label"],
                "month": r["month"],
                "family": r["family"],
                "capture_date": time.strftime("%Y-%m-%d", time.localtime(r["mtime"])),
                "sha256": r["sha256"],
                "imphash": r.get("imphash"),
            }, ensure_ascii=False) + "\n")
    os.replace(tmp, jl)

    mp = os.path.join(data_dir, "metadata.json")
    if os.path.exists(mp):
        with open(mp, "rb") as src, open(mp + ".bak-" + time.strftime("%Y%m%d-%H%M%S"), "wb") as dst:
            dst.write(src.read())
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[写入] metadata.jsonl (%d 行)" % len(records))
    print("[写入] metadata.json  (train_months=%s)" % meta["train_months"])
    print("\n完成。下一步: 若 per_month 已归档，可直接 run_all / 增量训练重建缓存。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
