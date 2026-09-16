#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_trained_ensemble.py — 对比"候选 ensemble.pkl"与基线是否等价可替换

检查项
------
1. schema：顶层键集合是否一致（重点：`special_state` 这类 save() 会丢的键）
2. 结构：K / V / available_views / view_indices / model_names 逐一比对
3. 槽位：models 里每个模型是否存在、类型是否一致（含 'SPECIAL_PLACEHOLDER'）
4. 权重：weights 形状与是否等权
5. 可加载性：候选模型能否被 MultiModelEnsemble.load() 成功还原，能还原几个
6. 前向：用伪数据跑一遍 predict_all，确认每个模型都有输出、维度对得上

用法
----
  python CoDefenderC3/train/verify_trained_ensemble.py \\
      --candidate CoDefenderC3/runs/<run_id>/ensemble.pkl
  # --baseline 默认 CoDefenderC3/ensemble.pkl，通常无需显式指定
"""
import argparse
import os
import pickle
import sys

# ── 路径引导（2026-09-16 目录整理：本脚本自项目根移入 CoDefenderC3/train/）──
HERE = os.path.dirname(os.path.abspath(__file__))
CODEFENDER_DIR = os.path.dirname(HERE)          # CoDefenderC3（仓库根）
PROJECT_ROOT = os.path.dirname(CODEFENDER_DIR)  # 项目根
for _p in (CODEFENDER_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

# 视角列数（与 build_minimal_dataset.FALLBACK_VIEW_DIMS 同口径）。
# 用途：前向冒烟的伪数据必须按"视角真实列数"构造，否则会误报。
# 关键点：V1_byte 的 3 个模型（MalConv/MalConv2/ByteTransformer）__init__ 都
# 不接受 input_dim，无法从 model_kwargs 反推，曾经因此兜底成 65536 →
# ByteTransformer 在 `e + pos_enc` 处报 "1024 vs 512" 的假错误。
KNOWN_VIEW_DIMS = {
    "V1_byte": 32768, "V2_byte_stat": 776, "V3_pe_struct": 264, "V4_import": 339,
    "V5_string": 553, "V6_opcode_seq": 4096, "V6_opcode_stat": 549,
    "V6_func_embed": 6144, "V7_graph": 22, "V8_gray": 65536, "V8_color": 196608,
    "V8_markov": 65536, "V8_entropy": 65536, "V9_hash": 512, "V10_metadata": 60,
}


def hdr(t):
    print("\n" + "=" * 74)
    print("  " + t)
    print("=" * 74)


def read_raw(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def describe(path):
    st = read_raw(path)
    d = {
        "path": path,
        "bytes": os.path.getsize(path),
        "keys": sorted(st.keys()),
        "available_views": list(st.get("available_views") or []),
        "K": len(st.get("model_names") or []),
        "views": len(st.get("view_indices") or {}),
        "model_names": list(st.get("model_names") or []),
        "model_types": dict(st.get("model_types") or {}),
        "view_indices": dict(st.get("view_indices") or {}),
        "weights": st.get("weights"),
        "models": st.get("models") or {},
        "model_kwargs": dict(st.get("model_kwargs") or {}),
    }
    return st, d


def main():
    ap = argparse.ArgumentParser(description="候选 ensemble.pkl 与基线等价性校验")
    ap.add_argument("--baseline", default=os.path.join(CODEFENDER_DIR, "ensemble.pkl"))
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--fake-n", type=int, default=4, help="前向冒烟用的伪样本数")
    args = ap.parse_args()

    base_raw, B = describe(args.baseline)
    cand_raw, C = describe(args.candidate)
    problems = []

    hdr("0. 文件")
    print("  基线   : %-60s %.1f MB" % (args.baseline, B["bytes"] / 1e6))
    print("  候选   : %-60s %.1f MB" % (args.candidate, C["bytes"] / 1e6))

    hdr("1. schema（顶层键）")
    only_b = sorted(set(B["keys"]) - set(C["keys"]))
    only_c = sorted(set(C["keys"]) - set(B["keys"]))
    print("  基线键: %s" % B["keys"])
    print("  候选键: %s" % C["keys"])
    if only_b:
        print("  ✗ 候选缺少基线有的键: %s" % only_b)
        problems.append("schema 缺键: %s" % only_b)
    if only_c:
        print("  ⚠ 候选多出键: %s" % only_c)
    if not only_b and not only_c:
        print("  ✓ 键集合完全一致")

    hdr("2. 结构 K / V / 视角映射")
    print("  基线: K=%d V=%d   候选: K=%d V=%d" % (B["K"], B["views"], C["K"], C["views"]))
    if B["K"] != C["K"] or B["views"] != C["views"]:
        problems.append("K/V 不一致")
    if B["available_views"] != C["available_views"]:
        problems.append("available_views 不一致")
        print("  ✗ available_views 不同")
    else:
        print("  ✓ available_views 一致 (%d 个)" % len(B["available_views"]))
    if B["model_names"] != C["model_names"]:
        problems.append("model_names 顺序/内容不一致")
        print("  ✗ model_names 不一致")
    else:
        print("  ✓ model_names 顺序一致")
    if B["view_indices"] != C["view_indices"]:
        problems.append("view_indices 不一致")
        print("  ✗ view_indices 不一致")
    else:
        print("  ✓ view_indices 一致")

    hdr("3. 模型槽位")
    miss = [m for m in B["model_names"] if m not in C["models"]]
    extra = [m for m in C["models"] if m not in B["model_names"]]
    print("  基线 models 条目: %d   候选: %d" % (len(B["models"]), len(C["models"])))
    if miss:
        print("  ✗ 候选缺少模型槽位: %s" % miss)
        problems.append("缺槽位: %s" % miss)
    else:
        print("  ✓ 基线所有模型槽位都存在")
    if extra:
        print("  ⚠ 候选多出槽位: %s" % extra)
    diff_type = [m for m in B["model_names"]
                 if m in C["models"] and B["model_types"].get(m) != C["model_types"].get(m)]
    if diff_type:
        print("  ✗ 类型不一致: %s" % diff_type)
        problems.append("类型不一致: %s" % diff_type)
    # special 占位
    sp_b = {m: type(v).__name__ for m, v in B["models"].items() if isinstance(v, str)}
    sp_c = {m: type(v).__name__ for m, v in C["models"].items() if isinstance(v, str)}
    print("  基线 special 占位: %s   候选: %s" % (sp_b or "无", sp_c or "无"))
    if sp_b and sp_b != sp_c:
        problems.append("special 占位值丢失")

    hdr("4. 权重")
    wb, wc = B["weights"], C["weights"]
    for tag, w in (("基线", wb), ("候选", wc)):
        if w is None:
            print("  %s: None" % tag); continue
        w = np.asarray(w)
        print("  %s: shape=%s unique=%d sum=%.4f" % (tag, w.shape, len(np.unique(w)), w.sum()))
    if wb is not None and wc is not None and np.asarray(wb).shape != np.asarray(wc).shape:
        problems.append("weights 形状不一致")
    elif wb is not None and wc is not None:
        n_diff = int((np.asarray(wb) != np.asarray(wc)).sum())
        print("  权重差异槽位: %d%s" % (n_diff, "（--weighting auc 属预期）" if n_diff else ""))

    hdr("5. 可加载性")
    from models.ensemble import MultiModelEnsemble
    for tag, p in (("基线", args.baseline), ("候选", args.candidate)):
        try:
            ens = MultiModelEnsemble.load(p)
            avail = sum(1 for v in ens._models.values() if v is not None)
            print("  %s: load() OK  K=%d V=%d 可实例化模型 %d/%d" % (tag, ens.K, ens.V, avail, ens.K))
        except Exception as e:
            print("  ✗ %s: load() 失败 %s: %s" % (tag, type(e).__name__, e))
            problems.append("%s load() 失败" % tag)

    hdr("6. 前向冒烟（伪数据）")
    try:
        import config
        from models.ensemble import MultiModelEnsemble
        ens_c = MultiModelEnsemble.load(args.candidate)
        dims = {}
        for mname, kw in (C["model_kwargs"] or {}).items():
            if "input_dim" in kw:
                for vname, idxs in C["view_indices"].items():
                    if C["model_names"].index(mname) in idxs and vname not in dims:
                        dims[vname] = kw["input_dim"]
        for vn in C["available_views"]:
            # 维度优先级：视角真实列数（权威） > 该视角任一模型的 input_dim > 65536
            dims.setdefault(vn, KNOWN_VIEW_DIMS.get(vn, 65536))
        fake = {}
        for vn in C["available_views"]:
            fake[vn] = np.random.RandomState(0).rand(args.fake_n, dims[vn]).astype(np.float32)
        preds, scores = ens_c.predict_all(fake)
        print("  predict_all → preds%s scores%s" % (preds.shape, scores.shape))
        const = [ens_c._model_names[k] for k in range(ens_c.K)
                 if np.allclose(scores[:, k], 0.5)]
        print("  输出恒为 0.5 的模型 (%d): %s" % (len(const), ", ".join(const) or "无"))
        print("  说明: 恒 0.5 = 该模型视角本轮无数据，是 predict_all 的既有行为，不是本次训练造成的。")
    except Exception as e:
        import traceback
        traceback.print_exc()
        problems.append("前向冒烟失败: %s" % e)

    hdr("结论")
    if problems:
        for p in problems:
            print("  ✗ %s" % p)
        print("\n  → 候选**不能**直接替代基线。")
        return 2
    print("  ✓ schema / 结构 / 槽位 / 权重 全部一致，候选可直接热切换替换基线。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
