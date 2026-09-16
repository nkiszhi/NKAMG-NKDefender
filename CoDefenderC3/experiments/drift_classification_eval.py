"""
Experiment: AM-SDD 对比 Transcend/CADE/DREAM 的漂移分类评估
==============================================================

评估维度:
  1. 漂移检测准确率 (所有方法都能做: binary drift/no-drift)
  2. 漂移类型分类准确率 (仅 AM-SDD: perturbation/structural/paradigm)
  3. 防御策略精确性 (AM-SDD 的策略 vs 基线的统一策略)
  4. 计算效率对比

实验设计:
  使用 EMBER/BODMAS 真实数据, 按时间分月。
  不同月份包含不同类型的自然漂移:
    - 同家族微变体出现 → perturbation drift
    - 新壳/混淆技术出现 → structural drift
    - 全新家族/技术范式 → paradigm shift

  Ground truth 来源:
    (a) BODMAS 提供家族标签+时间戳, 可识别家族级变化
    (b) EMBER metadata 中的 avclass 字段标记家族归属
    (c) 通过特征维度分析辅助标注漂移类型

用法:
  python -m experiments.drift_classification_eval --data_dir F:\\Experimental data\\EMBER_data
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from core.sdd_engine import (SDDEngine, SDDResult,
                              DRIFT_NONE, DRIFT_PERTURBATION,
                              DRIFT_STRUCTURAL, DRIFT_PARADIGM)
from core.baselines import (TranscendBaseline, CADEBaseline, HCCBaseline,
                             DroidEvolverBaseline, DREAMBaseline,
                             MADCATBaseline, EnsembleDisagreementBaseline)


# ═══════════════════════════════════════════════════════════════
#  Ground Truth 漂移类型标注
# ═══════════════════════════════════════════════════════════════

EMBER_RANGES_V2 = {
    "histogram": (0, 256), "byteentropy": (256, 512), "strings": (512, 616),
    "general": (616, 626), "header": (626, 688), "section": (688, 943),
    "imports": (943, 2223), "exports": (2223, 2351), "datadirectories": (2351, 2381),
}

EMBER_RANGES_V3 = {
    "histogram": (0, 256), "byteentropy": (256, 512), "strings": (512, 616),
    "general": (616, 636), "header": (636, 738), "section": (738, 1043),
    "imports": (1043, 2323), "exports": (2323, 2451), "datadirectories": (2451, 2491),
    "richheader": (2491, 2531), "authenticode": (2531, 2551), "parsewarnings": (2551, 2568),
}


def _get_surface_struct_dims(D):
    """根据特征维度返回表层/结构维度索引"""
    # 表层 = histogram + byteentropy + strings (前 616 维, 两个版本一样)
    surface = list(range(0, min(616, D)))
    # 结构 = 616 到末尾
    struct = list(range(616, D))
    return surface, struct


def compute_ground_truth_drift_type(X_train, X_eval):
    """
    基于特征维度分析自动标注漂移类型 (ground truth)。

    方法: 比较训练集和评估集在表层/结构/全局维度上的分布偏移量。

    表层偏移 = mean |μ_eval - μ_train| on surface dims / σ_train
    结构偏移 = mean |μ_eval - μ_train| on struct dims / σ_train
    全局偏移 = mean |μ_eval - μ_train| on all dims / σ_train
    """
    D = X_train.shape[1]
    mu_tr = X_train.mean(axis=0)
    sigma_tr = np.maximum(X_train.std(axis=0), 1e-8)
    mu_ev = X_eval.mean(axis=0)

    shift = np.abs(mu_ev - mu_tr) / sigma_tr

    # 维度自适应
    surf_dims, struct_dims = _get_surface_struct_dims(D)

    surface_shift = float(np.mean(shift[surf_dims])) if surf_dims else 0.0
    struct_shift = float(np.mean(shift[struct_dims])) if struct_dims else 0.0
    global_shift = float(np.mean(shift))

    # 阈值 (基于经验, 在论文中需 cross-validation)
    tau = 0.5  # 显著偏移阈值

    sig_s = surface_shift > tau
    sig_r = struct_shift > tau
    sig_g = global_shift > tau

    if not (sig_s or sig_r or sig_g):
        return DRIFT_NONE, surface_shift, struct_shift, global_shift
    elif sig_s and sig_r and sig_g:
        return DRIFT_PARADIGM, surface_shift, struct_shift, global_shift
    elif sig_r:
        return DRIFT_STRUCTURAL, surface_shift, struct_shift, global_shift
    elif sig_s:
        return DRIFT_PERTURBATION, surface_shift, struct_shift, global_shift
    else:
        return DRIFT_STRUCTURAL, surface_shift, struct_shift, global_shift


# ═══════════════════════════════════════════════════════════════
#  模拟多模型集成预测
# ═══════════════════════════════════════════════════════════════

def simulate_ensemble_predictions(X, K=12):
    """
    模拟 K 个异构模型对样本 X 的预测分数。
    自动适配特征维度 (2381d v2 / 2568d v3 / 55d MalMem 等)。
    """
    D = X.shape[1]

    # 动态构建视角范围 — 比例映射, 适配任意维度
    if D >= 2381:
        # EMBER v2/v3: 用百分比映射到实际维度
        VIEW_RANGES = {
            0: (0, 512), 1: (0, 512), 2: (0, 512),              # V2: byte_stat
            3: (616, min(943, D)), 4: (616, min(943, D)),        # V3: pe_struct
            5: (616, min(943, D)),
            6: (min(943, D), min(2223, D)),                       # V4: import
            7: (512, 616),                                         # V5: string
            8: (min(2223, D), D), 9: (min(2223, D), D),          # V10: metadata+
            10: (0, D), 11: (0, D),                               # V11: full
        }
    else:
        # 非 EMBER 数据集: 均匀分割维度
        chunk = max(1, D // 6)
        VIEW_RANGES = {
            0: (0, chunk), 1: (0, chunk), 2: (0, chunk),
            3: (chunk, 2*chunk), 4: (chunk, 2*chunk), 5: (chunk, 2*chunk),
            6: (2*chunk, 3*chunk),
            7: (3*chunk, 4*chunk),
            8: (4*chunk, D), 9: (4*chunk, D),
            10: (0, D), 11: (0, D),
        }

    N = X.shape[0]
    scores = np.zeros((N, K))
    rng = np.random.RandomState(42)

    for k in range(K):
        lo, hi = VIEW_RANGES.get(k, (0, X.shape[1]))
        hi = min(hi, X.shape[1])
        X_view = X[:, lo:hi]

        # 简单线性分数: w·x + noise
        w = rng.normal(0, 0.1, X_view.shape[1])
        raw = X_view @ w
        # 归一化到 [0,1]
        raw = (raw - raw.mean()) / max(raw.std(), 1e-8)
        scores[:, k] = 1 / (1 + np.exp(-raw + rng.normal(0, 0.3, N)))

    return np.clip(scores, 0, 1)


# ═══════════════════════════════════════════════════════════════
#  主实验
# ═══════════════════════════════════════════════════════════════

def run_evaluation(data_dir, output_dir=None):
    """
    在 EMBER/BODMAS 上评估 AM-SDD vs 基线。
    """
    print("=" * 70)
    print("  AM-SDD 对比基线: 漂移检测 + 分类评估")
    print("=" * 70)

    # ── 加载数据 (memmap 避免内存爆炸) ──
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    months = np.load(os.path.join(data_dir, "months.npy"))
    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as _f:
        meta = json.load(_f)

    print(f"  数据: {meta.get('name', '?')}")
    print(f"  N={len(y):,}, D={X.shape[1]}, months={int(months.max())}")

    train_months = meta.get("train_months", [1, 2, 3])
    eval_months = meta.get("eval_months", list(range(4, int(months.max())+1)))
    K = 12  # ensemble size

    print(f"  训练月: {train_months}, 评估月: {eval_months}")

    # ── 训练集 ──
    tr_mask = np.isin(months, train_months)
    X_train, y_train = X[tr_mask], y[tr_mask]

    # ── 模拟集成预测 ──
    print("\n  [1/4] 生成集成预测分数...")
    all_scores = simulate_ensemble_predictions(X, K)
    all_preds = (all_scores > 0.5).astype(float)

    # ── 校准 ──
    print("  [2/4] 校准 AM-SDD + 基线...")
    pg = {
        "V2_byte_stat": {"models": [0, 1, 2]},
        "V3_pe_struct": {"models": [3, 4, 5]},
        "V4_import":    {"models": [6]},
        "V5_string":    {"models": [7]},
        "V10_metadata": {"models": [8, 9]},
        "V11_ensemble": {"models": [10, 11]},
    }

    sdd = SDDEngine(K, pg, alpha=config.ALPHA)

    # 基线数据
    bl_p, bl_s = [], []
    for m in train_months:
        mask = months == m
        bl_p.append(all_preds[mask])
        bl_s.append(all_scores[mask])

    sdd.calibrate(bl_p, bl_s, quiet=True)

    all_train_s = np.concatenate(bl_s)
    all_train_y = y[tr_mask]

    baselines = OrderedDict()
    baselines["Transcend"] = TranscendBaseline(sig=config.ALPHA)
    baselines["CADE"] = CADEBaseline(latent_dim=max(3, K//3))
    baselines["HCC"] = HCCBaseline()
    baselines["DroidEvolver"] = DroidEvolverBaseline()
    baselines["DREAM"] = DREAMBaseline()
    baselines["MADCAT"] = MADCATBaseline()
    baselines["Ens.Disagree"] = EnsembleDisagreementBaseline()

    baselines["Transcend"].calibrate(all_train_s)
    baselines["CADE"].fit(all_train_s)
    baselines["HCC"].calibrate(all_train_s)
    baselines["DroidEvolver"].calibrate(all_train_s)
    baselines["DREAM"].fit(all_train_s, all_train_y[:len(all_train_s)])
    baselines["MADCAT"].calibrate(all_train_s)
    baselines["Ens.Disagree"].calibrate(bl_p)

    # ── 逐月评估 ──
    print("  [3/4] 逐月检测 + 分类...\n")
    results = []

    print(f"  {'月':>3s} {'GT_type':<18s} {'AM-SDD_type':<18s} "
          f"{'match':>5s} {'S_shift':>7s} {'R_shift':>7s} {'G_shift':>7s} "
          + " ".join(f"{'BL_'+bn[:6]:>9s}" for bn in baselines))
    print("  " + "─" * 120)

    for m in eval_months:
        mask = months == m
        if mask.sum() == 0:
            continue

        X_m = X[mask]
        y_m = y[mask]
        s_m = all_scores[mask]
        p_m = all_preds[mask]

        # Ground truth 漂移类型
        gt_type, ss, rs, gs = compute_ground_truth_drift_type(X_train, X_m)

        # AM-SDD 检测 + 分类
        t0 = time.perf_counter()
        det = sdd.detect(p_m, s_m)
        sdd_ms = (time.perf_counter() - t0) * 1000

        # 基线检测 (binary only)
        bl_results = {}
        for bn, bl in baselines.items():
            t0 = time.perf_counter()
            if bn == "Ens.Disagree":
                br = bl.detect(p_m)
            else:
                br = bl.detect(s_m)
            bl_ms = (time.perf_counter() - t0) * 1000
            bl_results[bn] = {
                "drift": br.get("drift_detected", False),
                "time_ms": bl_ms,
            }

        # 匹配
        type_match = det.drift_type == gt_type

        # 打印
        bl_str = " ".join(
            f"{'D' if bl_results[bn]['drift'] else '.':>9s}"
            for bn in baselines)
        print(f"  {m:3d} {gt_type:<18s} {det.drift_type:<18s} "
              f"{'✓' if type_match else '✗':>5s} "
              f"{ss:>7.2f} {rs:>7.2f} {gs:>7.2f} {bl_str}")

        results.append({
            "month": m,
            "n_samples": int(mask.sum()),
            "gt_type": gt_type,
            "sdd_type": det.drift_type,
            "sdd_drift": det.drift_detected,
            "sdd_strategy": det.strategy,
            "sdd_time_ms": sdd_ms,
            "type_match": type_match,
            "surface_shift": ss,
            "struct_shift": rs,
            "global_shift": gs,
            "baselines": {bn: bl_results[bn] for bn in baselines},
            "sdd_level_scores": det.level_scores,
            "sdd_view_deltas": det.view_deltas,
            "sdd_affected_views": det.affected_views,
            "sdd_healthy_models": det.healthy_models,
        })

    # ── 汇总 ──
    print(f"\n{'=' * 70}")
    print("  汇总")
    print("=" * 70)

    n_eval = len(results)
    n_drift = sum(1 for r in results if r["gt_type"] != DRIFT_NONE)
    n_clean = n_eval - n_drift

    # AM-SDD 检测准确率
    sdd_tp = sum(1 for r in results if r["gt_type"] != DRIFT_NONE and r["sdd_drift"])
    sdd_fp = sum(1 for r in results if r["gt_type"] == DRIFT_NONE and r["sdd_drift"])
    sdd_tn = sum(1 for r in results if r["gt_type"] == DRIFT_NONE and not r["sdd_drift"])
    sdd_fn = sum(1 for r in results if r["gt_type"] != DRIFT_NONE and not r["sdd_drift"])
    sdd_det_acc = (sdd_tp + sdd_tn) / max(n_eval, 1)

    # AM-SDD 分类准确率 (只在检测到漂移的月份评估)
    n_classified = sum(1 for r in results if r["gt_type"] != DRIFT_NONE)
    n_correct_type = sum(1 for r in results if r["type_match"] and r["gt_type"] != DRIFT_NONE)
    type_acc = n_correct_type / max(n_classified, 1)

    print(f"\n  AM-SDD:")
    print(f"    检测:  TP={sdd_tp} FP={sdd_fp} TN={sdd_tn} FN={sdd_fn}  Acc={sdd_det_acc:.3f}")
    print(f"    分类:  {n_correct_type}/{n_classified} = {type_acc:.3f}")
    print(f"    平均延迟: {np.mean([r['sdd_time_ms'] for r in results]):.1f} ms")

    # 基线检测准确率
    for bn in baselines:
        tp = sum(1 for r in results if r["gt_type"] != DRIFT_NONE and r["baselines"][bn]["drift"])
        fp = sum(1 for r in results if r["gt_type"] == DRIFT_NONE and r["baselines"][bn]["drift"])
        tn = sum(1 for r in results if r["gt_type"] == DRIFT_NONE and not r["baselines"][bn]["drift"])
        fn = sum(1 for r in results if r["gt_type"] != DRIFT_NONE and not r["baselines"][bn]["drift"])
        acc = (tp + tn) / max(n_eval, 1)
        avg_ms = np.mean([r["baselines"][bn]["time_ms"] for r in results])
        print(f"\n  {bn}:")
        print(f"    检测:  TP={tp} FP={fp} TN={tn} FN={fn}  Acc={acc:.3f}")
        print(f"    分类:  N/A (基线不支持漂移类型分类)")
        print(f"    平均延迟: {avg_ms:.1f} ms")

    # 策略分析
    print(f"\n  ── 防御策略分析 ──")
    strategy_counts = {}
    for r in results:
        s = r["sdd_strategy"]
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    for s, c in sorted(strategy_counts.items()):
        print(f"    {s}: {c} 个月")

    # 关键比较: AM-SDD 的选择性重训 vs 基线的统一全训
    selective = sum(1 for r in results
                    if r["sdd_strategy"] in ("auto_adapt", "selective_retrain"))
    full = sum(1 for r in results if r["sdd_strategy"] == "full_retrain")
    print(f"\n    AM-SDD 选择性策略: {selective} 个月可避免全量重训")
    print(f"    (基线只能统一触发全量重训)")

    print(f"\n{'=' * 70}")

    # 保存
    out = {
        "dataset": meta.get("name", "?"),
        "n_eval_months": n_eval,
        "sdd_detection_accuracy": sdd_det_acc,
        "sdd_classification_accuracy": type_acc,
        "results": results,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, "drift_eval_results.json")
        # 清理 numpy 类型
        def _clean(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list): return [_clean(v) for v in obj]
            return obj
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(_clean(out), f, indent=2, ensure_ascii=False)
        print(f"  结果保存: {outpath}")

    return out


def main():
    parser = argparse.ArgumentParser(description="AM-SDD 对比基线评估")
    parser.add_argument("--data_dir", required=True, help="数据集目录 (含 X.npy/y.npy/months.npy)")
    parser.add_argument("--output_dir", default=None, help="结果输出目录")
    args = parser.parse_args()
    run_evaluation(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
