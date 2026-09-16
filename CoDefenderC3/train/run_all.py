#!/usr/bin/env python3
"""
CoDefenderC3 — 一键运行实验管线
=================================
使用方法:
  1. 打开 config.py，设置 DATA_DIR 为你的样本目录
  2. python run_all.py

自动完成:
  - 识别数据类型 (EMBER预提取 / 原始PE)
  - 训练K个已发表恶意软件检测模型
  - 校准SDD + 7个基线方法
  - 逐月评估 + 漂移检测 + DACP + DSIR
  - 消融实验 (可选)
  - 生成图表 + HTML报告 + JSON/CSV结果
"""
import os, sys, time, json, base64, io, gc, subprocess, shutil, glob, re
import numpy as np
from scipy import stats as sp_stats
from collections import OrderedDict
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.facecolor": "white",
                     "axes.facecolor": "#fafbfc"})

# ── 路径引导（2026-09-16 目录整理：本脚本自 CoDefenderC3/ 根移入 CoDefenderC3/train/）──
# config.py / model_registry.py / feature_extraction/ 被 engine/ 与 models/ 共同依赖，
# 必须留在仓库根，故此处把 train/ 的上一级（仓库根）加回 sys.path。
_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TRAIN_DIR)
for _p in (_REPO_ROOT, _TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from data_loader import create_loader
from models import MultiModelEnsemble
from core.sdd_engine import SDDEngine
from core.baselines import (TranscendBaseline, CADEBaseline, HCCBaseline,
                             DroidEvolverBaseline, DREAMBaseline,
                             MADCATBaseline, EnsembleDisagreementBaseline)
from core.llm_module import analyze_drift
from core.stats import full_comparison, cohens_d, effect_size_label

MN = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
      7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}


# ═══════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════
def hdr(t): print(f"\n{'='*72}\n  {t}\n{'='*72}")

def _auc(y, s):
    try: from sklearn.metrics import roc_auc_score; return roc_auc_score(y, s)
    except Exception: return float("nan")

def _f1(y, p):
    tp = ((p==1)&(y==1)).sum(); fp = ((p==1)&(y==0)).sum(); fn = ((p==0)&(y==1)).sum()
    pr = tp/max(tp+fp,1); rc = tp/max(tp+fn,1); return 2*pr*rc/max(pr+rc,1e-8)

def _acc(y, p): return float(np.mean(y == p))


# ═══════════════════════════════════════
#  Phase 1-4: 自动加载 + 训练 + 校准
# ═══════════════════════════════════════
def _predict_month(loader, ens, month, cache_dir):
    """推断单月 → 写磁盘缓存, 返回 (preds, scores, y) 路径。"""
    pf = os.path.join(cache_dir, f"m{month}_preds.npy")
    sf = os.path.join(cache_dir, f"m{month}_scores.npy")
    yf = os.path.join(cache_dir, f"m{month}_y.npy")
    if os.path.exists(pf) and os.path.exists(sf) and os.path.exists(yf):
        return pf, sf, yf

    result, ym = loader.get_month(month)
    if len(ym) == 0:
        return None, None, None

    # PELoader 返回 month_id, EmberLoader 返回 X 矩阵
    if isinstance(result, (int, np.integer)):
        vm = loader.all_views(result)  # month id
    else:
        vm = loader.all_views(result)  # X matrix or samples list
        del result

    pm, sm = ens.predict_all(vm)
    del vm; gc.collect()
    np.save(pf, pm); np.save(sf, sm); np.save(yf, ym)
    del pm, sm, ym; gc.collect()
    return pf, sf, yf


def _load_cached(path):
    """按需从磁盘读取 (不常驻内存)"""
    return np.load(path)


def setup_pipeline():
    """加载数据、训练模型、校准SDD和基线。
    
    内存原则: 所有中间结果 (预测/分数/标签) 写磁盘, 按需加载。
    峰值内存 ≈ 单月数据 × 单视角 + 模型权重。
    """
    timings = {}

    # ── 1. 加载数据 ──
    hdr("1. 加载数据")
    loader, feature_type = create_loader()

    if feature_type == "PE_RAW" and hasattr(loader, "extract_features"):
        hdr("1b. BinaryNinja 特征提取")
        loader.extract_features()

        # SLFE 公理验证 (Theorem 1.5): 抽样验证 PE 结构隔离性
        try:
            from core.slfe_verify import verify_slfe
            import random
            all_paths = loader.paths if hasattr(loader, 'paths') else []
            sample_pe = random.sample(all_paths, min(5, len(all_paths))) if all_paths else []
            n_pass = 0
            for pe_path in sample_pe:
                try:
                    result = verify_slfe(pe_path=pe_path)
                    if result["struct_code_isolated"]:
                        n_pass += 1
                except Exception:
                    pass
            if sample_pe:
                print(f"  SLFE 验证: {n_pass}/{len(sample_pe)} PE 满足 Structure-Code 隔离")
        except Exception as e:
            print(f"  SLFE 验证跳过: {e}")

    n_train = sum(len(loader.month_idx.get(m, [])) for m in loader.train_months)
    n_eval = sum(len(loader.month_idx.get(m, [])) for m in loader.eval_months)
    print(f"\n  训练: {n_train} 样本 ({len(loader.train_months)} months)")
    print(f"  评估: {n_eval} 样本 ({len(loader.eval_months)} months)")

    # 磁盘缓存目录
    output_dir = os.path.join(config.OUTPUT_DIR, os.path.basename(config.DATA_DIR))
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # ── 2. 训练多模型集成 ──
    hdr("2. 训练多模型集成")
    t = time.time()

    # ── 2a. 确定完整视角集 (交集) ──
    #   从 per_month 文件名快速推断, 不 np.load 任何内容
    month_views = []
    valid_train_months = []

    if hasattr(loader, '_month_cache'):
        # PELoader: 从 per_month 目录的文件名推断视角
        for m in loader.train_months:
            loader._build_month_cache(m)  # 确保缓存已建 (已存在则秒返回)
            yp = loader._month_y_path(m)
            if not os.path.exists(yp):
                continue
            # 从文件名推断视角: m{月}_V{视角}.npy
            pattern = os.path.join(loader._month_cache, f"m{m}_V*.npy")
            vnames = set()
            for fp in glob.glob(pattern):
                fname = os.path.basename(fp)
                # m13_V2_byte_stat.npy → V2_byte_stat
                match = re.match(rf"m{m}_(.+)\.npy", fname)
                if match:
                    vnames.add(match.group(1))
            if vnames:
                month_views.append(vnames)
                valid_train_months.append(m)
    else:
        # EmberLoader: 每月的视角固定, 不需要逐月扫描
        for m in loader.train_months:
            result_m, ym = loader.get_month(m)
            if len(ym) == 0:
                continue
            if isinstance(result_m, (int, np.integer)):
                vm = loader.all_views(result_m)
            else:
                vm = loader.all_views(result_m)
                del result_m
            if not vm:
                continue
            month_views.append(set(vm.keys()))
            valid_train_months.append(m)
            del vm
            break  # EmberLoader 视角每月相同, 看一个月就够了
        # 复制到所有月
        if month_views:
            first_views = month_views[0]
            for m in loader.train_months:
                if m not in valid_train_months:
                    _, ym = loader.get_month(m)
                    if len(ym) > 0:
                        valid_train_months.append(m)
                        month_views.append(first_views)

    if not month_views:
        raise RuntimeError("训练月份全部为空, 无法训练模型")

    # 取所有月份的视角交集
    available_views = month_views[0]
    for mv in month_views[1:]:
        available_views = available_views & mv

    # 打印被排除的视角 (只出现在部分月份)
    all_seen = set()
    for mv in month_views:
        all_seen |= mv
    partial_views = all_seen - available_views
    if partial_views:
        print(f"  部分月份缺失的视角 (已排除): {sorted(partial_views)}")
    print(f"  可用视角 ({len(available_views)}): {sorted(available_views)}")

    model_path = config.MODEL_PATH
    ens = None
    if os.path.exists(model_path):
        print(f"  发现已保存模型: {model_path}")
        ens = MultiModelEnsemble.load(model_path)
        saved_views = set(ens.view_groups.keys())
        if saved_views == available_views:
            print(f"  K={ens.K}, V={ens.V}")
        elif saved_views > available_views:
            missing = saved_views - available_views
            print(f"  ⚠ 已保存模型有 {len(saved_views)} 视角, 当前只有 {len(available_views)} 视角")
            print(f"    缺失: {sorted(missing)} (D/E 提取可能未完成)")
            print(f"    使用已保存模型, 缺失视角将零填充")
        else:
            print(f"  已保存模型视角不匹配, 重新训练...")
            os.remove(model_path)
            ens = None

    if ens is None:
        ens = MultiModelEnsemble(available_views=available_views)
        # 模型重训练 → 清除旧的推断缓存 (K 可能变化)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            print(f"  推断缓存已清除 (模型重训练)")

        # ── 2b. 逐月拼接 (只拼交集内的视角, 行数天然一致) ──
        views_all = {}  # vn → [chunk1, chunk2, ...]
        y_chunks = []
        for m in valid_train_months:
            result, ym = loader.get_month(m)
            if len(ym) == 0:
                continue
            if isinstance(result, (int, np.integer)):
                vm = loader.all_views(result)
            else:
                vm = loader.all_views(result)
                del result
            if not vm:
                continue
            for vn in available_views:
                views_all.setdefault(vn, []).append(vm[vn])
            y_chunks.append(ym)
            del vm

        if not y_chunks:
            raise RuntimeError("训练月份全部为空, 无法训练模型")
        y_tr = np.concatenate(y_chunks); del y_chunks
        N_tr = len(y_tr)

        # ── 2c. 逐视角拼接 (对齐不同月份间的维度差异) ──
        MAX_VIEW_GB = 2.0
        views_tr = {}
        for vn in list(views_all.keys()):
            chunks = views_all.pop(vn)
            max_d = max(c.shape[1] for c in chunks)
            aligned = []
            for c in chunks:
                if c.shape[1] < max_d:
                    pad = np.zeros((c.shape[0], max_d - c.shape[1]), dtype=c.dtype)
                    aligned.append(np.hstack([c, pad]))
                else:
                    aligned.append(c)
            views_tr[vn] = np.vstack(aligned)
            del chunks, aligned
        del views_all; gc.collect()

        # ── 2d. 一致性校验 (理论上不会触发, 但防御性编程) ──
        bad_views = {vn: vd.shape[0] for vn, vd in views_tr.items() if vd.shape[0] != N_tr}
        if bad_views:
            msg = ", ".join(f"{vn}={n}" for vn, n in bad_views.items())
            raise RuntimeError(
                f"视角行数与 y_tr 不一致 (N_tr={N_tr}): {msg}\n"
                f"请删除 per_month 缓存后重跑: rd /s /q .cache\\per_month")
        print(f"  训练数据: {N_tr} 样本, {len(views_tr)} 视角, 行数一致 ✓")

        # ── 2e. 子采样 (所有视角 + y 使用相同索引) ──
        min_max_n = N_tr
        for vn, vdata in views_tr.items():
            view_gb = vdata.nbytes / 1024**3
            if view_gb > MAX_VIEW_GB:
                max_n = int(MAX_VIEW_GB * 1024**3 / (vdata.shape[1] * vdata.itemsize))
                min_max_n = min(min_max_n, max_n)
                print(f"    {vn}: {view_gb:.1f}GB > {MAX_VIEW_GB}GB → 需子采样")

        if min_max_n < N_tr:
            rng = np.random.RandomState(config.SEED)
            subsample_idx = rng.choice(N_tr, min_max_n, replace=False)
            subsample_idx.sort()
            print(f"    统一子采样: {N_tr} → {min_max_n}")
            y_tr = y_tr[subsample_idx]
            for vn in views_tr:
                views_tr[vn] = views_tr[vn][subsample_idx]

        ens.fit(views_tr, y_tr)
        del views_tr, y_tr; gc.collect()

        if config.SAVE_MODEL:
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            ens.save(model_path)
            print(f"  模型已保存: {model_path}")
    timings["train"] = time.time() - t
    K = ens.K
    print(f"\n{ens.model_info()}\n")

    # ── 3. 逐月推断 → 磁盘缓存 (训练月) ──
    hdr("3. 校准SDD引擎")
    for m in loader.train_months:
        pf, sf, yf = _predict_month(loader, ens, m, cache_dir)
        if yf:
            ym = _load_cached(yf); sm = _load_cached(sf)
            ep, es = ens.ensemble_predict(sm)
            print(f"  {MN.get(m,m)}: N={len(ym)}, Acc={_acc(ym,ep):.4f}, AUC={_auc(ym,es):.4f}")
            del ym, sm, ep, es

    # 从磁盘加载校准数据 (使用完全相同的月份集合)
    sdd = SDDEngine(K, ens.perspective_groups(), alpha=config.ALPHA,
                    ws=config.WS, wr=config.WR)
    bl_p, bl_s, cal_months_loaded = [], [], []
    for m in loader.train_months:
        pp = os.path.join(cache_dir, f"m{m}_preds.npy")
        sp = os.path.join(cache_dir, f"m{m}_scores.npy")
        yp = os.path.join(cache_dir, f"m{m}_y.npy")
        if os.path.exists(pp) and os.path.exists(sp) and os.path.exists(yp):
            bl_p.append(np.load(pp))
            bl_s.append(np.load(sp))
            cal_months_loaded.append(m)
    if len(bl_p) < 2:
        raise RuntimeError(f"校准需要至少 2 个月的数据, 仅找到 {len(bl_p)} 个月")
    sdd.calibrate(bl_p, bl_s)

    # 样本级基线: 逐月读取拼接
    all_s = np.vstack(bl_s)
    sdd.calibrate_sample_baseline(all_s)
    print(f"  样本级漂移基线已校准 (N_cal={len(all_s)})")

    # ── 4. 校准基线 ──
    hdr("4. 校准7个基线方法")
    # 使用与 bl_s 完全相同的月份加载 y, 保证长度一致
    all_y = np.concatenate([np.load(os.path.join(cache_dir, f"m{m}_y.npy"))
                            for m in cal_months_loaded])

    baselines = OrderedDict()
    baselines["Transcend"]    = TranscendBaseline(sig=config.ALPHA)
    baselines["CADE"]         = CADEBaseline(latent_frac=0.5)
    baselines["HCC"]          = HCCBaseline()
    baselines["DroidEvolver"] = DroidEvolverBaseline()
    baselines["DREAM"]        = DREAMBaseline()
    baselines["MADCAT"]       = MADCATBaseline()
    baselines["Ens.Disagree"] = EnsembleDisagreementBaseline()

    baselines["Transcend"].calibrate(all_s)
    baselines["CADE"].fit(all_s)
    baselines["HCC"].calibrate(all_s)
    baselines["DroidEvolver"].calibrate(all_s)
    baselines["DREAM"].fit(all_s, all_y)
    baselines["MADCAT"].calibrate(all_s)
    baselines["Ens.Disagree"].calibrate(bl_p)
    print(f"  7个基线方法全部校准完成")
    del all_s, all_y, bl_p, bl_s; gc.collect()

    # DACP 校准集 = 最后一个训练月 (从磁盘读取)
    last_m = loader.train_months[-1]
    cal_s = _load_cached(os.path.join(cache_dir, f"m{last_m}_scores.npy"))
    cal_y = _load_cached(os.path.join(cache_dir, f"m{last_m}_y.npy"))

    return loader, ens, sdd, baselines, cal_s, cal_y, timings, cache_dir



# ═══════════════════════════════════════
#  Phase 5: 逐月评估
# ═══════════════════════════════════════
def run_monthly_eval(loader, ens, sdd, baselines, cal_s, cal_y, cache_dir=None):
    hdr("5. 逐月时间流评估")
    K = ens.K
    bl_keys = list(baselines.keys())

    header = f"  {'Mon':>5s} {'N':>6s} {'AUC':>6s} {'F1':>6s} {'SDD':>6s} {'r':>2s} {'Gap':>5s} {'Strategy':>13s}"
    for bn in bl_keys: header += f" {bn[:3]:>3s}"
    print(header)
    print("  " + "-" * len(header))

    results, per_model_accs = [], []

    for month in loader.eval_months:
        # 推断 → 磁盘缓存 (或读取已有缓存)
        if cache_dir:
            pf, sf, yf = _predict_month(loader, ens, month, cache_dir)
            if not yf: continue
            pm = _load_cached(pf); sm = _load_cached(sf); ym = _load_cached(yf)
        else:
            result, ym = loader.get_month(month)
            if len(ym) == 0: continue
            vm = loader.all_views(result)
            if not isinstance(result, (int, np.integer)): del result
            pm, sm = ens.predict_all(vm)
            del vm; gc.collect()
        ep, es = ens.ensemble_predict(sm)
        a, f_ = _auc(ym, es), _f1(ym, ep)

        pm_acc = ens.per_model_eval(None, ym, cached_preds=pm)
        per_model_accs.append({"month": MN.get(month, str(month)), **pm_acc})

        det = sdd.detect(pm, sm)

        # ── CoDefenderC3: 样本级漂移证据 ──
        sample_ev = sdd.score_samples(sm)
        n_drift_candidates = int(sample_ev["is_drift_candidate"].sum())
        drift_type_counts = {}
        for dt in sample_ev["sample_drift_type"]:
            drift_type_counts[dt] = drift_type_counts.get(dt, 0) + 1

        dacp = sdd.dacp_predict(sm, cal_s, cal_y, det.healthy_models, config.DACP_ALPHA)
        stdcp = sdd.dacp_predict(sm, cal_s, cal_y, list(range(K)), config.DACP_ALPHA)
        dc = sdd.compute_coverage(dacp["prediction_sets"], ym) * 100
        sc = sdd.compute_coverage(stdcp["prediction_sets"], ym) * 100

        dw = sdd.dsir_weights(det)
        ep2, es2 = ens.ensemble_predict(sm, dw)
        a2, f2 = _auc(ym, es2), _f1(ym, ep2)

        pl, pli, plr = sdd.generate_pseudo_labels(sm, det.healthy_models)
        pla = float(np.mean(pl == ym[pli])) * 100 if len(pli) else 0

        llm = analyze_drift(det) if det.drift_detected else None

        bld = {}
        for bn in bl_keys:
            if bn == "Ens.Disagree":
                bld[bn] = bool(baselines[bn].detect(pm).get("drift_detected", False))
            else:
                bld[bn] = bool(baselines[bn].detect(sm).get("drift_detected", False))

        if not det.drift_detected:
            sdd.update_baseline(pm, sm, window=config.BASELINE_MONTHS + 3)

        row = {"month": month, "name": MN.get(month, str(month)), "n": len(ym),
               "auc": a, "f1": f_, "auc_dsir": a2, "f1_dsir": f2,
               "drift": det.drift_detected, "r": det.num_mechanisms,
               "drift_type": det.drift_type,
               "gap": det.spectral_gap, "strat": det.strategy,
               "views": det.affected_views, "mech_views": det.mechanism_views,
               "max_ev": float(det.eigenvalues[0]),
               "view_loadings": det.view_loadings,
               "confidence": det.confidence,
               "label_budget": det.label_budget,
               "mech_strengths": det.mechanism_strengths,
               "level_scores": det.level_scores,
               "drift_explanation": det.drift_explanation,
               "n_drift_candidates": n_drift_candidates,
               "drift_type_counts": drift_type_counts,
               "dacp": dc, "stdcp": sc, "gain": dc - sc,
               "setsize": dacp["avg_set_size"],
               "pl_rate": plr * 100, "pl_acc": pla,
               "sdd_timing": det.timing_ms,
               "llm_report": llm["report"] if llm else None,
               "llm_attck": llm["attck_mapping"] if llm else [],
               "llm_severity": llm["severity"] if llm else None}
        row.update({f"bl_{k}": v for k, v in bld.items()})
        results.append(row)

        sd = "DRIFT" if det.drift_detected else "clean"
        dt = det.drift_type[:8] if det.drift_detected else ""
        line = (f"  {MN.get(month,str(month)):>5s} {len(ym):6d} {a:.4f} {f_:.4f} "
                f"{sd:>6s} {dt:>8s} {det.num_mechanisms:>2d} {det.spectral_gap:.2f} "
                f"{det.strategy:>16s} ({n_drift_candidates}/{len(ym)} samples)")
        for bn in bl_keys:
            line += f" {'Y' if bld.get(bn) else '.':>3s}"
        print(line)
        del pm, sm, ym, ep, es, sample_ev, dacp, stdcp

    return results, per_model_accs


# ═══════════════════════════════════════
#  Phase 6: 消融实验
# ═══════════════════════════════════════
def run_ablation(loader, ens, sdd, cache_dir, figdir):
    hdr("6. 消融实验 (§7.13)")
    from experiments.ablation import ablation_model_count, ablation_weights, ablation_batch_size

    # 找一个漂移月份做消融 — 优先选漂移最强的月份 (高r, 高gap, 足够样本)
    test_p = test_s = test_y = None
    best_month, best_gap = None, -1
    for m in loader.eval_months:
        pf = os.path.join(cache_dir, f"m{m}_preds.npy") if isinstance(cache_dir, str) else ""
        sf = os.path.join(cache_dir, f"m{m}_scores.npy") if isinstance(cache_dir, str) else ""
        yf = os.path.join(cache_dir, f"m{m}_y.npy") if isinstance(cache_dir, str) else ""
        if os.path.exists(pf) and os.path.exists(sf):
            pm, sm, ym = np.load(pf), np.load(sf), np.load(yf)
        else:
            result, ym = loader.get_month(m)
            vm = loader.all_views(result)
            pm, sm = ens.predict_all(vm)
            del vm
        if len(ym) < 50:
            del pm, sm, ym
            continue
        det = sdd.detect(pm, sm)
        if det.drift_detected and det.spectral_gap > best_gap:
            best_month = m
            best_gap = det.spectral_gap
            test_p, test_s, test_y = pm, sm, ym
        else:
            del pm, sm, ym

    if test_p is not None:
        print(f"  消融基准月: {MN.get(best_month, best_month)} (gap={best_gap:.2f})")

    if test_p is None:
        m = loader.eval_months[-1] if loader.eval_months else loader.train_months[-1]
        pf = os.path.join(cache_dir, f"m{m}_preds.npy") if isinstance(cache_dir, str) else ""
        sf = os.path.join(cache_dir, f"m{m}_scores.npy") if isinstance(cache_dir, str) else ""
        yf = os.path.join(cache_dir, f"m{m}_y.npy") if isinstance(cache_dir, str) else ""
        if os.path.exists(pf):
            pm, sm, ym = np.load(pf), np.load(sf), np.load(yf)
        else:
            result, ym = loader.get_month(m)
            vm = loader.all_views(result)
            pm, sm = ens.predict_all(vm)
            del vm
        test_p, test_s, test_y = pm, sm, ym
        print(f"  无漂移月，使用 {MN.get(m, m)}")

    pg = ens.perspective_groups()

    # 从磁盘缓存加载基线数据
    if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
        bl_p = [np.load(os.path.join(cache_dir, f"m{m}_preds.npy")) for m in loader.train_months
                if os.path.exists(os.path.join(cache_dir, f"m{m}_preds.npy"))]
        bl_s = [np.load(os.path.join(cache_dir, f"m{m}_scores.npy")) for m in loader.train_months
                if os.path.exists(os.path.join(cache_dir, f"m{m}_scores.npy"))]
    else:
        bl_p, bl_s = [], []

    # §7.13.1 K消融
    print("\n  §7.13.1 模型数量消融:")
    k_results = ablation_model_count(bl_p, bl_s, test_p, test_s, test_y, pg)
    for r in k_results:
        det_str = "Y" if r.get("detected") else "."
        print(f"    {r['name']:>12s}: K={r['K']:>2d} V={r.get('V',0):>2d} "
              f"{det_str} r={r.get('r','?')} F1={r.get('f1',0):.3f} AUC={r.get('auc',0.5):.3f}")

    # §7.13.2 权重消融
    print("\n  §7.13.2 分歧权重消融 (ws:wr):")
    w_results = ablation_weights(bl_p, bl_s, test_p, test_s, test_y, pg)
    for r in w_results:
        det_str = "Y" if r.get("detected") else "."
        gap = r.get("gap", 0)
        conf = r.get("confidence", 0)
        rr = r.get("r", 0)
        gain = r.get("dsir_auc_gain", 0)
        print(f"    {r['label']:>25s}: {det_str} r={rr} gap={gap:.2f} conf={conf:.2f} DSIR={gain:+.4f}")

    # §7.13.3 批大小
    print("\n  §7.13.3 批大小敏感性:")
    b_results = ablation_batch_size(bl_p, bl_s, test_p, test_s, test_y, pg)
    for r in b_results:
        print(f"    N={r['N']:>5d}: detect={r['detection_rate']:.0%} "
              f"FPR={r.get('fpr',0):.0%} latency={r['avg_ms']:.1f}ms")

    # 消融图
    if k_results:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ks = [r["K"] for r in k_results if "f1" in r]
        f1s = [r.get("f1", 0) for r in k_results if "f1" in r]
        dets = [1 if r.get("detected") else 0 for r in k_results if "f1" in r]
        a1.plot(ks, f1s, "o-", c="#2563eb", lw=2, ms=8, label="F1")
        a1.plot(ks, dets, "s--", c="#dc2626", lw=2, ms=8, label="Detected")
        a1.set(xlabel="K (models)", ylabel="Score", title="K Ablation")
        a1.legend()
        if b_results:
            ns = [r["N"] for r in b_results]
            ms = [r["avg_ms"] for r in b_results]
            dr = [r["detection_rate"] for r in b_results]
            a2.plot(ns, ms, "o-", c="#2563eb", lw=2, ms=8, label="Latency (ms)")
            ax2 = a2.twinx()
            ax2.plot(ns, dr, "s--", c="#dc2626", lw=2, ms=8, label="Detect Rate")
            a2.set(xlabel="Batch Size N", ylabel="Latency (ms)", title="Batch Size")
            ax2.set_ylabel("Detection Rate")
            a2.legend(loc="upper left"); ax2.legend(loc="lower right")
        fig.tight_layout(); fig.savefig(f"{figdir}/fig8_ablation.png"); plt.close()

    return {"K": k_results, "weights": w_results, "batch": b_results}


# ═══════════════════════════════════════
#  Phase 7: 生成图表
# ═══════════════════════════════════════
def generate_figures(results, per_model_accs, figdir, ens):
    hdr("7. 生成图表")
    K = ens.K
    x = np.arange(len(results)); mns = [r["name"] for r in results]

    # Fig 1: 性能曲线
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, [r["auc"] for r in results], "o-", c="#2563eb", lw=2, ms=7, label="AUC")
    ax.plot(x, [r["f1"] for r in results], "s--", c="#dc2626", lw=2, ms=7, label="F1")
    ax.plot(x, [r["auc_dsir"] for r in results], "^:", c="#059669", lw=1.5, ms=6, label="AUC (DSIR)")
    for i, r in enumerate(results):
        if r["drift"]: ax.axvspan(i-.4, i+.4, alpha=.08, color="red")
    ax.set_xticks(x); ax.set_xticklabels(mns)
    ax.set(ylabel="Score", title="Monthly Performance", ylim=(.5, 1.02))
    ax.legend(loc="lower left"); fig.tight_layout()
    fig.savefig(f"{figdir}/fig1_performance.png"); plt.close()

    # Fig 2: SDD检测
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    cols = ["#dc2626" if r["drift"] else "#94a3b8" for r in results]
    a1.bar(x, [r["r"] for r in results], color=cols, alpha=.8)
    a1.set_ylabel("# Mechanisms"); a1.set_title("SDD Detection & Spectral Gap")
    a2.plot(x, [r["gap"] for r in results], "s-", c="#059669", lw=2, ms=7)
    a2.axhline(config.DELTA_THR, c="orange", ls="--", alpha=.7, label=f"thr={config.DELTA_THR}")
    a2.fill_between(x, 0, [r["gap"] for r in results], alpha=.1, color="#059669")
    a2.set_xticks(x); a2.set_xticklabels(mns)
    a2.set(ylabel="Spectral Gap", xlabel="Month", ylim=(-.05, 1.05))
    a2.legend(); fig.tight_layout(); fig.savefig(f"{figdir}/fig2_sdd.png"); plt.close()

    # Fig 3: DACP
    fig, ax = plt.subplots(figsize=(10, 5)); w = .35
    ax.bar(x-w/2, [r["stdcp"] for r in results], w, label="Standard CP", color="#94a3b8")
    ax.bar(x+w/2, [r["dacp"] for r in results], w, label="DACP (SDD)", color="#2563eb")
    ax.axhline(90, c="#dc2626", ls="--", alpha=.6, label="90% target")
    ax.set_xticks(x); ax.set_xticklabels(mns, rotation=30, ha="right")
    ax.set(ylabel="Coverage (%)", title="DACP vs Standard Conformal", ylim=(50, 102))
    ax.legend(); fig.tight_layout(); fig.savefig(f"{figdir}/fig3_dacp.png"); plt.close()

    # Fig 4: 基线对比
    if not results:
        return 0
    bl_keys = [k for k in results[0] if k.startswith("bl_")]
    all_bns = ["SDD"] + [k[3:] for k in bl_keys]
    cs = ["#2563eb","#f59e0b","#8b5cf6","#06b6d4","#6b7280","#ec4899","#84cc16","#f97316"]
    fig, ax = plt.subplots(figsize=(14, 5))
    off = np.linspace(-.35, .35, len(all_bns))
    for bi, bn in enumerate(all_bns):
        key = "drift" if bn == "SDD" else f"bl_{bn}"
        vals = [1 if r.get(key) else 0 for r in results]
        ax.bar(x+off[bi], vals, width=max(.06, .7/len(all_bns)),
               label=bn, color=cs[bi % len(cs)], alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels(mns)
    ax.set(ylabel="Detected", title="SDD vs 7 Baselines")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No", "Yes"])
    ax.legend(ncol=4, fontsize=8, loc="upper left"); fig.tight_layout()
    fig.savefig(f"{figdir}/fig4_baselines.png"); plt.close()

    # Fig 5: 视角归因
    dm = [r for r in results if r["drift"]]
    vns = list(ens.vg.keys())
    if dm:
        fig, ax = plt.subplots(figsize=(10, max(3, len(dm)*0.6+1)))
        data = np.zeros((len(dm), len(vns)))
        for i, r in enumerate(dm):
            for j, v in enumerate(vns):
                data[i, j] = r.get("view_loadings", {}).get(v, 1.0 if v in r["views"] else 0.0)
        for i in range(len(dm)):
            mx = data[i].max()
            if mx > 0: data[i] /= mx
        ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(vns)))
        ax.set_xticklabels([ens.vg[v]["label"] for v in vns], rotation=45, ha="right")
        ax.set_yticks(range(len(dm))); ax.set_yticklabels([r["name"] for r in dm])
        ax.set_title("View-Level Drift Attribution"); fig.tight_layout()
        fig.savefig(f"{figdir}/fig5_views.png"); plt.close()

    # Fig 6: Gap vs PL
    gv = [r["gap"] for r in results if r["pl_acc"] > 0]
    pv = [r["pl_acc"] for r in results if r["pl_acc"] > 0]
    if len(gv) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(gv, pv, s=120, c="#2563eb", zorder=5, ec="white", lw=1.5)
        for r in [r for r in results if r["pl_acc"] > 0]:
            ax.annotate(r["name"], (r["gap"], r["pl_acc"]), fontsize=9,
                        xytext=(5, 5), textcoords="offset points")
        if len(gv) >= 3:
            rho, pval = sp_stats.pearsonr(gv, pv)
            z = np.polyfit(gv, pv, 1); xr = np.linspace(min(gv)-.05, max(gv)+.05, 100)
            ax.plot(xr, np.polyval(z, xr), "--", c="#dc2626", alpha=.7)
            ax.set_title(f"Spectral Gap vs PL Quality (r={rho:.3f})")
        ax.set(xlabel="Spectral Gap", ylabel="Pseudo-Label Accuracy (%)")
        fig.tight_layout(); fig.savefig(f"{figdir}/fig6_gap_pl.png"); plt.close()

    # Fig 7: 逐模型精度热图
    if per_model_accs:
        model_names = ens._model_names
        fig, ax = plt.subplots(figsize=(14, 5))
        mat = np.array([[pm_acc.get(name, 0.5) for name in model_names]
                         for pm_acc in per_model_accs])
        im = ax.imshow(mat.T, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(per_model_accs)))
        ax.set_xticklabels([p["month"] for p in per_model_accs], rotation=45, ha="right")
        ylabels = [n[:15] for n in model_names]
        ax.set_yticks(range(K)); ax.set_yticklabels(ylabels, fontsize=8)
        plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
        ax.set_title("Per-Model Monthly Accuracy"); fig.tight_layout()
        fig.savefig(f"{figdir}/fig7_permodel.png"); plt.close()

    nfig = len([f for f in os.listdir(figdir) if f.endswith(".png")])
    print(f"  {nfig} 张图表 -> {figdir}/")
    return nfig


# ═══════════════════════════════════════
#  Phase 8: 统计分析
# ═══════════════════════════════════════
def statistical_analysis(results, baselines):
    hdr("8. 统计分析 (McNemar + Holm-Bonferroni)")
    dm = [r for r in results if r["drift"]]
    cl = [r for r in results if not r["drift"]]

    # McNemar 需要在所有月份上做配对比较, 不能只看漂移月
    sdd_det = [r["drift"] for r in results]
    bl_dict = {bn: [r.get(f"bl_{bn}", False) for r in results] for bn in baselines}
    comp = full_comparison(sdd_det, bl_dict)

    # 注意: 无 ground truth 漂移标签, TPR/FPR 是各方法自身的检出率
    # dm/cl 按 SDD 检出结果划分, 基线的 "TPR" 实际是"与 SDD 一致率"
    sdd_det_rate = sum(r["drift"] for r in results) / max(len(results), 1)

    print(f"\n  {'Method':>14s} {'DetRate':>7s} {'Agree':>6s} {'p':>8s} {'Sig':>5s} {'h':>6s} {'Effect':>8s}")
    print(f"  {'CoDefenderC3':>14s} {sdd_det_rate:7.2%} {'--':>6s} {'--':>8s} {'--':>5s} {'--':>6s} {'--':>8s}")
    for bn, c in comp.items():
        bl_det_rate = sum(r.get(f"bl_{bn}", False) for r in results) / max(len(results), 1)
        agree = sum(1 for r in results if r["drift"] == r.get(f"bl_{bn}", False)) / max(len(results), 1)
        print(f"  {bn:>14s} {bl_det_rate:7.2%} {agree:6.2%} {c['raw_p']:8.4f} "
              f"{'Y' if c['significant_corrected'] else '.':>5s} "
              f"{c['cohens_h']:6.3f} {c['effect_size']:>8s}")

    if dm:
        # Filter out months with nan AUC (too few samples)
        dm_valid = [r for r in dm
                    if not np.isnan(r.get("auc", float("nan")))
                    and not np.isnan(r.get("auc_dsir", float("nan")))]
        if dm_valid:
            dsir_gains = [r["auc_dsir"] - r["auc"] for r in dm_valid]
            d = cohens_d([r["auc_dsir"] for r in dm_valid], [r["auc"] for r in dm_valid])
            print(f"\n  DSIR AUC改善 (漂移月, N={len(dm_valid)}): mean={np.mean(dsir_gains):+.4f}")
            print(f"  Cohen's d (DSIR vs Base): {d:.3f} ({effect_size_label(d)})")
        else:
            print(f"\n  DSIR: 无有效漂移月 (所有月份 AUC=nan)")
    else:
        print(f"\n  DSIR: 无漂移月")

    return comp


# ═══════════════════════════════════════
#  Phase 9: HTML报告
# ═══════════════════════════════════════
def generate_html_report(results, figdir, output_dir, timings, ablation_results, stat_comp, ens):
    K = ens.K

    def embed(fname):
        fp = os.path.join(figdir, fname)
        if not os.path.exists(fp): return ""
        with open(fp, "rb") as f:
            return f'<img src="data:image/png;base64,{base64.b64encode(f.read()).decode()}" style="max-width:100%">'

    dm = [r for r in results if r["drift"]]
    cl = [r for r in results if not r["drift"]]
    avg_gain = np.mean([r["gain"] for r in results]) if results else 0
    sdd_det_rate = sum(r["drift"] for r in results)/max(len(results),1)*100
    avg_timing = np.mean([r.get("sdd_timing", {}).get("total_ms", 0) for r in results])

    month_rows = ""
    for r in results:
        badge = '<span style="color:#dc2626;font-weight:bold">DRIFT</span>' if r["drift"] else '<span style="color:#22c55e">clean</span>'
        month_rows += (f'<tr><td>{r["name"]}</td><td>{r["n"]:,}</td>'
                       f'<td>{r["auc"]:.4f}</td><td>{r["f1"]:.4f}</td>'
                       f'<td>{r["auc_dsir"]:.4f}</td><td>{r["f1_dsir"]:.4f}</td>'
                       f'<td>{badge}</td><td>{r["r"]}</td>'
                       f'<td>{r["gap"]:.2f}</td><td><code>{r["strat"]}</code></td>'
                       f'<td>{r["dacp"]:.1f}%</td><td>{r["gain"]:+.1f}%</td>'
                       f'<td>{r["pl_acc"]:.1f}%</td></tr>\n')

    detect_rows = f'<tr style="background:#e0f2fe"><td><b>CoDefenderC3</b></td><td>{sdd_det_rate:.0f}%</td><td>--</td><td>--</td><td>--</td></tr>\n'
    for bn in (stat_comp or {}):
        c = stat_comp[bn]
        bl_rate = sum(r.get(f"bl_{bn}", False) for r in results)/max(len(results),1)*100
        sig = "Y" if c["significant_corrected"] else "."
        detect_rows += f'<tr><td>{bn}</td><td>{bl_rate:.0f}%</td><td>--</td><td>{c["raw_p"]:.4f}</td><td>{sig} h={c["cohens_h"]:.3f}</td></tr>\n'

    llm_section = ""
    for r in results:
        if r.get("llm_report"):
            attck_tags = " ".join(f'<span style="background:#dbeafe;padding:2px 6px;border-radius:3px;font-size:12px">{a.get("technique_id","?")} {a.get("technique_name","")}</span>'
                                  for a in r.get("llm_attck", []))
            llm_section += f'<h4>{r["name"]} [{r.get("llm_severity","?")}]</h4>{attck_tags}<pre>{r["llm_report"]}</pre>\n'

    abl_html = ""
    if ablation_results:
        for key, title in [("K", "Model Count"), ("weights", "ws:wr Weights"), ("batch", "Batch Size")]:
            items = ablation_results.get(key, [])
            if not items: continue
            abl_html += f"<h3>{title}</h3><table><tr>"
            cols = list(items[0].keys())
            for c in cols: abl_html += f"<th>{c}</th>"
            abl_html += "</tr>"
            for r in items:
                abl_html += "<tr>" + "".join(f"<td>{r.get(c,'')}</td>" for c in cols) + "</tr>"
            abl_html += "</table>"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>CoDefenderC3 Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:20px;color:#1e293b;background:#f8fafc}}
h1{{color:#1e40af;border-bottom:3px solid #2563eb;padding-bottom:12px}}
h2{{color:#1e3a5f;margin-top:32px}} h3{{color:#334155}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}}
th,td{{border:1px solid #e2e8f0;padding:6px 10px;text-align:center}}
th{{background:#f1f5f9;font-weight:600}} pre{{background:#f1f5f9;padding:12px;border-radius:6px;font-size:12px;overflow-x:auto}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0}}
.card{{background:white;border-radius:8px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.card h3{{margin:0;font-size:11px;color:#64748b;text-transform:uppercase}} .card .v{{font-size:24px;font-weight:700;color:#1e40af;margin-top:4px}}
img{{max-width:100%;border-radius:8px;margin:8px 0}}
</style></head><body>
<h1>CoDefenderC3 Report</h1>
<p>K={K} models | V={ens.V} views | views={sorted(ens.view_groups.keys())} | {time.strftime('%Y-%m-%d %H:%M')}</p>
<div class="cards">
<div class="card"><h3>Drift Months</h3><div class="v">{len(dm)}/{len(results)}</div></div>
<div class="card"><h3>DACP Gain</h3><div class="v">{avg_gain:+.1f}%</div></div>
<div class="card"><h3>DetRate / Months</h3><div class="v">{sdd_det_rate:.0f}% / {len(dm)+len(cl)}</div></div>
<div class="card"><h3>SDD Latency</h3><div class="v">{avg_timing:.1f}ms</div></div>
<div class="card"><h3>Runtime</h3><div class="v">{sum(timings.values()):.0f}s</div></div>
</div>
<h2>Monthly Performance</h2>{embed("fig1_performance.png")}
<table><tr><th>Month</th><th>N</th><th>AUC</th><th>F1</th><th>AUC+DSIR</th><th>F1+DSIR</th><th>SDD</th><th>r</th><th>Gap</th><th>Strategy</th><th>DACP</th><th>Gain</th><th>PL%</th></tr>{month_rows}</table>
<h2>SDD Detection</h2>{embed("fig2_sdd.png")}
<h2>DACP Coverage</h2>{embed("fig3_dacp.png")}
<h2>Baseline Comparison</h2>{embed("fig4_baselines.png")}
<table><tr><th>Method</th><th>DetRate</th><th>--</th><th>McNemar p</th><th>Sig / h</th></tr>{detect_rows}</table>
<h2>View Attribution</h2>{embed("fig5_views.png")}
<h2>Gap vs PL</h2>{embed("fig6_gap_pl.png")}
<h2>LLM Analysis</h2>{llm_section or "<p>No drift detected.</p>"}
<h2>Per-Model Accuracy</h2>{embed("fig7_permodel.png")}
<h2>Ablation</h2>{embed("fig8_ablation.png")}{abl_html or "<p>Set RUN_ABLATION=True in config.py</p>"}
<hr><p style="color:#94a3b8;font-size:11px">CoDefenderC3 | train={timings.get('train',0):.0f}s eval={timings.get('eval',0):.0f}s total={sum(timings.values()):.0f}s</p>
</body></html>"""

    hp = os.path.join(output_dir, "report.html")
    with open(hp, "w", encoding="utf-8") as f: f.write(html)
    print(f"  HTML -> {hp}")


# ═══════════════════════════════════════
#  Phase 10: 保存结果
# ═══════════════════════════════════════
def save_outputs(results, output_dir, baselines):
    class _SafeEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, set):
                return list(o)
            return str(o)

        def encode(self, o):
            # 递归将 NaN/Inf 替换为 None
            return super().encode(self._sanitize(o))

        def _sanitize(self, obj):
            if isinstance(obj, (float, np.floating)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                if isinstance(obj, np.floating):
                    return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return self._sanitize(obj.tolist())
            if isinstance(obj, dict):
                return {k: self._sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [self._sanitize(v) for v in obj]
            if isinstance(obj, set):
                return [self._sanitize(v) for v in obj]
            return obj

    jp = os.path.join(output_dir, "results.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=_SafeEncoder, ensure_ascii=False)

    cp = os.path.join(output_dir, "summary.csv")
    bl_keys = list(baselines.keys())
    with open(cp, "w", encoding="utf-8") as f:
        cols = ["Month","N","AUC","F1","AUC_DSIR","F1_DSIR","Drift","r","Gap","Strategy",
                "DACP","StdCP","Gain","PL_Acc","SDD_ms"] + bl_keys
        f.write(",".join(cols) + "\n")
        for r in results:
            vals = [r["name"], r["n"], f'{r["auc"]:.4f}', f'{r["f1"]:.4f}',
                    f'{r["auc_dsir"]:.4f}', f'{r["f1_dsir"]:.4f}',
                    "Y" if r["drift"] else "N", r["r"], f'{r["gap"]:.2f}', r["strat"],
                    f'{r["dacp"]:.1f}', f'{r["stdcp"]:.1f}', f'{r["gain"]:.1f}',
                    f'{r["pl_acc"]:.1f}', f'{r.get("sdd_timing",{}).get("total_ms",0):.1f}']
            vals += ["Y" if r.get(f"bl_{bn}") else "N" for bn in bl_keys]
            f.write(",".join(str(v) for v in vals) + "\n")
    print(f"  JSON -> {jp}\n  CSV  -> {cp}")


# ═══════════════════════════════════════
#  主入口 — 一键运行
# ═══════════════════════════════════════
def run_on_dataset(data_dir):
    """在单个数据集上运行完整实验管线。"""
    config.DATA_DIR = data_dir
    dataset_name = os.path.basename(data_dir)

    output_dir = os.path.join(config.OUTPUT_DIR, dataset_name)
    figdir = os.path.join(output_dir, "figures")
    os.makedirs(figdir, exist_ok=True)

    # 每个数据集独立的模型存储路径
    config.MODEL_PATH = os.path.join(output_dir, "ensemble.pkl")

    print(f"""
    ╔═══════════════════════════════════════════════════╗
    ║       CoDefenderC3 — {dataset_name:^24s}  ║
    ╚═══════════════════════════════════════════════════╝""")
    print(f"  数据目录: {data_dir}")
    print(f"  输出目录: {output_dir}")

    # Phase 1-4
    loader, ens, sdd, baselines, cal_s, cal_y, timings, cache_dir = setup_pipeline()

    # Phase 5
    t_eval = time.time()
    results, per_model_accs = run_monthly_eval(loader, ens, sdd, baselines, cal_s, cal_y, cache_dir)
    timings["eval"] = time.time() - t_eval

    # Checkpoint
    save_outputs(results, output_dir, baselines)
    print(f"  [checkpoint] 评估结果已保存")

    # Phase 6
    ablation_results = {}
    if config.RUN_ABLATION:
        try:
            ablation_results = run_ablation(loader, ens, sdd, cache_dir, figdir)
        except Exception as e:
            print(f"  [WARN] 消融实验失败: {e} (跳过)")

    # Phase 7
    try:
        nfig = generate_figures(results, per_model_accs, figdir, ens)
    except Exception as e:
        nfig = 0
        print(f"  [WARN] 图表生成失败: {e} (跳过)")

    # Phase 8
    try:
        stat_comp = statistical_analysis(results, baselines)
    except Exception as e:
        stat_comp = {}
        print(f"  [WARN] 统计分析失败: {e} (跳过)")

    # Phase 9
    K = ens.K
    hdr("9. 总结")
    n_drift = sum(1 for r in results if r.get("drift"))
    print(f"\n  数据集: {dataset_name}")
    print(f"  模型: K={ens.K}, V={ens.V}")
    print(f"  漂移检测: {n_drift}/{len(results)} 月")
    print(f"  耗时: {sum(timings.values()):.0f}s")

    # Phase 10
    hdr("10. 保存")
    save_outputs(results, output_dir, baselines)
    try:
        generate_html_report(results, figdir, output_dir, timings,
                             ablation_results, stat_comp, ens)
    except Exception as e:
        print(f"  [WARN] HTML 报告失败: {e}")

    return dataset_name, results


def main():
    t0 = time.time()
    import gc

    # ── 查找所有已构建数据集 ──
    data_root = config.DATA_DIR
    os.makedirs(data_root, exist_ok=True)

    # 扫描 data_root 下的数据集
    # 两种类型:
    #   - EMBER 类: 有 X.npy (预提取特征)
    #   - PE_RAW 类: 有 metadata.json + samples/ (原始PE, 需 BinaryNinja 提取)
    datasets = []
    if os.path.exists(os.path.join(data_root, "X.npy")):
        datasets.append(data_root)
    elif os.path.exists(os.path.join(data_root, "metadata.json")):
        datasets.append(data_root)
    else:
        for sub in sorted(os.listdir(data_root)):
            sub_path = os.path.join(data_root, sub)
            if not os.path.isdir(sub_path):
                continue
            if os.path.exists(os.path.join(sub_path, "X.npy")):
                datasets.append(sub_path)
            elif os.path.exists(os.path.join(sub_path, "metadata.json")):
                datasets.append(sub_path)

    # 排序: RawPE_data 优先, 其余按名称
    def _sort_key(d):
        name = os.path.basename(d)
        if name == "RawPE_data":
            return (0, name)
        return (1, name)
    datasets.sort(key=_sort_key)

    # 没有任何数据集 → 构建 ember2018
    if not datasets:
        print(f"\n  {data_root} 下无已构建数据集, 构建 ember2018...")
        from build_dataset import build_dataset, DATASETS
        build_dataset(which="ember2018")
        built = DATASETS["ember2018"]
        if os.path.exists(os.path.join(built, "X.npy")):
            datasets.append(built)
        else:
            print(f"\n  ✗ 数据集构建失败")
            print(f"    请手动运行: python build_dataset.py ember2018")
            sys.exit(1)

    print(f"\n  发现 {len(datasets)} 个数据集:")
    for d in datasets:
        print(f"    - {os.path.basename(d)}")

    # ── 在每个数据集上运行实验 ──
    all_results = {}
    for data_dir in datasets:
        dataset_name = os.path.basename(data_dir)
        # 每个数据集在独立子进程中运行, 防止内存泄漏
        print(f"\n  运行 {dataset_name} (独立子进程)...")
        ret = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--single", data_dir],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if ret.returncode == 0:
            all_results[dataset_name] = True
            print(f"  ✓ {dataset_name}")
        else:
            print(f"  ✗ {dataset_name} 退出码 {ret.returncode}")

    hdr("DONE")
    print(f"\n  完成 {len(all_results)}/{len(datasets)} 个数据集")
    for name in all_results:
        print(f"    ✓ {name}")
    print(f"  总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--single":
        # 子进程: 运行单个数据集
        data_dir = sys.argv[2]
        try:
            run_on_dataset(data_dir)
        except Exception as e:
            print(f"\n  ✗ {os.path.basename(data_dir)} 失败: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)
    else:
        main()
