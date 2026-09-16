"""
Ablation Experiments (Paper §7.13)
====================================
v4: Comprehensive fixes for realistic results.
  - §7.13.1: Model count — tracks detection, F1, AUC with progressive view removal
  - §7.13.2: ws:wr weights — shows impact on DETECTION SENSITIVITY (gap, confidence)
  - §7.13.3: Batch size — uses FULL baseline calibration, only subsamples test data
"""
import numpy as np
import time
from sklearn.metrics import f1_score, roc_auc_score
from core.sdd_engine import SDDEngine


def _metrics(y_true, scores):
    """Compute AUC and F1 safely."""
    preds = (scores >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float('nan')
    f1 = f1_score(y_true, preds, zero_division=0)
    return float(auc), float(f1)


def ablation_model_count(pred_list_bl, score_list_bl,
                         pred_test, score_test, y_test,
                         full_pg, configs=None):
    """§7.13.1: SDD with progressively fewer views/models."""
    if configs is None:
        all_views = list(full_pg.keys())
        configs = [
            {"name": "Full",     "views": all_views},
            {"name": "-1 view",  "views": all_views[:-1]},
            {"name": "-2 views", "views": all_views[:-2]},
        ]
        if len(all_views) > 3:
            configs.append({"name": "3 views", "views": all_views[:3]})
        if len(all_views) > 1:
            configs.append({"name": "1 view", "views": all_views[-1:]})

    results = []
    for cfg in configs:
        sub_pg = {v: full_pg[v] for v in cfg["views"] if v in full_pg}
        sub_models = sorted(m for v in sub_pg.values() for m in v["models"])
        K_sub = len(sub_models)

        if K_sub < 2:
            results.append({"name": cfg["name"], "K": K_sub, "V": len(sub_pg),
                            "detected": False, "r": 0, "gap": 0, "f1": 0, "auc": 0.5})
            continue

        idx_map = {old: new for new, old in enumerate(sub_models)}
        remap_pg = {}
        for vn, vi in sub_pg.items():
            remap_pg[vn] = {"models": [idx_map[m] for m in vi["models"] if m in idx_map],
                            "label": vi["label"]}

        bl_p_sub = [p[:, sub_models] for p in pred_list_bl]
        bl_s_sub = [s[:, sub_models] for s in score_list_bl]
        test_p_sub = pred_test[:, sub_models]
        test_s_sub = score_test[:, sub_models]

        sdd = SDDEngine(K_sub, remap_pg, alpha=0.05, ws=0.6, wr=0.4)
        try:
            sdd.calibrate(bl_p_sub, bl_s_sub, quiet=True)
            det = sdd.detect(test_p_sub, test_s_sub)

            ens_scores = test_s_sub.mean(axis=1)
            auc, f1 = _metrics(y_test, ens_scores)

            results.append({
                "name": cfg["name"], "K": K_sub, "V": len(sub_pg),
                "detected": det.drift_detected, "r": det.num_mechanisms,
                "gap": det.spectral_gap, "strategy": det.strategy,
                "f1": f1, "auc": auc,
            })
        except Exception as e:
            results.append({"name": cfg["name"], "K": K_sub, "V": len(sub_pg),
                            "error": str(e)})

    return results


def ablation_weights(pred_list_bl, score_list_bl,
                     pred_test, score_test, y_test, full_pg,
                     weight_configs=None):
    """§7.13.2: ws:wr impact on detection SENSITIVITY.

    Key fix: reports detection sensitivity metrics (gap, confidence, r)
    as primary results, not AUC gain (which is always ~0 for ws:wr changes).
    DSIR gain is secondary.
    """
    if weight_configs is None:
        weight_configs = [
            (1.0, 0.0, "1.0:0.0 (score only)"),
            (0.8, 0.2, "0.8:0.2"),
            (0.6, 0.4, "0.6:0.4 (default)"),
            (0.4, 0.6, "0.4:0.6"),
            (0.2, 0.8, "0.2:0.8"),
            (0.0, 1.0, "0.0:1.0 (pred only)"),
        ]

    K = pred_list_bl[0].shape[1]
    results = []
    for ws, wr, label in weight_configs:
        sdd = SDDEngine(K, full_pg, alpha=0.05, ws=ws, wr=wr)
        try:
            sdd.calibrate(pred_list_bl, score_list_bl, quiet=True)
            det = sdd.detect(pred_test, score_test)

            # Base ensemble AUC
            ens_scores = score_test.mean(axis=1)
            auc, f1 = _metrics(y_test, ens_scores)

            # DSIR-weighted
            dw = sdd.dsir_weights(det)
            ens_dsir = np.average(score_test, axis=1, weights=dw)
            auc_dsir, f1_dsir = _metrics(y_test, ens_dsir)

            results.append({
                "label": label, "ws": ws, "wr": wr,
                "detected": det.drift_detected, "r": det.num_mechanisms,
                "gap": det.spectral_gap,
                "confidence": det.confidence,
                "auc": auc, "f1": f1,
                "auc_dsir": auc_dsir, "f1_dsir": f1_dsir,
                "dsir_auc_gain": auc_dsir - auc,
            })
        except Exception as e:
            results.append({"label": label, "ws": ws, "wr": wr, "error": str(e)})

    return results


def ablation_batch_size(pred_bl, score_bl, pred_test, score_test, y_test,
                        full_pg, sizes=None, n_repeats=10):
    """§7.13.3: Batch size sensitivity.

    Key fix: uses FULL baseline for calibration (not subsampled).
    Only subsamples the TEST data. This avoids noisy calibration → FPR explosion.
    """
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000, 2000]

    K = pred_bl[0].shape[1]
    rng = np.random.RandomState(42)
    N_test = pred_test.shape[0]

    # Calibrate ONCE on full baseline (not per-repeat)
    sdd_full = SDDEngine(K, full_pg, alpha=0.05, ws=0.6, wr=0.4)
    sdd_full.calibrate(pred_bl, score_bl, quiet=True)

    results = []
    for N in sizes:
        if N > N_test:
            continue
        detections, fprs, timings, rs = [], [], [], []

        for rep in range(n_repeats):
            # Subsample TEST only
            idx = rng.choice(N_test, min(N, N_test), replace=False)

            t = time.perf_counter()
            det = sdd_full.detect(pred_test[idx], score_test[idx])
            elapsed = (time.perf_counter() - t) * 1000

            detections.append(det.drift_detected)
            rs.append(det.num_mechanisms)
            timings.append(elapsed)

            # FPR: test on random subset of calibration data
            bl_idx = rng.randint(len(pred_bl))
            cal_p = pred_bl[bl_idx]
            cal_s = score_bl[bl_idx]
            n_cal = min(N, len(cal_p))
            ci = rng.choice(len(cal_p), n_cal, replace=False)
            det_clean = sdd_full.detect(cal_p[ci], cal_s[ci])
            fprs.append(det_clean.drift_detected)

        results.append({
            "N": N,
            "detection_rate": float(np.mean(detections)),
            "fpr": float(np.mean(fprs)),
            "avg_r": float(np.mean(rs)),
            "avg_ms": float(np.mean(timings)),
            "std_ms": float(np.std(timings)),
        })

    return results
