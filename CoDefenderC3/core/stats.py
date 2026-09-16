"""
Statistical Tests (Paper §7.10.2, §7.14)
==========================================
v3: added bootstrap CI for detection rate, Wilcoxon signed-rank test for
    continuous metrics, effect size interpretation, LaTeX table export.
"""
import numpy as np
from scipy import stats as sp_stats


def mcnemar_exact(det_a, det_b):
    """McNemar's exact test for paired binary outcomes.

    det_a, det_b: lists of bools (same length).
    Returns (statistic, p_value, a_only, b_only).
    """
    a, b = np.asarray(det_a, bool), np.asarray(det_b, bool)
    a_only = int(np.sum(a & ~b))   # a detects, b misses
    b_only = int(np.sum(~a & b))   # b detects, a misses
    n = a_only + b_only
    if n == 0:
        return 0.0, 1.0, 0, 0
    stat = (abs(a_only - b_only) - 1)**2 / max(n, 1)
    # Exact binomial test (more accurate for small n)
    p = float(sp_stats.binomtest(min(a_only, b_only), n, 0.5).pvalue)
    return float(stat), p, a_only, b_only


def holm_bonferroni(p_values, alpha=0.05):
    """Holm-Bonferroni correction for multiple comparisons.

    Returns list of (p, rank, threshold, significant).
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * m
    any_rejected = True
    for rank_0, (orig_idx, p) in enumerate(indexed):
        rank = rank_0 + 1
        thr = alpha / (m - rank_0)
        sig = bool(p < thr and any_rejected)
        if not sig:
            any_rejected = False
        results[orig_idx] = (float(p), rank, float(thr), sig)
    return results


def cohens_h(p1, p2):
    """Cohen's h effect size for two proportions."""
    return float(2 * np.arcsin(np.sqrt(max(0, min(1, p1))))
                 - 2 * np.arcsin(np.sqrt(max(0, min(1, p2)))))


def cohens_d(x1, x2):
    """Cohen's d effect size for two continuous samples."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    if len(x1) < 2 or len(x2) < 2:
        return 0.0
    pooled = np.sqrt(((len(x1)-1)*x1.var(ddof=1) + (len(x2)-1)*x2.var(ddof=1))
                     / max(len(x1) + len(x2) - 2, 1))
    return float((x1.mean() - x2.mean()) / max(pooled, 1e-8))


def effect_size_label(h):
    """Interpret Cohen's h/d magnitude."""
    ah = abs(h)
    if ah < 0.2:   return "negligible"
    if ah < 0.5:   return "small"
    if ah < 0.8:   return "medium"
    return "large"


def bootstrap_ci(data, stat_fn=np.mean, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    boots = [float(stat_fn(data[rng.choice(len(data), len(data), replace=True)]))
             for _ in range(n_boot)]
    lo = np.percentile(boots, 100 * (1 - ci) / 2)
    hi = np.percentile(boots, 100 * (1 + ci) / 2)
    return float(lo), float(hi)


def wilcoxon_test(x, y):
    """Wilcoxon signed-rank test for paired continuous data."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) < 3:
        return 0.0, 1.0
    stat, p = sp_stats.wilcoxon(diff)
    return float(stat), float(p)


def full_comparison(sdd_det, baseline_dict, alpha=0.05):
    """
    Complete statistical comparison: McNemar + Holm-Bonferroni + Cohen's h.

    Parameters
    ----------
    sdd_det : list of bools for SDD detections
    baseline_dict : {name: list of bools}

    Returns dict of {name: {raw_p, holm_rank, holm_threshold,
                            significant_corrected, cohens_h, effect_size}}.
    """
    names = list(baseline_dict.keys())
    p_vals = []
    raw_results = {}

    sdd_rate = np.mean(sdd_det)

    for bn in names:
        stat, p, a_only, b_only = mcnemar_exact(sdd_det, baseline_dict[bn])
        bl_rate = np.mean(baseline_dict[bn])
        h = cohens_h(sdd_rate, bl_rate)
        raw_results[bn] = {
            "mcnemar_stat": stat, "raw_p": p,
            "sdd_only": a_only, "bl_only": b_only,
            "sdd_rate": float(sdd_rate), "bl_rate": float(bl_rate),
            "cohens_h": h, "effect_size": effect_size_label(h),
        }
        p_vals.append(p)

    # Holm-Bonferroni correction
    holm = holm_bonferroni(p_vals, alpha)
    for i, bn in enumerate(names):
        _, rank, thr, sig = holm[i]
        raw_results[bn].update({
            "holm_rank": rank,
            "holm_threshold": thr,
            "significant_corrected": sig,
        })

    # Bootstrap CI for SDD detection rate
    if len(sdd_det) >= 3:
        ci_lo, ci_hi = bootstrap_ci(sdd_det, np.mean)
        for bn in names:
            raw_results[bn]["sdd_rate_ci"] = (ci_lo, ci_hi)

    return raw_results


def comparison_latex(comp, caption="Statistical comparison"):
    """Export comparison results as LaTeX table."""
    rows = []
    rows.append(r"\begin{tabular}{lrrrrcrl}")
    rows.append(r"\toprule")
    rows.append(r"Baseline & TPR & $p$ & Rank & Thr & Sig & $h$ & Effect \\")
    rows.append(r"\midrule")
    for bn, c in comp.items():
        sig = r"\checkmark" if c["significant_corrected"] else ""
        rows.append(f"{bn} & {c['bl_rate']:.2f} & {c['raw_p']:.4f} & "
                    f"{c['holm_rank']} & {c['holm_threshold']:.4f} & "
                    f"{sig} & {c['cohens_h']:.3f} & {c['effect_size']} \\\\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    return "\n".join(rows)
