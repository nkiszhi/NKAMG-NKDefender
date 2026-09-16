"""
Baseline drift detectors — conceptual reimplementations (Paper §7.8).
=====================================================================
7 methods, each using different drift detection paradigms:
  1. Transcend  [USENIX'17]: KS test on nonconformity measures
  2. CADE       [USENIX'21]: Contrastive AE (PCA proxy) reconstruction distance
  3. HCC        [USENIX'23]: Prediction uncertainty via MC-Dropout simulation
  4. DroidEvolver [EuroS&P'19]: Self-poisoning via pseudo-label tracking
  5. DREAM      [CCS'25]: Class-conditional centroid shift + reliability
  6. MADCAT     [ICML'25 WS]: Confidence distribution shift
  7. Ens.Disagree: Pairwise model disagreement shift

Input: scores matrix (N, K) with per-model probability scores ∈ [0,1].
All return {"drift_detected": bool, ...} with diagnostic info.
"""
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu


# ────────────────────────────────────────────────────────────────────
#  1. Transcend — KS test on subsampled nonconformity measures
# ────────────────────────────────────────────────────────────────────
class TranscendBaseline:
    """Transcend [USENIX'17]: Distribution test on nonconformity scores.

    NCM = model uncertainty = 1 - max(p, 1-p).
    Uses both per-model mean uncertainty AND model disagreement (std).
    Subsamples calibration to avoid large-N oversensitivity in KS test.
    """
    def __init__(self, sig=0.05, max_cal=3000):
        self.sig = sig
        self.max_cal = max_cal
        self._rng = np.random.RandomState(42)

    def calibrate(self, scores):
        # Two NCM signals: uncertainty + disagreement
        ens = scores.mean(1)
        self._cal_unc = 1.0 - np.maximum(ens, 1 - ens)         # uncertainty
        self._cal_dis = scores.std(1) if scores.ndim > 1 else np.zeros_like(ens)  # disagreement
        # Combined NCM (both matter)
        self._cal_ncm = self._cal_unc + self._cal_dis
        # Subsample for stable KS test
        if len(self._cal_ncm) > self.max_cal:
            idx = self._rng.choice(len(self._cal_ncm), self.max_cal, replace=False)
            self._cal_sub = self._cal_ncm[idx]
        else:
            self._cal_sub = self._cal_ncm
        self._cal_mean = float(self._cal_ncm.mean())
        self._cal_std = float(max(self._cal_ncm.std(), 1e-6))

    def detect(self, scores):
        ens = scores.mean(1) if scores.ndim > 1 else scores
        unc = 1.0 - np.maximum(ens, 1 - ens)
        dis = scores.std(1) if scores.ndim > 1 else np.zeros_like(ens)
        ncm = unc + dis
        # KS test on subsampled calibration vs full test
        stat, p = ks_2samp(self._cal_sub, ncm)
        # Mean shift in standard deviations
        shift = abs(ncm.mean() - self._cal_mean) / self._cal_std
        # Require both statistical significance AND practical significance
        # min_ks_stat scales with sqrt of smaller sample size to avoid N-sensitivity
        n_eff = min(len(self._cal_sub), len(ncm))
        min_stat = max(0.04, 1.0 / np.sqrt(n_eff) * 3)  # ~3/sqrt(n) floor
        detected = bool((p < self.sig and stat > min_stat) or shift > 2.5)
        return {"drift_detected": detected, "p_value": float(p),
                "ks_statistic": float(stat), "mean_shift_sigma": float(shift),
                "min_stat_threshold": float(min_stat)}


# ────────────────────────────────────────────────────────────────────
#  2. CADE — PCA reconstruction error + distributional test
# ────────────────────────────────────────────────────────────────────
class CADEBaseline:
    """CADE [USENIX'21]: Contrastive autoencoder proxy via PCA.

    Uses PCA reconstruction error as anomaly measure.
    Detects drift via Welch t-test comparing cal vs test reconstruction errors,
    PLUS ratio of outlier samples exceeding calibration threshold.
    """
    def __init__(self, latent_frac=0.5, pct=95):
        self.latent_frac = latent_frac
        self.pct = pct

    def fit(self, X):
        self.mu = X.mean(0)
        C = X - self.mu
        # PCA: keep fraction of dimensions
        k = max(2, int(X.shape[1] * self.latent_frac))
        k = min(k, X.shape[1] - 1, X.shape[0] - 1)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        self.V = Vt[:k]
        # Reconstruction error per sample
        Z = C @ self.V.T
        recon = Z @ self.V
        errors = np.sqrt(((C - recon) ** 2).sum(1))
        self._cal_errors = errors
        self._thr = float(np.percentile(errors, self.pct))
        self._cal_mean = float(errors.mean())
        self._cal_std = float(max(errors.std(), 1e-6))

    def detect(self, X):
        C = X - self.mu
        Z = C @ self.V.T
        recon = Z @ self.V
        errors = np.sqrt(((C - recon) ** 2).sum(1))
        # Outlier ratio: fraction exceeding calibration threshold
        outlier_ratio = float(np.mean(errors > self._thr))
        # Mean error shift
        mean_err = float(errors.mean())
        shift = (mean_err - self._cal_mean) / self._cal_std
        # Two criteria: outlier ratio OR significant mean shift
        detected = bool(outlier_ratio > 0.10 or shift > 2.0)
        return {"drift_detected": detected, "mean_error": mean_err,
                "outlier_ratio": outlier_ratio, "threshold": self._thr,
                "error_shift_sigma": float(shift)}


# ────────────────────────────────────────────────────────────────────
#  3. HCC — MC-Dropout prediction uncertainty
# ────────────────────────────────────────────────────────────────────
class HCCBaseline:
    """HCC [USENIX'23]: MC-Dropout uncertainty estimation.

    Simulates MC-Dropout by applying model-level binary masks
    (dropping entire models with probability p). Compares prediction
    variance distribution between calibration and test.
    """
    def __init__(self, n_forward=30, dropout_rate=0.25):
        self.n_fw = n_forward
        self.drop_rate = dropout_rate

    def _mc_variance(self, scores, seed):
        """Compute per-sample MC-Dropout variance."""
        K = scores.shape[1]
        rng = np.random.RandomState(seed)
        ens_list = []
        for _ in range(self.n_fw):
            mask = (rng.random(K) > self.drop_rate).astype(float)
            if mask.sum() == 0:
                mask[rng.randint(K)] = 1.0
            ens = (scores * mask[None, :]).sum(1) / mask.sum()
            ens_list.append(ens)
        return np.stack(ens_list, 0).var(0)  # (N,)

    def calibrate(self, scores):
        var_cal = self._mc_variance(scores, seed=42)
        self._cal_var_mean = float(var_cal.mean())
        self._cal_var_std = float(max(var_cal.std(), 1e-8))
        self._cal_var_p90 = float(np.percentile(var_cal, 90))

    def detect(self, scores):
        var_test = self._mc_variance(scores, seed=123)
        mean_var = float(var_test.mean())
        # Fraction exceeding calibration 90th percentile
        exceed_ratio = float(np.mean(var_test > self._cal_var_p90))
        # Mean variance shift
        shift = (mean_var - self._cal_var_mean) / self._cal_var_std
        # Detect: significant increase in uncertainty
        detected = bool(exceed_ratio > 0.15 or shift > 2.0)
        return {"drift_detected": detected, "mean_var": mean_var,
                "exceed_ratio": exceed_ratio,
                "var_shift_sigma": float(shift)}


# ────────────────────────────────────────────────────────────────────
#  4. DroidEvolver — Self-poisoning simulation
# ────────────────────────────────────────────────────────────────────
class DroidEvolverBaseline:
    """DroidEvolver [EuroS&P'19]: Blind pseudo-label self-update detection.

    Tracks cumulative prediction entropy and class ratio drift.
    DroidEvolver itself doesn't detect drift — it suffers from it.
    We detect drift by observing symptoms: entropy increase + class ratio shift.
    Stateful: accumulates evidence across consecutive detect() calls.
    """
    def __init__(self):
        self._cal_entropy = None
        self._cal_class_ratio = None
        self._cumulative_shift = 0.0
        self._call_count = 0

    def calibrate(self, scores):
        ens = scores.mean(1) if scores.ndim > 1 else scores
        p = np.clip(ens, 1e-8, 1 - 1e-8)
        self._cal_entropy = float(np.mean(-p * np.log2(p) - (1-p) * np.log2(1-p)))
        self._cal_class_ratio = float((ens >= 0.5).mean())
        self._cal_confidence = float(np.maximum(ens, 1 - ens).mean())
        self._cumulative_shift = 0.0
        self._call_count = 0

    def detect(self, scores):
        ens = scores.mean(1) if scores.ndim > 1 else scores
        p = np.clip(ens, 1e-8, 1 - 1e-8)
        entropy = float(np.mean(-p * np.log2(p) - (1-p) * np.log2(1-p)))
        class_ratio = float((ens >= 0.5).mean())
        confidence = float(np.maximum(ens, 1 - ens).mean())
        self._call_count += 1

        # Entropy increase (models becoming uncertain)
        ent_shift = (entropy - self._cal_entropy) / max(self._cal_entropy, 1e-6)
        # Class ratio shift (label distribution changing)
        ratio_shift = abs(class_ratio - self._cal_class_ratio)
        # Confidence drop
        conf_drop = (self._cal_confidence - confidence) / max(self._cal_confidence, 1e-6)
        # Accumulate evidence of self-poisoning
        self._cumulative_shift += max(ent_shift, 0) * 0.3 + ratio_shift * 0.5

        # Detect: entropy spike, class ratio shift, or accumulated poisoning
        detected = bool(ent_shift > 0.15 or ratio_shift > 0.08
                        or conf_drop > 0.05
                        or self._cumulative_shift > 0.5)
        return {"drift_detected": detected,
                "entropy_shift": float(ent_shift),
                "class_ratio_shift": float(ratio_shift),
                "confidence_drop": float(conf_drop),
                "cumulative_shift": float(self._cumulative_shift)}


# ────────────────────────────────────────────────────────────────────
#  5. DREAM — Class-conditional centroid shift
# ────────────────────────────────────────────────────────────────────
class DREAMBaseline:
    """DREAM [CCS'25]: Class centroid shift + concept reliability measure.

    Computes class-conditional centroids in score space.
    Detects drift when: (a) test samples are far from centroids (z-score),
    or (b) class reliability drops below calibration baseline.
    """
    def __init__(self, n_sigma=2.0):
        self.n_sigma = n_sigma
        self.centroids = {}
        self.stds = {}

    def fit(self, X, y):
        for c in np.unique(y):
            m = y == c
            if m.sum() > 5:
                self.centroids[c] = X[m].mean(0)
                self.stds[c] = np.maximum(X[m].std(0), 1e-6)
        # Per-sample z-scores for calibration
        zs = []
        for c in self.centroids:
            m = y == c
            z = np.mean(np.abs(X[m] - self.centroids[c]) / self.stds[c], axis=1)
            zs.extend(z)
        zs = np.array(zs)
        self._cal_z_mean = float(zs.mean()) if len(zs) > 0 else 1.0
        self._cal_z_std = float(max(zs.std(), 0.1)) if len(zs) > 0 else 0.5
        self._cal_z_p90 = float(np.percentile(zs, 90)) if len(zs) > 0 else 2.0
        # Class reliability: fraction within 90th percentile
        self._cal_reliability = float(np.mean(zs < self._cal_z_p90))

    def detect(self, X):
        if not self.centroids:
            return {"drift_detected": False}
        # Distance to nearest centroid
        dists = []
        for c in self.centroids:
            z = np.mean(np.abs(X - self.centroids[c]) / self.stds[c], axis=1)
            dists.append(z)
        all_z = np.stack(dists, 0).min(0)
        mean_z = float(np.mean(all_z))
        # Reliability: fraction within calibration 90th percentile
        reliability = float(np.mean(all_z < self._cal_z_p90))
        # z-score shift
        z_shift = (mean_z - self._cal_z_mean) / self._cal_z_std
        # Detect: centroid shift OR reliability drop
        detected = bool(z_shift > self.n_sigma or reliability < self._cal_reliability * 0.85)
        return {"drift_detected": detected, "z_score": mean_z,
                "z_shift_sigma": float(z_shift), "reliability": reliability,
                "cal_reliability": self._cal_reliability}


# ────────────────────────────────────────────────────────────────────
#  6. MADCAT — Confidence distribution shift
# ────────────────────────────────────────────────────────────────────
class MADCATBaseline:
    """MADCAT [ICML'25 WS]: Confidence-based test-time training proxy.

    Tracks ensemble confidence distribution.
    Detects drift when confidence distribution shifts significantly
    (lower mean confidence, higher entropy, or more low-confidence samples).
    """
    def __init__(self, sig=0.05, max_cal=3000):
        self.sig = sig
        self.max_cal = max_cal
        self._rng = np.random.RandomState(42)

    def calibrate(self, scores):
        ens = scores.mean(1) if scores.ndim > 1 else scores
        conf = np.maximum(ens, 1 - ens)
        self._cal_conf = conf
        self._cal_conf_mean = float(conf.mean())
        self._cal_conf_std = float(max(conf.std(), 1e-6))
        # Entropy
        p = np.clip(ens, 1e-8, 1 - 1e-8)
        ent = -p * np.log2(p) - (1-p) * np.log2(1-p)
        self._cal_ent_mean = float(ent.mean())
        self._cal_ent_std = float(max(ent.std(), 1e-6))
        # Subsample for KS test
        if len(conf) > self.max_cal:
            idx = self._rng.choice(len(conf), self.max_cal, replace=False)
            self._cal_conf_sub = conf[idx]
        else:
            self._cal_conf_sub = conf

    def detect(self, scores):
        ens = scores.mean(1) if scores.ndim > 1 else scores
        conf = np.maximum(ens, 1 - ens)
        p = np.clip(ens, 1e-8, 1 - 1e-8)
        ent = -p * np.log2(p) - (1-p) * np.log2(1-p)
        # Confidence shift
        conf_shift = (self._cal_conf_mean - conf.mean()) / self._cal_conf_std
        # Entropy shift
        ent_shift = (ent.mean() - self._cal_ent_mean) / self._cal_ent_std
        # KS test on confidence distribution
        stat, p_val = ks_2samp(self._cal_conf_sub, conf)
        n_eff = min(len(self._cal_conf_sub), len(conf))
        min_stat = max(0.04, 1.0 / np.sqrt(n_eff) * 3)
        # Detect: confidence drop OR entropy spike OR distributional shift
        detected = bool(conf_shift > 1.5 or ent_shift > 1.5
                        or (p_val < self.sig and stat > min_stat))
        return {"drift_detected": detected,
                "conf_shift_sigma": float(conf_shift),
                "ent_shift_sigma": float(ent_shift),
                "ks_stat": float(stat), "ks_p": float(p_val)}


# ────────────────────────────────────────────────────────────────────
#  7. Ens.Disagree — Pairwise model disagreement
# ────────────────────────────────────────────────────────────────────
class EnsembleDisagreementBaseline:
    """Pairwise ensemble disagreement with bootstrapped null distribution.

    Computes mean pairwise disagreement rate between models.
    Uses calibration data to establish null distribution (mean ± std).
    Detects drift when test disagreement exceeds calibration baseline.
    """
    def __init__(self, n_sigma=2.5):
        self.n_sigma = n_sigma
        self.thr = None

    def calibrate(self, pred_list):
        ds = []
        for p in pred_list:
            if p.shape[0] > 1 and p.shape[1] > 1:
                # Pairwise disagreement rate
                d = float(np.mean(p[:, :, None] != p[:, None, :]))
                ds.append(d)
        if ds:
            self._cal_mean = float(np.mean(ds))
            self._cal_std = float(max(np.std(ds), 1e-6))
            self.thr = self._cal_mean + self.n_sigma * self._cal_std
        else:
            self._cal_mean = 0.0
            self._cal_std = 0.1
            self.thr = 0.5

    def detect(self, preds):
        if preds.shape[1] < 2:
            return {"drift_detected": False, "disagreement": 0.0}
        d = float(np.mean(preds[:, :, None] != preds[:, None, :]))
        shift = (d - self._cal_mean) / self._cal_std
        detected = bool(d > self.thr)
        return {"drift_detected": detected, "disagreement": d,
                "threshold": self.thr, "shift_sigma": float(shift)}
