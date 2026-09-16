"""
engine/result_formatter — 扫描结果格式化
=======================================
单样本检测结果的标准输出结构。
"""
import numpy as np
from engine.virus_namer import generate_virus_name


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _evidence_to_level_dict(evidence_row, level_names):
    if evidence_row.ndim == 2:
        evidence_row = evidence_row[0]
    result = {}
    for index, name in enumerate(level_names):
        result[name] = _to_native(evidence_row[index]) if index < len(evidence_row) else 0.0
    return result


_STATUS_LABELS = {0: "Benign", 1: "Malicious", 2: "DriftSuspect"}


def _status_label(status: int) -> str:
    return _STATUS_LABELS.get(status, "Unknown")


def format_scan_result(evidence: dict, identifiers: dict, pack_info: dict, model_names: list) -> dict:
    predictions = evidence["predictions"]
    scores = evidence["scores"]
    per_model_scores = evidence["per_model_scores"]
    per_model_preds = evidence["per_model_preds"]
    drift_evidence = evidence["drift_evidence"]
    sample_drift_type = evidence["sample_drift_type"]
    anomaly_score = evidence["anomaly_score"]
    is_drift_candidate = evidence["is_drift_candidate"]
    model_stats = evidence.get("model_stats")  # 可选，由 app.py 传入

    pred = int(predictions[0]) if predictions.ndim == 1 else int(predictions[0, 0])
    score = float(scores[0]) if scores.ndim == 1 else float(scores[0, 0])
    drift_type = str(sample_drift_type[0]) if hasattr(sample_drift_type, "__len__") else str(sample_drift_type)
    is_drift = bool(is_drift_candidate[0]) if hasattr(is_drift_candidate, "__len__") else bool(is_drift_candidate)
    anomaly = float(anomaly_score[0]) if hasattr(anomaly_score, "__len__") else float(anomaly_score)

    if pred == 1:
        status = 1
    elif is_drift:
        status = 2
    else:
        status = 0

    pms = per_model_scores[0] if per_model_scores.ndim == 2 else per_model_scores
    pmp = per_model_preds[0] if per_model_preds.ndim == 2 else per_model_preds
    per_model_scores_dict = {name: _to_native(pms[index]) for index, name in enumerate(model_names)}
    per_model_preds_dict = {name: _to_native(pmp[index]) for index, name in enumerate(model_names)}

    level_names = ["surface", "structure", "global"]
    drift_values = drift_evidence[0] if drift_evidence.ndim == 2 else drift_evidence
    drift_evidence_dict = _evidence_to_level_dict(drift_values, level_names)

    virus_name = ""
    if status > 0:
        virus_name = generate_virus_name(
            drift_type=drift_type,
            per_model_scores=pms,
            model_names=model_names,
            sha256=identifiers.get("SHA256", ""),
        )

    # ---- 组装 ScanResult ----
    scan_result = {
        "VirusName": virus_name,
        "Packed": {
            "IsPacked": pack_info.get("packed", False),
            "PackerName": pack_info.get("packer_name", ""),
            "IsArchive": pack_info.get("archive", False),
            "ArchiveName": pack_info.get("archive_name", ""),
        },
        "PerModelScores": per_model_scores_dict,
        "PerModelPreds": per_model_preds_dict,
        "DriftEvidence": drift_evidence_dict,
        "SampleDriftType": drift_type,
        "AnomalyScore": anomaly,
        "IsDriftCandidate": is_drift,
    }

    # 可选：模型投票统计（仅当 app.py 传入时包含）
    if model_stats:
        scan_result["ModelStats"] = model_stats

    return {
        "MD5": identifiers["MD5"],
        "SHA256": identifiers["SHA256"],
        "CRC32": identifiers["CRC32"],
        "FileName": identifiers["FileName"],
        "FileSize": identifiers["FileSize"],
        "Status": status,
        "StatusLabel": _status_label(status),
        "IsMalicious": pred == 1,
        "Score": score,
        "ScanResult": scan_result,
    }
