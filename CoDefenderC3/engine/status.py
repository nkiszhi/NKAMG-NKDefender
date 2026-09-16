"""
engine/status — 引擎状态与健康检查
===================================
"""
import os
import time

ENGINE_VERSION = "6.0.0"
ENGINE_NAME = "CoDefenderC3_v6"


def get_engine_info(ensemble=None, sdd_engine=None, model_path: str = "") -> dict:
    model_count = 0
    view_count = 0
    if ensemble is not None:
        model_count = getattr(ensemble, "K", 0)
        view_count = getattr(ensemble, "V", 0)

    sdd_status = "not_initialized"
    sdd_levels = []
    if sdd_engine is not None:
        sdd_status = "calibrated" if getattr(sdd_engine, "calibrated", False) else "not_calibrated"
        sdd_levels = getattr(sdd_engine, "level_names", [])

    virus_db_version = "unknown"
    if model_path and os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        virus_db_version = time.strftime("%Y%m%d.%H%M%S", time.localtime(mtime))

    return {
        "EngineName": ENGINE_NAME,
        "EngineVersion": ENGINE_VERSION,
        "Mode": "data_driven",
        "ModelCount": model_count,
        "ViewCount": view_count,
        "SDDStatus": sdd_status,
        "SDDLevels": sdd_levels,
        "VirusDBVersion": virus_db_version,
        "Capabilities": [
            "malware_detection",
            "drift_detection",
            "drift_classification",
            "per_model_scores",
            "packer_detection",
            "model_hot_update",
        ],
    }


def get_health(ensemble=None, sdd_engine=None) -> dict:
    issues = []

    if ensemble is None:
        issues.append("ensemble_not_loaded")
    elif not getattr(ensemble, "_fitted", False):
        issues.append("ensemble_not_fitted")

    if sdd_engine is None:
        issues.append("sdd_not_initialized")
    elif not getattr(sdd_engine, "calibrated", False):
        issues.append("sdd_not_calibrated")

    if not issues:
        status = "ok"
    elif issues == ["sdd_not_calibrated"]:
        status = "degraded"
    else:
        status = "error"

    return {
        "status": status,
        "issues": issues,
    }
