"""
engine/virus_namer — VirusName 生成
=====================================
"""
import numpy as np

VIEW_FAMILY_MAP = {
    "V2_byte_stat": "ByS",
    "V3_pe_struct": "PEs",
    "V4_import": "Imp",
    "V5_string": "Str",
    "V10_metadata": "Met",
    "V11_ensemble": "Ens",
}

DRIFT_PREFIX_MAP = {
    "perturbation": "Gen",
    "structural": "GenKD",
    "paradigm_shift": "GenKD.2",
    "none": "",
}

MALWARE_TYPE_BY_VIEW = {
    "V3_pe_struct": "Trojan",
    "V4_import": "Trojan",
    "V5_string": "Worm",
    "V2_byte_stat": "Virus",
    "V10_metadata": "Trojan",
    "V11_ensemble": "Virus",
}


def generate_virus_name(drift_type: str, per_model_scores: np.ndarray,
                        model_names: list, sha256: str = "") -> str:
    strongest_view = _find_strongest_view(per_model_scores, model_names)
    family = VIEW_FAMILY_MAP.get(strongest_view, "Unk")
    suffix = sha256[:8] if sha256 else "00000000"

    if drift_type in DRIFT_PREFIX_MAP and DRIFT_PREFIX_MAP[drift_type]:
        prefix = DRIFT_PREFIX_MAP[drift_type]
        return f"{prefix}.{family}.{suffix}"

    mal_type = MALWARE_TYPE_BY_VIEW.get(strongest_view, "Trojan")
    return f"{mal_type}.{family}.{suffix}"


def _find_strongest_view(per_model_scores: np.ndarray, model_names: list) -> str:
    try:
        import config
        view_groups = getattr(config, "VIEW_GROUPS_EMBER", None) or getattr(config, "VIEW_GROUPS", {})
    except Exception:
        view_groups = {}

    model_to_view = {}
    for view_name, view_info in view_groups.items():
        for model_tuple in view_info.get("models", []):
            if isinstance(model_tuple, (list, tuple)) and model_tuple:
                model_to_view[model_tuple[0]] = view_name

    if len(per_model_scores) == 0:
        return "V11_ensemble"

    best_idx = int(np.argmax(per_model_scores))
    if best_idx < len(model_names):
        best_model = model_names[best_idx]
        return model_to_view.get(best_model, "V11_ensemble")

    return "V11_ensemble"
