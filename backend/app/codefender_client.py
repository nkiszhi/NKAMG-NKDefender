import os
import re
import shutil
# Docker CLI is used as an optional local integration, without shell=True.
import subprocess  # nosec B404
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import requests

from .config import settings


DEFAULT_MODEL_NAMES = [
    "MalConv", "MalConv2", "ByteTransformer", "SaxeBerlinDNN", "EmberDNN",
    "NatarajKNN", "DrebinSVM", "DL4MD", "PEMinerRF", "DrebinImport",
    "ALOHANet", "EmberGBDT_Import", "RaffFeatureNet", "DrebinString",
    "OpcodeLSTM", "OpcodeTransformer", "OpcodeStatNet", "AsmEmbedNet",
    "MalGraph", "CFGGAT", "CFGGCN", "CFGDGCNN", "CallGraphGNN",
    "GraphStatNet", "GrayscaleCNN", "InceptionV3", "IMCFN", "ColorCNN",
    "MarkovCNN", "EntropyMapCNN", "HashEmbedNet", "MetadataNet", "ResourceNet",
]


def scan_file(file_path: Path) -> Dict[str, Dict[str, Any]]:
    path_result = _scan_by_path(file_path)
    if path_result is not None:
        return path_result

    upload_result = _scan_by_upload(file_path)
    if upload_result is not None:
        return upload_result

    return {
        "CoDefender集成模型": {"probability": None, "result": "预测失败"},
        "集成结果": {"probability": None, "result": "无有效预测"},
    }


def _scan_by_path(file_path: Path) -> Optional[Dict[str, Dict[str, Any]]]:
    endpoint = _normalize_endpoint(settings.codefender_scan_endpoint)
    if not endpoint:
        return None

    url = f"{settings.codefender_api_base}{endpoint}"
    payload = {"file_path": _prepare_scan_path(file_path)}
    try:
        response = requests.post(url, json=payload, timeout=settings.codefender_timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return parse_codefender_response(response.json())
    except Exception as error:
        print(f"CoDefender路径扫描API调用失败: {error}")
        return None


def _scan_by_upload(file_path: Path) -> Optional[Dict[str, Dict[str, Any]]]:
    endpoint = _normalize_endpoint(settings.codefender_predict_endpoint)
    if not endpoint:
        return None

    url = f"{settings.codefender_api_base}{endpoint}"
    try:
        with Path(file_path).open("rb") as handle:
            response = requests.post(url, files={"file": handle}, timeout=settings.codefender_timeout)
        response.raise_for_status()
        return parse_codefender_response(response.json())
    except Exception as error:
        print(f"CoDefender上传预测API调用失败: {error}")
        return None


def _prepare_scan_path(file_path: Path) -> str:
    if settings.host_path_prefix and settings.container_path_prefix:
        return _translate_path_for_container(file_path)
    if settings.docker_container_name:
        copied_path = _copy_file_to_container(file_path)
        if copied_path:
            return copied_path
    return str(Path(file_path).resolve())


def _copy_file_to_container(file_path: Path) -> Optional[str]:
    docker_exe = shutil.which("docker")
    if not docker_exe:
        print("未找到docker命令，无法复制样本到容器")
        return None

    container_name = settings.docker_container_name
    container_dir = settings.docker_container_upload_dir.rstrip("/") or "/codefender_uploads"
    if not _is_safe_container_name(container_name) or not _is_safe_container_path(container_dir):
        print("Docker容器名或容器上传目录配置不安全")
        return None

    container_path = f"{container_dir}/{Path(file_path).name}"
    try:
        subprocess.run(
            [docker_exe, "exec", container_name, "mkdir", "-p", container_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )  # nosec B603
        subprocess.run(
            [docker_exe, "cp", str(Path(file_path).resolve()), f"{container_name}:{container_path}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )  # nosec B603
        return container_path
    except Exception as error:
        print(f"复制样本到Docker容器失败: {error}")
        return None


def _is_safe_container_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value or ""))


def _is_safe_container_path(value: str) -> bool:
    return bool(re.fullmatch(r"/[A-Za-z0-9_./-]{1,240}", value or "")) and ".." not in Path(value).parts


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = (endpoint or "").strip()
    if not endpoint:
        return ""
    return endpoint if endpoint.startswith("/") else f"/{endpoint}"


def _translate_path_for_container(file_path: Path) -> str:
    absolute = Path(file_path).resolve()
    if not settings.host_path_prefix or not settings.container_path_prefix:
        return str(absolute)

    host_prefix = Path(settings.host_path_prefix).resolve()
    try:
        relative = absolute.relative_to(host_prefix)
    except ValueError:
        return str(absolute)
    return os.path.join(settings.container_path_prefix, str(relative)).replace("\\", "/")


def parse_codefender_response(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result_root = data.get("Result") or data.get("result") or data
    model_results = _read_first(
        result_root,
        [
            "ModelResults", "model_results", "models", "Models", "SubModels", "sub_models",
            "model_scores", "ModelScores", "Scores", "scores", "details", "Details",
        ],
    )

    results: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    if model_results is not None:
        for fallback_name, raw_result in _iter_model_items(model_results):
            results[fallback_name] = _format_model_result(raw_result, fallback_name)

    if not results:
        final_raw = result_root if isinstance(result_root, dict) else data
        results["CoDefender集成模型"] = _format_model_result(final_raw, "CoDefender集成模型")

    ensemble = _build_ensemble(result_root if isinstance(result_root, dict) else data, results)
    results["集成结果"] = ensemble
    return results


def _build_ensemble(final_raw: Dict[str, Any], results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    probability = _extract_probability(final_raw)
    result = _extract_result(final_raw, probability)
    virus_name = _extract_virus_name(final_raw)

    if probability is None:
        valid_probs = [item["probability"] for item in results.values() if item.get("probability") is not None]
        if valid_probs:
            probability = sum(valid_probs) / len(valid_probs)
            result = "恶意" if probability > settings.threshold else "安全"

    ensemble = {
        "probability": round(probability, 4) if probability is not None else None,
        "result": result,
    }
    if virus_name:
        ensemble["virus_name"] = virus_name
    return ensemble


def _iter_model_items(model_results: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(model_results, dict):
        for name, value in model_results.items():
            if isinstance(value, dict):
                yield str(name), value
            else:
                yield str(name), {"probability": value}
    elif isinstance(model_results, list):
        for index, value in enumerate(model_results):
            fallback_name = DEFAULT_MODEL_NAMES[index] if index < len(DEFAULT_MODEL_NAMES) else f"Model-{index + 1}"
            if isinstance(value, dict):
                yield _model_name(value, fallback_name), value
            else:
                yield fallback_name, {"probability": value}


def _format_model_result(raw_result: Dict[str, Any], fallback_name: str) -> Dict[str, Any]:
    probability = _extract_probability(raw_result)
    result = _extract_result(raw_result, probability)
    virus_name = _extract_virus_name(raw_result)

    formatted = {
        "probability": round(probability, 4) if probability is not None else None,
        "result": result,
    }
    if virus_name:
        formatted["virus_name"] = virus_name

    raw_name = _read_first(raw_result, ["model", "model_name", "Model", "ModelName", "name"], fallback_name)
    if raw_name:
        formatted["model_name"] = str(raw_name)
    return formatted


def _read_first(data: Any, keys: Iterable[str], default: Any = None) -> Any:
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _model_name(raw_result: Dict[str, Any], fallback_name: str) -> str:
    name = _read_first(raw_result, ["model", "model_name", "Model", "ModelName", "name", "engine_name"], fallback_name)
    return str(name or fallback_name)


def _normalize_probability(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("%"):
            try:
                return max(0.0, min(1.0, float(text[:-1]) / 100))
            except ValueError:
                return None
        try:
            value = float(text)
        except ValueError:
            return None
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if probability > 1:
        probability = probability / 100
    return max(0.0, min(1.0, probability))


def _extract_probability(data: Dict[str, Any]) -> Optional[float]:
    probability = _normalize_probability(
        _read_first(
            data,
            [
                "probability", "malware_probability", "malicious_probability", "malware_prob",
                "malicious_prob", "score", "Score", "confidence", "Confidence", "prob",
            ],
        )
    )
    if probability is not None:
        return probability

    label = str(_read_first(data, ["label", "Label", "result", "Result", "verdict", "Verdict"], "")).lower()
    benign_probability = _normalize_probability(_read_first(data, ["benign_probability", "benign_prob"]))
    if benign_probability is not None and label in ("benign", "safe", "clean", "正常", "安全"):
        return 1 - benign_probability
    return None


def _extract_result(data: Dict[str, Any], probability: Optional[float]) -> str:
    is_malware = _read_first(data, ["is_malware", "isMalware", "IsMalware", "malicious", "is_malicious"])
    if isinstance(is_malware, bool):
        return "恶意" if is_malware else "安全"

    label = str(_read_first(data, ["label", "Label", "result", "Result", "verdict", "Verdict", "Status"], "")).strip().lower()
    if label in ("malware", "malicious", "virus", "detected", "infected", "bad", "恶意"):
        return "恶意"
    if label in ("benign", "safe", "clean", "undetected", "normal", "good", "安全", "正常"):
        return "安全"

    if probability is None:
        return "预测失败"
    return "恶意" if probability > settings.threshold else "安全"


def _extract_virus_name(data: Dict[str, Any]) -> str:
    value = _read_first(
        data,
        [
            "virus_name", "VirusName", "threat_name", "ThreatName", "malware_name",
            "MalwareName", "family", "Family", "name", "Name",
        ],
        "",
    )
    return "" if value is None else str(value)
