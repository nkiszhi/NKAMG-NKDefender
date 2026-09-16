"""
FastAPI service for single-sample inference.

Run:
    uvicorn engine.app:app --host 0.0.0.0 --port 8000

Main endpoint:
    POST /predict-single
    form-data:
      - file: uploaded PE sample
      - include_de: whether to extract D/E (disassembly/graph) features via BinaryNinja
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
import traceback
import uuid
import zlib
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from threading import Lock
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from starlette.responses import JSONResponse

import config
from feature_extraction.extract_feature import BinjaExtractor
from feature_extraction.unified import FeatureType, extract_feature
from models.ensemble import MultiModelEnsemble, _input_dtype
from engine.inference import compute_identifiers, _is_valid_pe
from engine.result_formatter import format_scan_result

# nkrepo 哈希库（SHA256 + MD5 预检）：scaner 目录入 sys.path 后顶层导入。
# 注意目录名用 scaner（非 scanner），避免与 scanner.py 模块同名遮蔽。
_SCANER_DIR = os.path.join(config.ROOT, "scaner")
if _SCANER_DIR not in sys.path:
    sys.path.insert(0, _SCANER_DIR)
import scanner as nkrepo_scanner  # noqa: E402
import staticinfo as nkrepo_staticinfo  # noqa: E402  # fuzzy 哈希计算 (ppdeep/pefile 缺失自动降级)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


RAW_LEN = 32768
INT_VIEWS = {"V1_byte", "V6_opcode_seq"}
FEATURE_DIMS = BinjaExtractor.feature_dims()
MAX_UPLOAD_BYTES = int(os.environ.get("CODEFENDER_MAX_UPLOAD_BYTES", 128 * 1024 * 1024))
MODEL_PATH = os.environ.get("CODEFENDER_MODEL_PATH", config.MODEL_PATH)

# 单样本处理超时（秒），/predict-single 超时返回 504，/predict-directory 超时跳过继续下一个
REQUEST_TIMEOUT = int(os.environ.get("CODEFENDER_REQUEST_TIMEOUT", 120))

MALGRAPH_ENABLED = True
MALGRAPH_BASE_DIR = os.path.join(config.ROOT, "models", "MalGraph_2022")
_MALGRAPH_CKPT_ENV = os.environ.get("CODEFENDER_MALGRAPH_CKPT", "").strip()
_MALGRAPH_VOCAB_ENV = os.environ.get("CODEFENDER_MALGRAPH_VOCAB", "").strip()
_MALGRAPH_MAX_VOCAB_ENV = os.environ.get("CODEFENDER_MALGRAPH_MAX_VOCAB_SIZE", "").strip()
MALGRAPH_PARAMS_FILE = os.environ.get("CODEFENDER_MALGRAPH_PARAMS_FILE", "").strip()
MALGRAPH_PARAMS_JSON = os.environ.get("CODEFENDER_MALGRAPH_PARAMS_JSON", "").strip()
MALGRAPH_DEVICE = os.environ.get("CODEFENDER_MALGRAPH_DEVICE", "cuda").strip() or "cuda"
DEFAULT_MALGRAPH_PARAMS = {
    "gnn_type": "graphsage",
    "pool_type": "global_max_pool",
    "acfg_init_dims": 11,
    "cfg_filters": "128-128-128",
    "fcg_filters": "128-128-128",
    "number_classes": 2,
    "dropout_rate": 0.3,
    "ablation_models": "none",
}


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_INCLUDE_DE = _env_bool("CODEFENDER_INCLUDE_DE", True)
MALGRAPH_ENABLED = _env_bool("CODEFENDER_ENABLE_MALGRAPH", MALGRAPH_ENABLED)

app = FastAPI(
    title="CoDefenderC3 Single-Sample API",
    version="1.0.0",
    description="Upload one PE sample and return per-model prediction results.",
)
# =============================================================================
#  生产加固模块 (items 1-7)
# =============================================================================

# 1. 探针分离：启动时间
_START_TIME = time.time()

# 2. 并发限制信号量
_SCAN_SEMAPHORE = asyncio.Semaphore(
    int(os.environ.get("CODEFENDER_MAX_CONCURRENT", 4))
)

# 3. 请求追踪中间件
@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]
    start_t = time.time()
    response = await call_next(request)
    elapsed_ms = round((time.time() - start_t) * 1000, 1)
    logging.info(
        "[%s] %s %s -> %d  %.1fms",
        req_id, request.method, request.url.path,
        response.status_code, elapsed_ms,
    )
    response.headers["X-Request-ID"] = req_id
    with _METRICS_LOCK:
        _METRICS["total_requests"] += 1
    return response

# 4. 运行时指标
_METRICS: Dict[str, Any] = {
    "total_requests": 0,
    "predict_single_ok": 0,
    "predict_single_error": 0,
    "predict_dir_ok": 0,
    "predict_dir_error": 0,
    "scan_files_total": 0,
    "scan_files_malicious": 0,
    "scan_files_benign": 0,
    "last_predict_ms": 0.0,
    "unhandled_errors": 0,
}
_METRICS_LOCK = Lock()

# 6. 全局异常处理器
@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    with _METRICS_LOCK:
        _METRICS["unhandled_errors"] += 1
    logging.error(
        "[UNHANDLED] %s %s -> %s: %s",
        request.method, request.url.path,
        type(exc).__name__, exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error", "error_type": type(exc).__name__},
    )


_STATE_LOCK = Lock()
_ENSEMBLE: Optional[MultiModelEnsemble] = None
_EXTRACTOR: Optional[BinjaExtractor] = None
_MALGRAPH_RUNTIME: Optional[Dict[str, Any]] = None
_SDD_ENGINE: Optional[Any] = None


def _load_ensemble() -> MultiModelEnsemble:
    global _ENSEMBLE
    if _ENSEMBLE is not None:
        return _ENSEMBLE

    with _STATE_LOCK:
        if _ENSEMBLE is not None:
            return _ENSEMBLE
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"model file not found: {MODEL_PATH}")
        _ENSEMBLE = MultiModelEnsemble.load(MODEL_PATH)
        return _ENSEMBLE


def _get_sdd_engine():
    """Lazy-load SDDEngine (uncalibrated; score_samples returns zero drift evidence)."""
    global _SDD_ENGINE
    if _SDD_ENGINE is not None:
        return _SDD_ENGINE

    with _STATE_LOCK:
        if _SDD_ENGINE is not None:
            return _SDD_ENGINE

        from core.sdd_engine import SDDEngine
        ens = _load_ensemble()
        _SDD_ENGINE = SDDEngine(
            K=ens.K,
            perspective_groups=ens.perspective_groups(),
        )
        return _SDD_ENGINE


def _get_extractor() -> BinjaExtractor:
    global _EXTRACTOR
    if _EXTRACTOR is not None:
        return _EXTRACTOR

    with _STATE_LOCK:
        if _EXTRACTOR is None:
            _EXTRACTOR = BinjaExtractor()
        return _EXTRACTOR


def _load_malgraph_params() -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    if MALGRAPH_PARAMS_FILE:
        if not os.path.exists(MALGRAPH_PARAMS_FILE):
            raise FileNotFoundError(f"MalGraph params file not found: {MALGRAPH_PARAMS_FILE}")
        with open(MALGRAPH_PARAMS_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("MalGraph params file must be a JSON object")
        params.update(loaded)

    if MALGRAPH_PARAMS_JSON:
        loaded = json.loads(MALGRAPH_PARAMS_JSON)
        if not isinstance(loaded, dict):
            raise ValueError("CODEFENDER_MALGRAPH_PARAMS_JSON must be a JSON object")
        params.update(loaded)

    return params


def _resolve_malgraph_ckpt_path() -> str:
    if _MALGRAPH_CKPT_ENV:
        return _MALGRAPH_CKPT_ENV
    candidates = [
        os.path.join(MALGRAPH_BASE_DIR, "best_model.pt"),
        os.path.join(MALGRAPH_BASE_DIR, "saved", "best_model.pt"),
        os.path.join(MALGRAPH_BASE_DIR, "saved", "malgraph_best.pth"),
        os.path.join(MALGRAPH_BASE_DIR, "malgraph_best.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # fallback for health message
    return candidates[0]


def _resolve_malgraph_vocab_path() -> str:
    if _MALGRAPH_VOCAB_ENV:
        return _MALGRAPH_VOCAB_ENV
    candidates = [
        os.path.join(MALGRAPH_BASE_DIR, "saved", "train_external_function_name_vocab.jsonl"),
        os.path.join(MALGRAPH_BASE_DIR, "train_external_function_name_vocab.jsonl"),
        os.path.join(MALGRAPH_BASE_DIR, "output", "train_external_function_name_vocab.jsonl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, Any]:
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                state = ckpt_obj[key]
                break
        else:
            state = ckpt_obj
    else:
        raise ValueError("unsupported MalGraph checkpoint format")

    if not isinstance(state, dict) or not state:
        raise ValueError("empty/invalid MalGraph state dict")

    # DataParallel checkpoints may prefix params with 'module.'
    if any(isinstance(k, str) and k.startswith("module.") for k in state):
        state = {
            (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
            for k, v in state.items()
        }
    return state


def _infer_malgraph_params_from_state(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[int]]:
    params = dict(DEFAULT_MALGRAPH_PARAMS)

    key_has_lin_l = any(".lin_l.weight" in k for k in state.keys() if isinstance(k, str))
    params["gnn_type"] = "graphsage" if key_has_lin_l else "gcn"

    def _infer_filters(prefix: str, gnn_type: str) -> Tuple[Optional[int], List[int]]:
        if gnn_type == "graphsage":
            pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.lin_l\.weight$")
            suffix = ".lin_l.weight"
        else:
            pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.lin\.weight$")
            suffix = ".lin.weight"

        idx_to_key = {}
        for k in state.keys():
            if not isinstance(k, str):
                continue
            m = pat.match(k)
            if m:
                idx_to_key[int(m.group(1))] = k

        if not idx_to_key:
            # fallback: scan by suffix
            for k in state.keys():
                if isinstance(k, str) and k.startswith(prefix + ".") and k.endswith(suffix):
                    parts = k.split(".")
                    if len(parts) >= 4 and parts[1].isdigit():
                        idx_to_key[int(parts[1])] = k

        if not idx_to_key:
            return None, []

        init_dim = None
        outs: List[int] = []
        for i in sorted(idx_to_key.keys()):
            w = state[idx_to_key[i]]
            if not hasattr(w, "shape") or len(getattr(w, "shape", [])) < 2:
                continue
            out_dim = int(w.shape[0])
            in_dim = int(w.shape[1])
            if init_dim is None:
                init_dim = in_dim
            outs.append(out_dim)
        return init_dim, outs

    cfg_init, cfg_outs = _infer_filters("cfg_conv_layers", params["gnn_type"])
    if cfg_init is not None:
        params["acfg_init_dims"] = int(cfg_init)
    if cfg_outs:
        params["cfg_filters"] = "-".join(str(x) for x in cfg_outs)

    _, fcg_outs = _infer_filters("fcg_conv_layers", params["gnn_type"])
    if fcg_outs:
        params["fcg_filters"] = "-".join(str(x) for x in fcg_outs)

    # No parameter in state dict can reveal pooling type reliably; keep default.

    inferred_vocab_size = None
    emb = state.get("external_embedding_layer.weight")
    if hasattr(emb, "shape") and len(getattr(emb, "shape", [])) >= 2:
        inferred_vocab_size = int(emb.shape[0]) - 2

    return params, inferred_vocab_size


def _load_malgraph_runtime() -> Dict[str, Any]:
    global _MALGRAPH_RUNTIME
    if _MALGRAPH_RUNTIME is not None:
        return _MALGRAPH_RUNTIME

    if not MALGRAPH_ENABLED:
        raise RuntimeError("MalGraph disabled by CODEFENDER_ENABLE_MALGRAPH")
    if torch is None:
        raise RuntimeError("torch is not available")
    ckpt_path = _resolve_malgraph_ckpt_path()
    vocab_path = _resolve_malgraph_vocab_path()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"MalGraph checkpoint not found: {ckpt_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"MalGraph vocab file not found: {vocab_path}")

    with _STATE_LOCK:
        if _MALGRAPH_RUNTIME is not None:
            return _MALGRAPH_RUNTIME

        from models.MalGraph_2022.ParameterClasses import ModelParams
        from models.MalGraph_2022.Vocabulary import Vocab
        from models.MalGraph_2022.malgraph import HierarchicalGraphNeuralNetwork

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = _extract_state_dict(ckpt)
        inferred_params, inferred_vocab_size = _infer_malgraph_params_from_state(state)
        user_params = _load_malgraph_params()
        inferred_params.update(user_params)
        params_dict = inferred_params

        if _MALGRAPH_MAX_VOCAB_ENV:
            vocab_max_size = int(_MALGRAPH_MAX_VOCAB_ENV)
        elif inferred_vocab_size is not None and inferred_vocab_size > 0:
            vocab_max_size = inferred_vocab_size
        else:
            vocab_max_size = 10000

        model_params = ModelParams(**params_dict)
        vocab = Vocab(freq_file=vocab_path, max_vocab_size=vocab_max_size)
        logger = logging.getLogger("codefender.malgraph")

        model = HierarchicalGraphNeuralNetwork(
            model_params=model_params,
            external_vocab=vocab,
            global_log=logger,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)

        req_dev = MALGRAPH_DEVICE.lower()
        if req_dev.startswith("cuda") and not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(req_dev)
        model = model.to(device)
        model.eval()

        _MALGRAPH_RUNTIME = {
            "model": model,
            "device": device,
            "params": params_dict,
            "vocab_max_size": vocab_max_size,
            "checkpoint_path": ckpt_path,
            "vocab_path": vocab_path,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
        return _MALGRAPH_RUNTIME


def _predict_malgraph_single(pe_path: str) -> Tuple[int, float, Dict[str, Any]]:
    runtime = _load_malgraph_runtime()

    from torch_geometric.data import Batch, Data
    from models.MalGraph_2022.utils import extract_cfg_and_fcg

    sample = extract_cfg_and_fcg(pe_path, vocab_dict={})
    if not sample:
        raise RuntimeError("MalGraph feature extraction failed")

    acfg_list = sample.get("acfg_list", [])
    if not acfg_list:
        raise RuntimeError("MalGraph no ACFG extracted")

    pyg_graphs = []
    for one_acfg in acfg_list:
        block_features = one_acfg.get("block_features", [])
        block_edges = one_acfg.get("block_edges", [[], []])

        x = torch.tensor(block_features, dtype=torch.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.numel() == 0:
            x = torch.zeros((1, 11), dtype=torch.float32)

        edge_index = torch.tensor(block_edges, dtype=torch.long)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        pyg_graphs.append(Data(x=x, edge_index=edge_index))

    if not pyg_graphs:
        raise RuntimeError("MalGraph no valid function graph")

    local_batch = Batch.from_data_list(pyg_graphs).to(runtime["device"])
    local_count = len(pyg_graphs)

    all_names = sample.get("function_names", [])
    external_names = all_names[local_count:] if len(all_names) >= local_count else []
    function_edges = sample.get("function_edges", [[], []])
    if not isinstance(function_edges, list) or len(function_edges) != 2:
        function_edges = [[], []]

    model = runtime["model"]
    with torch.no_grad():
        out = model(
            real_local_batch=local_batch,
            real_bt_positions=[0, local_count],
            bt_external_names=[external_names],
            bt_all_function_edges=[function_edges],
            local_device=runtime["device"],
        )

    score = float(out.reshape(-1)[0].detach().cpu().item())
    pred = int(score > 0.5)
    meta = {
        "local_functions": local_count,
        "external_functions": len(external_names),
        "fcg_edges": len(function_edges[0]) if function_edges else 0,
        "checkpoint": runtime.get("checkpoint_path", ""),
        "vocab": runtime.get("vocab_path", ""),
    }
    return pred, score, meta


def _to_numpy_1d(x: Any, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(x).reshape(-1)
    return arr.astype(dtype, copy=False)


def _coerce_row_dim(row: np.ndarray, expected_dim: Optional[int]) -> Tuple[np.ndarray, Optional[str]]:
    if expected_dim is None or expected_dim <= 0:
        return row, None
    got = int(row.shape[0])
    if got == expected_dim:
        return row, None
    if got < expected_dim:
        out = np.zeros(expected_dim, dtype=row.dtype)
        out[:got] = row
        return out, f"padded_to_expected_dim:{got}->{expected_dim}"
    return row[:expected_dim], f"truncated_to_expected_dim:{got}->{expected_dim}"


def _expected_dim_for_model(ens: MultiModelEnsemble, model_name: str) -> Optional[int]:
    # noqa: SLF001 - reading ensemble internal attributes
    model = ens._models.get(model_name)
    mtype = ens._model_types.get(model_name, "unknown")

    if model is not None and mtype == "numpy":
        nfi = getattr(model, "n_features_in_", None)
        if nfi is not None:
            try:
                d = int(nfi)
                if d > 0:
                    return d
            except Exception:
                pass

    kwargs = ens._actual_kwargs.get(model_name, {})  # noqa: SLF001
    if isinstance(kwargs, dict):
        in_dim = kwargs.get("input_dim")
        if in_dim is not None:
            try:
                d = int(in_dim)
                if d > 0:
                    return d
            except Exception:
                pass
    return None


def _infer_expected_view_dims(ens: MultiModelEnsemble) -> Dict[str, int]:
    dims: Dict[str, int] = {}

    for vname, model_names in ens.view_groups.items():
        cand: List[int] = []
        for mname in model_names:
            d = _expected_dim_for_model(ens, mname)
            if d is not None and d > 0:
                cand.append(d)
        if cand:
            dims[vname] = Counter(cand).most_common(1)[0][0]
            continue

        if vname == "V1_byte":
            dims[vname] = RAW_LEN
            continue

        cfg = config.VIEW_GROUPS.get(vname, {})
        binja_ids = cfg.get("binja_features", [])
        if binja_ids:
            fallback = 0
            for fid in binja_ids:
                fallback += int(FEATURE_DIMS.get(fid, 0))
            if fallback > 0:
                dims[vname] = fallback

    return dims


def _build_ember_view(
    ember_full: np.ndarray,
    ember_ranges: Dict[str, Tuple[int, int]],
    range_names: List[str],
) -> Optional[np.ndarray]:
    names = list(ember_ranges.keys()) if range_names == ["__ALL__"] else range_names
    parts: List[np.ndarray] = []
    dim = ember_full.shape[0]
    for rn in names:
        if rn not in ember_ranges:
            continue
        lo, hi = ember_ranges[rn]
        if lo < dim:
            parts.append(ember_full[lo:min(hi, dim)])
    if not parts:
        return None
    return np.concatenate(parts).astype(np.float32, copy=False)


def _extract_views_for_single_sample(
    pe_path: str,
    required_views: List[str],
    include_de: bool,
    expected_view_dims: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    ext = _get_extractor()
    feats = {}
    view_meta: Dict[str, Dict[str, Any]] = {}

    try:
        feats.update(ext.extract_fast(pe_path))
    except Exception as e:
        raise RuntimeError(f"extract_fast failed: {e}") from e

    de_status = "skipped"
    if include_de:
        try:
            feats.update(ext.extract_de_only(pe_path))
            de_status = "ok"
        except Exception as e:
            de_status = f"failed: {e}"

    with open(pe_path, "rb") as f:
        raw_bytes = f.read(RAW_LEN)

    ember_full: Optional[np.ndarray] = None
    ember_ranges: Optional[Dict[str, Tuple[int, int]]] = None

    def ensure_ember_full() -> np.ndarray:
        nonlocal ember_full, ember_ranges
        if ember_full is None:
            tensor = extract_feature(pe_path, FeatureType.EMBER_FULL)
            if hasattr(tensor, "detach"):
                ember_full = tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            else:
                ember_full = np.asarray(tensor, dtype=np.float32).reshape(-1)
            ember_ranges = config.get_ember_ranges(len(ember_full))
        return ember_full

    views: Dict[str, np.ndarray] = {}

    for vn in required_views:
        cfg = config.VIEW_GROUPS.get(vn, {})
        binja_ids = list(cfg.get("binja_features", []))
        expected_dim = (expected_view_dims or {}).get(vn)
        row: Optional[np.ndarray] = None
        source = ""
        reason = ""

        if vn == "V1_byte":
            row_i = np.zeros(RAW_LEN, dtype=np.int32)
            if "B01" in feats:
                src = _to_numpy_1d(feats["B01"], np.int32)[:RAW_LEN]
                row_i[: len(src)] = src
            else:
                raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.int32, copy=False)
                row_i[: len(raw_arr)] = raw_arr
            row = row_i
            source = "binja:B01/raw_fallback"
        elif binja_ids:
            available_ids = [fid for fid in binja_ids if fid in feats]
            missing_ids = [fid for fid in binja_ids if fid not in feats]
            if available_ids:
                dtype = np.int32 if vn in INT_VIEWS else np.float32
                parts = [_to_numpy_1d(feats[fid], dtype) for fid in available_ids]
                row = np.concatenate(parts)
                source = f"binja:{','.join(available_ids)}"
                if missing_ids:
                    reason = f"partial_binja_features_missing:{','.join(missing_ids)}"
            else:
                reason = f"missing_binja_features:{','.join(missing_ids)}"
                if (not include_de) and any(fid.startswith(("D", "E")) for fid in missing_ids):
                    reason += "; hint=include_de=true"

        if row is None and "ember_ranges" in cfg:
            try:
                full = ensure_ember_full()
                ranges = ember_ranges or config.get_ember_ranges(len(full))
                range_names = cfg.get("ember_ranges", [])
                ember_row = _build_ember_view(full, ranges, range_names)
                if ember_row is not None:
                    # For mixed views (both binja+ember defined), only use ember
                    # fallback when model-expected dim matches the ember slice dim.
                    if binja_ids and expected_dim is not None and ember_row.shape[0] != expected_dim:
                        if not reason:
                            reason = (
                                "ember_fallback_dim_mismatch:"
                                f"{ember_row.shape[0]}!=expected_{expected_dim}"
                            )
                    else:
                        row = ember_row
                        source = f"ember:{','.join(range_names)}"
                elif not reason:
                    reason = "empty_ember_slice"
            except Exception as e:
                if not reason:
                    reason = f"ember_extract_failed:{e}"

        if row is None and not reason:
            reason = "no_feature_path_for_view"

        if row is not None:
            row, dim_note = _coerce_row_dim(row.reshape(-1), expected_dim)
            if dim_note:
                reason = f"{reason}; {dim_note}" if reason else dim_note
            views[vn] = row.reshape(1, -1)
            meta = {
                "status": "ok",
                "source": source,
                "shape": list(views[vn].shape),
            }
            if expected_dim is not None:
                meta["expected_dim"] = expected_dim
            if reason:
                meta["note"] = reason
            view_meta[vn] = meta
        else:
            view_meta[vn] = {
                "status": "missing",
                "reason": reason,
            }

    view_meta["_de_feature_extraction"] = {"status": de_status}
    return views, view_meta


def _extract_single_sample_wrapper(args):
    """
    Module-level wrapper function for ProcessPoolExecutor to extract features for a single sample.
    This must be at module level (not nested) for pickling by ProcessPoolExecutor.
    """
    idx, pe_path, required_views, include_de, expected_view_dims = args
    try:
        views, view_meta = _extract_views_for_single_sample(
            pe_path=pe_path,
            required_views=required_views,
            include_de=include_de,
            expected_view_dims=expected_view_dims,
        )
        return idx, views, view_meta, None
    except Exception as e:
        # Return error information for failed samples
        return idx, None, None, str(e)


def _extract_views_for_batch(
    pe_paths: List[str],
    required_views: List[str],
    include_de: bool,
    expected_view_dims: Optional[Dict[str, int]] = None,
) -> Dict[int, Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]]:
    """
    Extract features from multiple PE samples in parallel using ProcessPoolExecutor.
    
    Args:
        pe_paths: List of PE file paths to process
        required_views: List of view names to extract
        include_de: Whether to extract D/E features (BinaryNinja disassembly/graph)
        expected_view_dims: Optional dict mapping view names to expected dimensions
    
    Returns:
        Dict mapping sample indices to tuples of (views_dict, view_meta_dict).
        Failed samples are excluded from the result.
    """
    if not pe_paths:
        return {}
    
    # Determine optimal worker count
    max_workers = min(len(pe_paths), os.cpu_count() or 32)
    
    results: Dict[int, Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]] = {}
    
    # Use ProcessPoolExecutor for parallel feature extraction
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create tasks with indices and all required parameters
        tasks = [(i, path, required_views, include_de, expected_view_dims) 
                 for i, path in enumerate(pe_paths)]
        
        # Execute parallel extraction
        for idx, views, view_meta, error in executor.map(_extract_single_sample_wrapper, tasks):
            if error is None:
                # Successful extraction
                results[idx] = (views, view_meta)
            else:
                # Failed extraction - log but continue with other samples
                logging.warning(
                    "[batch_feature_extraction] Sample %d (%s) failed: %s",
                    idx,
                    pe_paths[idx] if idx < len(pe_paths) else "unknown",
                    error,
                )
    
    return results


def _predict_per_model(
    ens: MultiModelEnsemble,
    views: Dict[str, np.ndarray],
    view_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    pe_path: Optional[str] = None,
    include_de: bool = True,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for vname, model_names in ens.view_groups.items():
        x_view = views.get(vname)
        for mname in model_names:
            model = ens._models.get(mname)  # noqa: SLF001 - existing internal structure
            mtype = ens._model_types.get(mname, "unknown")  # noqa: SLF001

            item: Dict[str, Any] = {
                "model": mname,
                "view": vname,
                "engine": mtype,
            }

            if mtype == "special":
                if mname == "MalGraph":
                    if not include_de:
                        item["status"] = "skipped"
                        item["reason"] = "MalGraph requires D/E features; set include_de=true"
                        results.append(item)
                        continue
                    if not pe_path:
                        item["status"] = "error"
                        item["reason"] = "missing pe_path for MalGraph inference"
                        results.append(item)
                        continue
                    try:
                        pred, score, mg_meta = _predict_malgraph_single(pe_path)
                        item["status"] = "ok"
                        item["score"] = score
                        item["prediction"] = pred
                        item["malgraph_meta"] = mg_meta
                    except Exception as e:
                        item["status"] = "unavailable"
                        item["reason"] = str(e)
                    results.append(item)
                    continue
                item["status"] = "skipped"
                item["reason"] = "special_pipeline_not_supported"
                results.append(item)
                continue

            if x_view is None:
                item["status"] = "skipped"
                meta = (view_meta or {}).get(vname, {})
                miss_reason = meta.get("reason", "missing_view_feature")
                item["reason"] = f"missing_view_feature:{miss_reason}"
                results.append(item)
                continue

            if model is None:
                item["status"] = "unavailable"
                item["reason"] = "model_not_loaded"
                results.append(item)
                continue

            x_model = x_view
            expected_model_dim = _expected_dim_for_model(ens, mname)
            if expected_model_dim is not None:
                adj_row, dim_note = _coerce_row_dim(
                    np.asarray(x_view).reshape(-1),
                    expected_model_dim,
                )
                x_model = adj_row.reshape(1, -1)
                if dim_note:
                    item["input_adjust"] = dim_note

            try:
                if mtype == "numpy":
                    x_np = np.asarray(x_model, dtype=np.float32)
                    proba = model.predict_proba(x_np)
                    score = float(proba[0, 1])
                    pred = int(score > 0.5)
                elif mtype == "pytorch":
                    if torch is None:
                        raise RuntimeError("torch is not available")
                    x_dtype = torch.long if _input_dtype(mname) == "long" else torch.float32
                    x_t = torch.tensor(x_model, dtype=x_dtype)

                    try:
                        device = next(model.parameters()).device
                    except StopIteration:
                        device = torch.device("cpu")

                    model.eval()
                    with torch.no_grad():
                        out = model(x_t.to(device))
                        if isinstance(out, tuple):
                            out = out[0]
                        if out.shape[-1] == 1:
                            score = float(out.squeeze(-1).detach().cpu().numpy()[0])
                        else:
                            score = float(torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()[0])
                        pred = int(score > 0.5)
                else:
                    raise RuntimeError(f"unsupported engine: {mtype}")

                item["status"] = "ok"
                item["score"] = score
                item["prediction"] = pred
            except Exception as e:
                item["status"] = "error"
                item["reason"] = str(e)

            results.append(item)

    return results



# =============================================================================
#  5. MD5 校验工具函数
# =============================================================================

def _file_md5(path: str) -> str:
    """计算文件 MD5 哈希，用于模型完整性校验。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
#  7. 临时文件安全清理
# =============================================================================

@contextmanager
def _tmp_upload(suffix: str = ".bin"):
    """安全临时文件上下文管理器，确保异常路径下也能清理。"""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# =============================================================================
#  nkrepo SHA256 + MD5 哈希库预检（命中直出 Malicious，跳过模型）
#  + fuzzy 模糊哈希库第三级预检（ssdeep / imphash / authentihash）
# =============================================================================

_sha256_db = None
_md5_db = None
_fuzzy_db = None


def _init_hash_dbs():
    """启动时初始化双哈希库（失败自动降级为纯模型模式，功能不回退）。

    降级粒度：sha256 库挂 → 双库全降；仅 md5 库挂 → 保留 sha256 预检。
    """
    global _sha256_db, _md5_db
    if not config.HASH_DB_ENABLED:
        logging.getLogger("codefender.engine").info("[hashdb] 哈希库预检已禁用 (CODEFENDER_HASH_DB_ENABLED=0)")
        return
    sig_dir = config.HASH_DB_DIR
    _log = logging.getLogger("codefender.engine")
    try:
        _sha256_db = nkrepo_scanner.HashSignatureDB(
            os.path.join(sig_dir, "sha256.db"),
            shard_count=config.HASH_DB_SHARDS,
            bloom_fp_rate=config.HASH_DB_FP_RATE,
            max_open_shards=config.HASH_DB_MAX_OPEN,
            layout=config.HASH_DB_LAYOUT,
        )
        _log.info("[hashdb] SHA256 签名库就绪: %s 条", f"{_sha256_db.count:,}")
    except Exception as e:
        _sha256_db = None
        _log.warning("[hashdb] SHA256 库不可用，双库降级为纯模型检测: %s", e)

    if _sha256_db is not None and os.path.isdir(os.path.join(sig_dir, "md5.db.shards")):
        try:
            _md5_db = nkrepo_scanner.HashSignatureDB(
                os.path.join(sig_dir, "md5.db"),
                shard_count=config.HASH_DB_SHARDS,
                bloom_fp_rate=config.HASH_DB_FP_RATE,
                max_open_shards=config.HASH_DB_MAX_OPEN,
                layout=config.HASH_DB_LAYOUT,
                hash_algo="md5",
            )
            _log.info("[hashdb] MD5 签名库就绪: %s 条", f"{_md5_db.count:,}")
        except Exception as e:
            _md5_db = None
            _log.warning("[hashdb] MD5 库不可用，降级（仅 SHA256 预检）: %s", e)

    # fuzzy 模糊哈希库 (5 表 256 分片): 独立降级, 挂了不影响 sha256/md5 预检
    if config.FUZZY_DB_ENABLED and os.path.isdir(os.path.join(sig_dir, "fuzzy.db.shards")):
        try:
            _fuzzy_db = nkrepo_scanner.FuzzySignatureDB(
                os.path.join(sig_dir, "fuzzy.db"),
                max_open_shards=config.FUZZY_DB_MAX_OPEN,
                layout="hex",
            )
            _log.info("[hashdb] Fuzzy 模糊哈希库就绪: %s 条 %s",
                      f"{_fuzzy_db.count:,}", _fuzzy_db._counts)
        except Exception as e:
            _fuzzy_db = None
            _log.warning("[hashdb] Fuzzy 库不可用，降级（仅双哈希预检）: %s", e)
    elif not config.FUZZY_DB_ENABLED:
        _log.info("[hashdb] Fuzzy 预检已禁用 (CODEFENDER_FUZZY_DB_ENABLED=0)")


_init_hash_dbs()


def _lookup_hashes(md5_hex: str, sha256_hex: str, file_size: int):
    """双库查询：先 SHA256（更抗碰撞、覆盖广）后 MD5（补漏老样本哈希）。

    Returns:
        (hit_dict, source) 命中时；否则 (None, None)。source ∈ {"sha256_db", "md5_db"}
    """
    if _sha256_db is None and _md5_db is None:
        return None, None
    hits, source = None, None
    if _sha256_db is not None:
        hits = _sha256_db.check_hash(sha256_hex, file_size=file_size)
        source = "sha256_db"
    if not hits and _md5_db is not None:
        hits = _md5_db.check_hash(md5_hex, file_size=file_size)
        source = "md5_db"
    if not hits:
        return None, None
    return hits[0], source


def _compute_fuzzy_hashes(content: bytes) -> Dict[str, Optional[str]]:
    """计算预检用的 fuzzy 哈希（单项失败/缺依赖自动降级为 None）。

    - ssdeep: ppdeep 纯 Python, 仅对 ≤ FUZZY_DB_MAX_BYTES 的文件计算 (默认 256KB,
      上游实测 1MB≈24s, 预检为同步路径必须限流; 超限跳过——截断数据算出的
      ssdeep 与库中全文件签名不匹配, 无意义)
    - imphash / authentihash: 仅 PE 样本, pefile 解析毫秒级, 不限大小
    """
    out: Dict[str, Optional[str]] = {"ssdeep": None, "imphash": None, "authentihash": None}
    try:
        if len(content) <= config.FUZZY_DB_MAX_BYTES:
            out["ssdeep"] = nkrepo_staticinfo.compute_ssdeep(content)
    except Exception:
        pass
    try:
        out["imphash"] = nkrepo_staticinfo.compute_imphash(content)
    except Exception:
        pass
    try:
        out["authentihash"] = nkrepo_staticinfo.compute_authentihash(content)
    except Exception:
        pass
    return out


def _lookup_fuzzy(content: bytes, file_size: int):
    """第三级预检：fuzzy 模糊哈希库（ssdeep / imphash / authentihash 按值查 5 表）。

    带 file_size 做大小比对——imphash/vhash 类哈希对常见壳/运行时存在碰撞
    （如 PyInstaller 打包的良性文件），大小校验可显著抑制误报。

    Returns:
        (hit_dict, "fuzzy_db") 命中时；否则 (None, None)。
    """
    if _fuzzy_db is None:
        return None, None
    fz = _compute_fuzzy_hashes(content)
    if not any(fz.values()):
        return None, None
    try:
        hits = _fuzzy_db.check_by_computed_hashes(
            ssdeep=fz.get("ssdeep"),
            imphash_hex=fz.get("imphash"),
            authentihash_hex=fz.get("authentihash"),
            file_size=file_size,
        )
    except Exception as e:
        logging.getLogger("codefender.engine").warning("[hashdb] fuzzy 查询异常(忽略): %s", e)
        return None, None
    if not hits:
        return None, None
    return hits[0], "fuzzy_db"


def _build_hash_hit_result(md5_hex: str, sha256_hex: str, crc32: str,
                           file_name: str, file_size: int, hit: dict, source: str) -> Dict[str, Any]:
    """组装哈希命中结果：结构与 format_scan_result 完全同构，仅扩展 Source / HashMatch。"""
    return {
        "MD5": md5_hex, "SHA256": sha256_hex, "CRC32": crc32,
        "FileName": file_name, "FileSize": file_size,
        "Status": 1, "StatusLabel": "Malicious", "IsMalicious": True, "Score": 1.0,
        "Source": source,
        "ScanResult": {
            "VirusName": hit.get("name") or "KnownMalware",  # v4 库不存名称, 命中统一显示通用标签
            "Packed": {"IsPacked": False, "PackerName": "", "IsArchive": False, "ArchiveName": ""},
            "PerModelScores": {}, "PerModelPreds": {},
            "DriftEvidence": {"surface": 0.0, "structure": 0.0, "global": 0.0},
            "SampleDriftType": "normal", "AnomalyScore": 0.0, "IsDriftCandidate": False,
            # HashMatch 不暴露 name/size —— 双哈希库 v4 起不存检出名称 (统一 KnownMalware);
            # fuzzy 库虽存 name/size 但命中即恶意, 大小无比对价值, 口径与双哈希库一致。
            # fuzzy 命中额外带 fuzzy_type/sha256 (关联的原始样本哈希) 透出。
            "HashMatch": {k: v for k, v in hit.items() if k not in ("name", "size")},
        },
    }


def _hash_precheck(content: bytes, file_name: str, file_size: int) -> Optional[Dict[str, Any]]:
    """内存三级预检（/predict-single 用）：sha256 → md5 → fuzzy，任一命中返回结果 dict，全部未命中返回 None。"""
    if _sha256_db is None and _md5_db is None and _fuzzy_db is None:
        return None
    md5_hex, _sha1_hex, sha256_hex = nkrepo_scanner.compute_hashes_bytes(content)
    crc32 = format(zlib.crc32(content) & 0xFFFFFFFF, "08x")
    hit, source = _lookup_hashes(md5_hex, sha256_hex, file_size)
    if hit is None:
        hit, source = _lookup_fuzzy(content, file_size)
    if hit is None:
        return None
    return _build_hash_hit_result(md5_hex, sha256_hex, crc32, file_name, file_size, hit, source)


def _hash_precheck_file(path: str) -> Optional[Dict[str, Any]]:
    """流式三级哈希预检（/predict-directory 用，1MB 分块避免整文件载入内存）。

    fuzzy 级需要文件内容算哈希: 未命中双哈希库后才读文件（绝大多数干净文件
    走不到这一步），读入上限 64MB 防御异常大文件。
    """
    if _sha256_db is None and _md5_db is None and _fuzzy_db is None:
        return None
    md5_h, sha256_h = hashlib.md5(), hashlib.sha256()
    size = 0
    crc = 0
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            md5_h.update(chunk)
            sha256_h.update(chunk)
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
    md5_hex, sha256_hex = md5_h.hexdigest(), sha256_h.hexdigest()
    crc32 = format(crc & 0xFFFFFFFF, "08x")
    hit, source = _lookup_hashes(md5_hex, sha256_hex, size)
    if hit is None and _fuzzy_db is not None and size <= 64 * 1024 * 1024:
        try:
            with open(path, "rb") as f:
                content = f.read()
            hit, source = _lookup_fuzzy(content, size)
        except OSError:
            hit, source = None, None
    if hit is None:
        return None
    return _build_hash_hit_result(md5_hex, sha256_hex, crc32, os.path.basename(path), size, hit, source)


# =============================================================================
#  Endpoints: 探针 / 指标
# =============================================================================

# 1. 存活探针（不碰模型）
@app.get("/livez")
def livez() -> Dict[str, Any]:
    """存活探针：只检查进程是否存活，不接触模型。"""
    uptime_sec = int(time.time() - _START_TIME)
    return {
        "alive": True,
        "uptime_sec": uptime_sec,
        "uptime_human": datetime.utcfromtimestamp(uptime_sec).strftime("%H:%M:%S"),
    }


# 1. 就绪探针（检查模型是否已加载）
@app.get("/readyz")
def readyz():
    """就绪探针：检查模型是否已加载并可服务。"""
    if _ENSEMBLE is None:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": "model not loaded",
                "model_path": MODEL_PATH,
            },
        )
    return {
        "ready": True,
        "model_count": _ENSEMBLE.K,
        "view_count": _ENSEMBLE.V,
    }


# 4. 运行时指标端点
@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """Prometheus 兼容的运行时指标端点。"""
    with _METRICS_LOCK:
        m = dict(_METRICS)
    m["uptime_sec"] = int(time.time() - _START_TIME)
    m["models_loaded"] = bool(_ENSEMBLE)
    if _ENSEMBLE:
        m["model_count"] = _ENSEMBLE.K
    return m


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        ens = _load_ensemble()
        # 5. 模型 MD5 完整性校验
        model_md5 = ""
        if os.path.exists(MODEL_PATH):
            try:
                model_md5 = _file_md5(MODEL_PATH)
            except OSError:
                model_md5 = "read_error"

        base = {
            "status": "ok",
            "model_path": MODEL_PATH,
            "model_md5": model_md5,
            "model_count": ens.K,
            "view_count": ens.V,
            "views": list(ens.view_groups.keys()),
            "malgraph": {
                "enabled": MALGRAPH_ENABLED,
                "checkpoint": _resolve_malgraph_ckpt_path(),
                "vocab": _resolve_malgraph_vocab_path(),
            },
        }
        if MALGRAPH_ENABLED:
            try:
                rt = _load_malgraph_runtime()
                base["malgraph"]["status"] = "ready"
                base["malgraph"]["device"] = str(rt["device"])
                base["malgraph"]["params"] = rt.get("params", {})
                base["malgraph"]["vocab_max_size"] = rt.get("vocab_max_size")
                base["malgraph"]["missing_keys"] = rt.get("missing_keys", [])
                base["malgraph"]["unexpected_keys"] = rt.get("unexpected_keys", [])
            except Exception as e:
                base["malgraph"]["status"] = "not_ready"
                base["malgraph"]["reason"] = str(e)
        return base
    except Exception as e:
        return {
            "status": "error",
            "model_path": MODEL_PATH,
            "reason": str(e),
        }


# ──────────────────────────────────────────────────────────────────
# 病毒库信息查询
# ──────────────────────────────────────────────────────────────────

@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    """返回当前病毒库/集成模型的完整档案信息。"""
    result: Dict[str, Any] = {}
    try:
        st = os.stat(MODEL_PATH)
        model_md5 = ""
        try:
            model_md5 = _file_md5(MODEL_PATH)
        except OSError:
            model_md5 = "read_error"
        result["file"] = {
            "path": MODEL_PATH,
            "size_bytes": st.st_size,
            "size_mb": round(st.st_size / (1024 * 1024), 2),
            "last_modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
            "md5": model_md5,
        }
    except OSError:
        result["file"] = {"path": MODEL_PATH, "status": "not_found"}
    if _ENSEMBLE is None:
        result["status"] = "not_loaded"
        return result
    ens = _ENSEMBLE
    result["status"] = "loaded"
    try:
        result["ensemble"] = {"model_count": ens.K, "view_count": ens.V}
        models: List[Dict[str, Any]] = []
        for mname in ens._model_names:
            mtype = ens._model_types.get(mname, "unknown")
            model_obj = ens._models.get(mname)
            models.append({"name": mname, "type": mtype, "loaded": model_obj is not None})
        result["models"] = models
        views: Dict[str, Any] = {}
        for vname, mnames in ens.view_groups.items():
            vg_cfg = config.VIEW_GROUPS.get(vname, {})
            views[vname] = {
                "label": vg_cfg.get("label", vname),
                "attck": vg_cfg.get("attck", ""),
                "models": mnames,
                "indices": ens._view_indices.get(vname, []),
            }
        result["views"] = views
        result["sdd"] = {"initialized": _SDD_ENGINE is not None}
        result["malgraph"] = {"enabled": MALGRAPH_ENABLED}
        if MALGRAPH_ENABLED:
            result["malgraph"]["checkpoint"] = _resolve_malgraph_ckpt_path()
            result["malgraph"]["vocab"] = _resolve_malgraph_vocab_path()
            result["malgraph"]["loaded"] = _MALGRAPH_RUNTIME is not None
    except Exception:
        logging.error("model-info build failed:\n%s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build model info: {traceback.format_exc()}",
        )
    return result


# ──────────────────────────────────────────────────────────────────
# 病毒库热更新
# ──────────────────────────────────────────────────────────────────

@app.post("/reload-model")
def reload_model(
    model_path: Optional[str] = Query(
        default=None,
        description="新模型文件绝对路径。不传则重新加载当前 MODEL_PATH。",
    ),
) -> Dict[str, Any]:
    """热更新集成模型。不传参则原地重载；传参则切换到新路径。"""
    global _ENSEMBLE, _SDD_ENGINE, _MALGRAPH_RUNTIME, MODEL_PATH
    target_path = model_path or MODEL_PATH
    if not os.path.isfile(target_path):
        raise HTTPException(400, detail=f"model file not found: {target_path}")
    old_ens = _ENSEMBLE
    old_info: Dict[str, Any] = {"model_path": MODEL_PATH}
    if old_ens is not None:
        old_info["model_count"] = old_ens.K
        old_info["view_count"] = old_ens.V
        try:
            old_info["file_size"] = os.path.getsize(MODEL_PATH)
        except OSError:
            old_info["file_size"] = 0
    else:
        old_info["model_count"] = 0
        old_info["view_count"] = 0
        old_info["file_size"] = 0
    try:
        new_ens = MultiModelEnsemble.load(target_path)
    except Exception:
        logging.error('reload-model 加载新模型失败:\n%s', traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {traceback.format_exc()}",
        )
    with _STATE_LOCK:
        _ENSEMBLE = new_ens
        _SDD_ENGINE = None
        _MALGRAPH_RUNTIME = None
        if model_path:
            os.environ["CODEFENDER_MODEL_PATH"] = target_path
            MODEL_PATH = target_path
    new_info: Dict[str, Any] = {
        "model_path": target_path,
        "model_count": new_ens.K,
        "view_count": new_ens.V,
        "models": [
            {"name": m, "type": new_ens._model_types.get(m, "?")}
            for m in new_ens._model_names
        ],
    }
    try:
        new_info["file_size"] = os.path.getsize(target_path)
    except OSError:
        new_info["file_size"] = 0
    return {"status": "reloaded", "previous": old_info, "current": new_info}


# ──────────────────────────────────────────────────────────────────
# 单文件推理核心函数（供 /predict-single 和 /predict-directory 共用）
# ──────────────────────────────────────────────────────────────────

def _scan_single_pe(
    pe_path: str,
    file_name: str,
    file_size: int,
    ens: MultiModelEnsemble,
    expected_view_dims: Dict[str, int],
    include_de: bool = True,
) -> Dict[str, Any]:
    """对单个 PE 文件执行完整推理链路，返回 format_scan_result 格式的 dict。"""
    # PE文件格式验证
    if not _is_valid_pe(pe_path):
        raise ValueError("文件结构不对，或者文件格式损坏")

    views, view_meta = _extract_views_for_single_sample(
        pe_path=pe_path,
        required_views=list(ens.view_groups.keys()),
        include_de=include_de,
        expected_view_dims=expected_view_dims,
    )

    per_model = _predict_per_model(
        ens, views, view_meta=view_meta, pe_path=pe_path, include_de=include_de,
    )
    # ── Build (1, K) matrices from flat per_model list ──
    ok_results = [r for r in per_model if r.get("status") == "ok"]
    K = len(ok_results)
    model_names = [r["model"] for r in ok_results]
    scores_arr = np.array(
        [r["score"] for r in ok_results], dtype=np.float32,
    ).reshape(1, K) if K > 0 else np.zeros((1, 1), dtype=np.float32)
    preds_arr = np.array(
        [r["prediction"] for r in ok_results], dtype=np.int64,
    ).reshape(1, K) if K > 0 else np.zeros((1, 1), dtype=int)

    # ── Ensemble prediction (weighted average, aligns with run_all.py) ──
    ens_model_set = set(ens._model_names)
    ens_ok = [r for r in ok_results if r["model"] in ens_model_set]
    if ens_ok:
        ok_scores = np.array([r["score"] for r in ens_ok], dtype=np.float32)
        ens_score_arr = np.array([ok_scores.mean()], dtype=np.float32)
        ens_pred_arr = np.array([int(ens_score_arr[0] > 0.5)], dtype=int)
    else:
        ens_pred_arr = np.array([0], dtype=int)
        ens_score_arr = np.array([0.0], dtype=np.float32)

    # ── File identifiers + packer detection ──
    identifiers = compute_identifiers(pe_path)
    identifiers["FileName"] = file_name
    identifiers["FileSize"] = file_size
    from engine.packer_detector import detect_packer
    pack_info = detect_packer(pe_path)

    # ── SDD drift evidence ──
    try:
        sdd = _get_sdd_engine()
        drift_ev = sdd.score_samples(scores_arr)
        drift_evidence = drift_ev.get("drift_evidence", np.zeros(3))
        sample_drift_type = drift_ev.get("sample_drift_type", "normal")
        anomaly_score = drift_ev.get("anomaly_score", 0.0)
        is_drift_candidate = drift_ev.get("is_drift_candidate", False)
    except Exception:
        drift_evidence = np.zeros(3)
        sample_drift_type = "normal"
        anomaly_score = 0.0
        is_drift_candidate = False

    # ── Model voting stats ──
    total_models = len(per_model)
    ok_count = len(ok_results)
    skipped_count = sum(1 for r in per_model if r.get("status") == "skipped")
    error_count = total_models - ok_count - skipped_count
    malicious_votes = sum(1 for r in ok_results if r.get("prediction") == 1)
    benign_votes = ok_count - malicious_votes
    agreement_rate = malicious_votes / ok_count if ok_count > 0 else 0.0
    ok_scores = [r["score"] for r in ok_results if "score" in r]

    model_stats = {
        "TotalModels": total_models,
        "OkModels": ok_count,
        "SkippedModels": skipped_count,
        "ErrorModels": error_count,
        "MaliciousVotes": malicious_votes,
        "BenignVotes": benign_votes,
        "AgreementRate": round(agreement_rate, 4),
        "EnsembleScore": round(float(ens_score_arr[0]), 6) if K > 0 else 0.0,
        "EnsemblePred": int(ens_pred_arr[0]) if K > 0 else 0,
        "ScoreRange": [round(min(ok_scores), 4), round(max(ok_scores), 4)] if ok_scores else [0.0, 0.0],
    }

    # ── Build evidence dict for format_scan_result ──
    evidence = {
        "predictions": ens_pred_arr,
        "scores": ens_score_arr,
        "per_model_scores": scores_arr,
        "per_model_preds": preds_arr,
        "drift_evidence": drift_evidence,
        "sample_drift_type": sample_drift_type,
        "anomaly_score": anomaly_score,
        "is_drift_candidate": is_drift_candidate,
        "model_stats": model_stats,
    }

    return format_scan_result(
        evidence=evidence,
        identifiers=identifiers,
        pack_info=pack_info,
        model_names=model_names,
    )

# 7. 同步包装器：使用 _tmp_upload 安全创建临时文件
def _scan_single_pe_wrapper(
    content: bytes,
    suffix: str,
    file_name: str,
    file_size: int,
    ens: MultiModelEnsemble,
    expected_view_dims: Dict[str, int],
    include_de: bool,
) -> Dict[str, Any]:
    """同步包装器：使用 _tmp_upload 安全创建临时文件后调用 _scan_single_pe。"""
    with _tmp_upload(suffix=suffix) as temp_path:
        with open(temp_path, "wb") as f:
            f.write(content)
        return _scan_single_pe(
            pe_path=temp_path,
            file_name=file_name,
            file_size=file_size,
            ens=ens,
            expected_view_dims=expected_view_dims,
            include_de=include_de,
        )


@app.post("/predict-single")
async def predict_single(
    file: UploadFile = File(...),
    include_de: bool = Query(
        default=DEFAULT_INCLUDE_DE,
        description="Whether to run D/E feature extraction (BinaryNinja disassembly/graph).",
    ),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="empty filename")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file content")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file too large: {len(content)} bytes > {MAX_UPLOAD_BYTES} bytes",
        )

    # ── SHA256 + MD5 哈希库预检：命中直接返回 Malicious，跳过模型（毫秒级短路）──
    precheck = _hash_precheck(content, file.filename, len(content))
    if precheck is not None:
        return precheck

    ens = _load_ensemble()
    expected_view_dims = _infer_expected_view_dims(ens)
    suffix = os.path.splitext(file.filename)[1] or ".bin"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(content)
            temp_path = tf.name

        # 超时控制：120s 未完成则放弃，返回 504
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    _scan_single_pe,
                    temp_path,
                    file.filename,
                    len(content),
                    ens,
                    expected_view_dims,
                    include_de,
                ),
                timeout=REQUEST_TIMEOUT,
            )
            return result
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"scan timeout after {REQUEST_TIMEOUT}s",
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────
# 目录批量扫描
# ──────────────────────────────────────────────────────────────────

_PE_EXTENSIONS = {".exe", ".dll", ".sys", ".ocx", ".scr", ".bat", ".cmd",
                   ".msi", ".ps1", ".vbs", ".js", ".wsf", ".hta", ".cpl",
                   ".com", ".src", ".pif", ".drv", ".bin"}
MAX_BATCH_FILES = int(os.environ.get("CODEFENDER_MAX_BATCH_FILES", 500))


def _build_batch_views(
    sample_views: Dict[int, Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]],
    view_names: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[int, Dict[str, Dict[str, Any]]]]:
    """
    Build batch view matrices from per-sample views.
    
    Args:
        sample_views: Dict mapping sample indices to (views_dict, view_meta_dict)
        view_names: List of view names to build batch matrices for
    
    Returns:
        Tuple of (batch_views, per_sample_meta) where:
        - batch_views: Dict mapping view names to (N, dim) numpy arrays
        - per_sample_meta: Dict mapping sample indices to their view metadata
    """
    if not sample_views:
        return {}, {}
    
    batch_views: Dict[str, np.ndarray] = {}
    per_sample_meta: Dict[int, Dict[str, Dict[str, Any]]] = {}
    
    # Get sorted sample indices
    sorted_indices = sorted(sample_views.keys())
    
    # Extract metadata for all samples (do once, not per view)
    for idx in sorted_indices:
        _, view_meta = sample_views[idx]
        per_sample_meta[idx] = view_meta
    
    # First pass: determine dimension for each view from available samples
    view_dims: Dict[str, int] = {}
    for vname in view_names:
        for idx in sorted_indices:
            views, _ = sample_views[idx]
            if vname in views:
                view_dims[vname] = views[vname].reshape(-1).shape[0]
                break
        # Fallback if no sample has this view
        if vname not in view_dims:
            view_dims[vname] = RAW_LEN if vname == "V1_byte" else 2381
    
    # Second pass: build batch matrix for each view
    for vname in view_names:
        view_rows: List[np.ndarray] = []
        expected_dim = view_dims[vname]
        
        for idx in sorted_indices:
            views, _ = sample_views[idx]
            
            if vname in views:
                # View exists for this sample
                row = views[vname].reshape(-1)
                # Ensure consistent dimension
                if row.shape[0] != expected_dim:
                    if row.shape[0] < expected_dim:
                        # Pad
                        padded = np.zeros(expected_dim, dtype=np.float32)
                        padded[:row.shape[0]] = row
                        view_rows.append(padded)
                    else:
                        # Truncate
                        view_rows.append(row[:expected_dim])
                else:
                    view_rows.append(row)
            else:
                # View missing - use zero-filled placeholder
                view_rows.append(np.zeros(expected_dim, dtype=np.float32))
        
        if view_rows:
            # Stack into (N, dim) batch matrix
            batch_views[vname] = np.vstack(view_rows).astype(np.float32)
    
    return batch_views, per_sample_meta


def _format_batch_results(
    pe_paths: List[str],
    batch_predictions: np.ndarray,
    batch_scores: np.ndarray,
    per_model_predictions: np.ndarray,
    per_model_scores: np.ndarray,
    model_names: List[str],
    per_sample_meta: Dict[int, Dict[str, Dict[str, Any]]],
    valid_indices: List[int],
) -> List[Dict[str, Any]]:
    """
    Format batch inference results into per-sample result dictionaries.
    
    Args:
        pe_paths: List of all PE file paths
        batch_predictions: (N,) or (N, 1) ensemble predictions
        batch_scores: (N,) or (N, 1) ensemble scores
        per_model_predictions: (N, K) per-model predictions
        per_model_scores: (N, K) per-model scores
        model_names: List of model names (length K)
        per_sample_meta: Dict mapping sample indices to view metadata
        valid_indices: List of sample indices that were successfully processed
    
    Returns:
        List of formatted result dictionaries
    """
    results: List[Dict[str, Any]] = []
    
    # Ensure predictions and scores are at least 1D
    batch_predictions = np.atleast_1d(batch_predictions).flatten()
    batch_scores = np.atleast_1d(batch_scores).flatten()
    
    for batch_idx, sample_idx in enumerate(valid_indices):
        fpath = pe_paths[sample_idx]
        fname = os.path.basename(fpath)
        
        try:
            file_size = os.path.getsize(fpath)
            
            # Extract this sample's predictions and scores (handle 1D arrays)
            ens_pred = int(batch_predictions[batch_idx])
            ens_score = float(batch_scores[batch_idx])
            sample_per_model_preds = per_model_predictions[batch_idx, :].reshape(1, -1)
            sample_per_model_scores = per_model_scores[batch_idx, :].reshape(1, -1)
            
            # Get identifiers and packer info
            identifiers = compute_identifiers(fpath)
            identifiers["FileName"] = fname
            identifiers["FileSize"] = file_size
            
            from engine.packer_detector import detect_packer
            pack_info = detect_packer(fpath)
            
            # SDD drift evidence (placeholder for batch - would need batch SDD implementation)
            drift_evidence = np.zeros(3)
            sample_drift_type = "normal"
            anomaly_score = 0.0
            is_drift_candidate = False
            
            # Model voting stats
            total_models = len(model_names)
            malicious_votes = int(np.sum(sample_per_model_preds > 0.5))
            benign_votes = total_models - malicious_votes
            agreement_rate = malicious_votes / total_models if total_models > 0 else 0.0
            
            model_stats = {
                "TotalModels": total_models,
                "OkModels": total_models,
                "SkippedModels": 0,
                "ErrorModels": 0,
                "MaliciousVotes": malicious_votes,
                "BenignVotes": benign_votes,
                "AgreementRate": round(agreement_rate, 4),
                "EnsembleScore": round(ens_score, 6),
                "EnsemblePred": ens_pred,
                "ScoreRange": [
                    round(float(np.min(sample_per_model_scores)), 4),
                    round(float(np.max(sample_per_model_scores)), 4)
                ],
            }
            
            # Build evidence dict
            evidence = {
                "predictions": np.array([[ens_pred]]),
                "scores": np.array([[ens_score]]),
                "per_model_scores": sample_per_model_scores,
                "per_model_preds": sample_per_model_preds,
                "drift_evidence": drift_evidence,
                "sample_drift_type": sample_drift_type,
                "anomaly_score": anomaly_score,
                "is_drift_candidate": is_drift_candidate,
                "model_stats": model_stats,
            }
            
            result = format_scan_result(
                evidence=evidence,
                identifiers=identifiers,
                pack_info=pack_info,
                model_names=model_names,
            )
            result["FilePath"] = fpath
            results.append(result)
            
        except Exception as e:
            results.append({
                "FileName": fname,
                "FilePath": fpath,
                "Status": -1,
                "StatusLabel": "Error",
                "IsMalicious": False,
                "Error": f"format_result_failed: {e}",
            })
    
    return results


async def _batch_scan_samples(
    pe_paths: List[str],
    ens: MultiModelEnsemble,
    expected_view_dims: Dict[str, int],
    include_de: bool = False,
) -> List[Dict[str, Any]]:
    """
    Batch process multiple PE samples with parallel feature extraction and batch inference.
    
    Args:
        pe_paths: List of PE file paths to process
        ens: MultiModelEnsemble instance
        expected_view_dims: Dict mapping view names to expected dimensions
        include_de: Whether to extract D/E features (BinaryNinja disassembly/graph)
    
    Returns:
        List of formatted result dictionaries (one per sample)
    """
    loop = asyncio.get_event_loop()
    
    # Step 1: Parallel feature extraction
    sample_views = await loop.run_in_executor(
        None,
        _extract_views_for_batch,
        pe_paths,
        list(ens.view_groups.keys()),
        include_de,
        expected_view_dims,
    )
    
    # Handle completely failed samples
    valid_indices = sorted(sample_views.keys())
    if not valid_indices:
        # All samples failed - return error results
        return [{
            "FileName": os.path.basename(fpath),
            "FilePath": fpath,
            "Status": -1,
            "StatusLabel": "Error",
            "IsMalicious": False,
            "Error": "feature_extraction_failed",
        } for fpath in pe_paths]
    
    # Step 2: Build batch view matrices
    batch_views, per_sample_meta = _build_batch_views(
        sample_views,
        list(ens.view_groups.keys()),
    )
    
    try:
        per_model_predictions, per_model_scores = await loop.run_in_executor(
            None,
            ens.predict_all,
            batch_views,
        )
        
        # Step 4: Ensemble prediction (仅有效模型参与, 缺失模型不计入分母)
        valid_k = [k for k, mname in enumerate(ens._model_names) if ens._models.get(mname) is not None]
        if valid_k:
            valid_scores = per_model_scores[:, valid_k]       # (N, K_valid)
            batch_scores = valid_scores.mean(axis=1)           # 等权平均
            batch_predictions = (batch_scores > 0.5).astype(int)
        else:
            N_batch = per_model_scores.shape[0]
            batch_predictions = np.zeros(N_batch, dtype=int)
            batch_scores = np.full(N_batch, 0.5, dtype=np.float32)

        # Step 5: Format results
        results = _format_batch_results(
            pe_paths=pe_paths,
            batch_predictions=batch_predictions,
            batch_scores=batch_scores,
            per_model_predictions=per_model_predictions,
            per_model_scores=per_model_scores,
            model_names=ens._model_names,
            per_sample_meta=per_sample_meta,
            valid_indices=valid_indices,
        )
        
        # Add back failed samples
        all_results: List[Dict[str, Any]] = []
        result_idx = 0
        for i, fpath in enumerate(pe_paths):
            if i in valid_indices:
                all_results.append(results[result_idx])
                result_idx += 1
            else:
                all_results.append({
                    "FileName": os.path.basename(fpath),
                    "FilePath": fpath,
                    "Status": -1,
                    "StatusLabel": "Error",
                    "IsMalicious": False,
                    "Error": "feature_extraction_failed",
                })
        
        return all_results
        
    except Exception as e:
        # Batch inference failed - return errors for all samples
        logging.error("Batch inference failed: %s", e)
        return [{
            "FileName": os.path.basename(fpath),
            "FilePath": fpath,
            "Status": -1,
            "StatusLabel": "Error",
            "IsMalicious": False,
            "Error": f"batch_inference_failed: {e}",
        } for fpath in pe_paths]


@app.post("/predict-directory")
async def predict_directory(
    directory: str = Query(..., description="服务器端目录绝对路径"),
    recursive: bool = Query(default=True, description="是否递归扫描子目录"),
    include_de: bool = Query(
        default=False,
        description="批量模式下默认关闭D/E特征提取以加速",
    ),
) -> Dict[str, Any]:
    """批量扫描目录下所有 PE 样本，返回逐文件检测结果。"""
    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail=f"directory not found: {directory}")

    # 收集所有文件（不再依赖扩展名过滤）
    all_files: List[str] = []
    walk_root = directory
    for dirpath, _dirnames, filenames in os.walk(walk_root):
        for fn in filenames:
            all_files.append(os.path.join(dirpath, fn))
        if not recursive:
            break  # 仅扫描顶层目录

    if not all_files:
        raise HTTPException(status_code=404, detail="no files found in directory")

    # 通过读取文件头来判断是否为PE文件（支持无扩展名或非标准扩展名的PE文件）
    pe_files: List[str] = []
    skipped_non_pe: int = 0
    for fpath in all_files:
        if _is_valid_pe(fpath):
            pe_files.append(fpath)
        else:
            skipped_non_pe += 1

    if not pe_files:
        raise HTTPException(
            status_code=404, 
            detail=f"no PE files found in directory (scanned {len(all_files)} files, {skipped_non_pe} non-PE)"
        )

    if len(pe_files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"too many files: {len(pe_files)} > {MAX_BATCH_FILES}",
        )

    # ── SHA256 + MD5 哈希库预检：命中直出 Malicious（不占模型批量额度）──
    hash_hits: List[Dict[str, Any]] = []
    remaining_files: List[str] = []
    for fpath in pe_files:
        hit = _hash_precheck_file(fpath)
        if hit is not None:
            hit["FilePath"] = fpath
            hash_hits.append(hit)
        else:
            remaining_files.append(fpath)

    # 2. 并发信号量 + 4. 指标埋点
    start_t = time.time()
    try:
        async with _SCAN_SEMAPHORE:
            # 模型加载阶段计时（全部命中时跳过模型加载）
            model_load_start = time.time()
            if remaining_files:
                ens = _load_ensemble()
                expected_view_dims = _infer_expected_view_dims(ens)
            else:
                ens = None
                expected_view_dims = {}
            model_load_time = time.time() - model_load_start

            # 批量扫描阶段计时 - 仅未命中样本进模型批量（可能为空）
            batch_scan_start = time.time()
            if remaining_files:
                results = await _batch_scan_samples(remaining_files, ens, expected_view_dims, include_de)
            else:
                results = []
            batch_scan_time = time.time() - batch_scan_start

            # 合并结果：哈希命中在前，模型结果在后
            results = hash_hits + results
            
            # Calculate per-sample average time and add to results
            successful_count = sum(1 for r in results if r.get("Status") != -1)
            if successful_count > 0:
                avg_per_sample = batch_scan_time / successful_count
                for r in results:
                    if r.get("Status") != -1:
                        r["ProcessingTime"] = round(avg_per_sample, 3)

        # 统计结果
        error_count = sum(1 for r in results if r.get("Status") == -1 and r.get("StatusLabel") == "Error")
        timeout_count = sum(1 for r in results if r.get("Status") == -1 and r.get("StatusLabel") == "Timeout")
        malicious_count = sum(1 for r in results if r.get("IsMalicious") and r.get("Status") != -1)
        benign_count = sum(1 for r in results if not r.get("IsMalicious") and r.get("Status") != -1)
        
        # 单样本处理时间统计
        sample_times = [r.get("ProcessingTime", 0.0) for r in results if r.get("ProcessingTime") and r.get("Status") != -1]
        min_sample_time = min(sample_times) if sample_times else 0.0
        max_sample_time = max(sample_times) if sample_times else 0.0
        avg_sample_time = sum(sample_times) / len(sample_times) if sample_times else 0.0
        sorted_times = sorted(sample_times)
        median_sample_time = sorted_times[len(sorted_times) // 2] if sorted_times else 0.0

        # 总体时间统计
        total_elapsed = time.time() - start_t
        elapsed_ms = round(total_elapsed * 1000, 1)
        
        with _METRICS_LOCK:
            _METRICS["predict_dir_ok"] += 1
            _METRICS["scan_files_total"] += len(pe_files)
            _METRICS["scan_files_malicious"] += malicious_count
            _METRICS["scan_files_benign"] += benign_count
            _METRICS["last_predict_ms"] = elapsed_ms

        summary = {
            "TotalFiles": len(pe_files),
            "TotalFilesInDirectory": len(all_files),  # 目录中所有文件数
            "SkippedNonPE": skipped_non_pe,           # 跳过的非PE文件数
            "ScannedFiles": len(results) - error_count - timeout_count,
            "MaliciousCount": malicious_count,
            "BenignCount": benign_count,
            "ErrorCount": error_count,
            "TimeoutCount": timeout_count,
            "Directory": directory,
            # 新增详细时间统计
            "Timing": {
                "TotalTime": round(total_elapsed, 3),          # 总耗时(秒)
                "ModelLoadTime": round(model_load_time, 3),    # 模型加载时间(秒)
                "BatchScanTime": round(batch_scan_time, 3),    # 批量扫描时间(秒)
                "AvgSampleTime": round(avg_sample_time, 3),    # 平均每样本时间(秒)
                "MedianSampleTime": round(median_sample_time, 3),  # 中位数样本时间(秒)
                "MinSampleTime": round(min_sample_time, 3),
                "MaxSampleTime": round(max_sample_time, 3),
                "Throughput": round(len(pe_files) / total_elapsed, 3) if total_elapsed > 0 else 0.0,  # 样本/秒
            },
        }

        return {
            "Summary": summary,
            "Results": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        with _METRICS_LOCK:
            _METRICS["predict_dir_error"] += 1
        raise HTTPException(status_code=500, detail=str(e)) from e
