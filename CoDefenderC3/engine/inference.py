"""
engine/inference — 推理入口
============================
"""
import hashlib
import logging
import os
import struct
import zlib

import numpy as np

logger = logging.getLogger("codefender.engine")


def _is_valid_pe(path: str) -> bool:
    """校验 PE 文件结构完整性（MZ 魔数 + PE 签名）。"""
    try:
        size = os.path.getsize(path)
        if size < 64:
            return False
        with open(path, "rb") as f:
            if f.read(2) != b"MZ":
                return False
            f.seek(0x3C)
            pe_offset = struct.unpack("<I", f.read(4))[0]
            if pe_offset < 4 or pe_offset > min(size - 4, 0x10000):
                return False
            f.seek(pe_offset)
            if f.read(4) != b"PE\x00\x00":
                return False
        return True
    except Exception:
        return False


def compute_identifiers(pe_path: str) -> dict:
    with open(pe_path, "rb") as handle:
        data = handle.read()

    return {
        "MD5": hashlib.md5(data).hexdigest(),
        "SHA256": hashlib.sha256(data).hexdigest(),
        "CRC32": format(zlib.crc32(data) & 0xFFFFFFFF, "08x"),
        "FileName": os.path.basename(pe_path),
        "FileSize": len(data),
    }


def extract_ember_views(pe_path: str) -> dict:
    import config
    from feature_extraction.unified import FeatureType, extract_feature

    tensor = extract_feature(pe_path, FeatureType.EMBER_FULL)
    if hasattr(tensor, "numpy"):
        full_feature = tensor.numpy().reshape(1, -1).astype(np.float32)
    else:
        full_feature = np.asarray(tensor).reshape(1, -1).astype(np.float32)

    views = {}
    view_groups = getattr(config, "VIEW_GROUPS_EMBER", None) or getattr(config, "VIEW_GROUPS", {})
    ember_ranges = getattr(config, "EMBER_RANGES", {})

    for view_name, view_info in view_groups.items():
        ranges = view_info.get("ember_ranges", [])
        parts = []
        for range_name in ranges:
            if range_name not in ember_ranges:
                continue
            lo, hi = ember_ranges[range_name]
            parts.append(full_feature[:, lo:hi])
        if parts:
            views[view_name] = np.hstack(parts).astype(np.float32)

    return views


def scan_single(pe_path: str, ensemble, sdd_engine) -> dict:
    if not _is_valid_pe(pe_path):
        raise ValueError("文件结构不对，或者文件格式损坏")

    from engine.packer_detector import detect_packer
    from engine.result_formatter import format_scan_result

    identifiers = compute_identifiers(pe_path)
    views = extract_ember_views(pe_path)
    result = ensemble.predict_with_evidence(views, sdd_engine)
    pack_info = detect_packer(pe_path)

    return format_scan_result(
        evidence=result,
        identifiers=identifiers,
        pack_info=pack_info,
        model_names=ensemble._model_names,
    )
