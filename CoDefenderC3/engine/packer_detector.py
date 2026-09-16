"""
engine/packer_detector — PE 壳/自解压检测
============================================
使用 pefile 库检测常见壳签名和自解压归档。
"""
import logging

logger = logging.getLogger("codefender.packer")

SECTION_SIGNATURES = {
    "UPX0": "UPX",
    "UPX1": "UPX",
    "UPX2": "UPX",
    ".themida": "Themida",
    ".vmp0": "VMProtect",
    ".vmp1": "VMProtect",
    ".vmp2": "VMProtect",
    ".aspack": "ASPack",
    ".adata": "ASPack",
    "pec2": "PECompact",
    "pec": "PECompact",
    ".enigma1": "Enigma",
    ".enigma2": "Enigma",
    ".mpress1": "MPRESS",
    ".mpress2": "MPRESS",
    ".nsp0": "NSIS",
    ".nsp1": "NSIS",
    ".nsp2": "NSIS",
}

ENTRY_POINT_HEURISTICS = {
    ".vmp0": "VMProtect",
    ".vmp1": "VMProtect",
    ".themida": "Themida",
    "UPX0": "UPX",
    "UPX1": "UPX",
}

SFX_SIGNATURES = [
    (-4, b"NSIS", "NSIS"),
    (-5, b"Inno", "Inno Setup"),
    (0x1C, b"7-Zip", "7-Zip SFX"),
    (0, b"WinRAR SFX", "WinRAR SFX"),
    (0, b"MZ\x90\x00", None),
]


def detect_packer(pe_path: str) -> dict:
    result = {
        "packed": False,
        "packer_name": "",
        "archive": False,
        "archive_name": "",
    }

    try:
        import pefile
    except ImportError:
        logger.debug("pefile 未安装, 跳过壳检测")
        return result

    try:
        pe = pefile.PE(pe_path, fast_load=True)
    except Exception as exc:
        logger.debug(f"pefile 解析失败 {pe_path}: {exc}")
        return result

    try:
        if hasattr(pe, "sections") and pe.sections:
            for section in pe.sections:
                try:
                    sec_name = section.Name.rstrip(b"\x00").decode("ascii", errors="ignore")
                    for sig_name, packer in SECTION_SIGNATURES.items():
                        if sig_name.lower() in sec_name.lower():
                            result["packed"] = True
                            result["packer_name"] = packer
                            break
                    if result["packed"]:
                        break
                except Exception:
                    continue

        if not result["packed"] and hasattr(pe, "sections") and pe.sections:
            try:
                ep = pe.OPTIONAL_HEADER.AddressOfEntryPoint
                for section in pe.sections:
                    if section.VirtualAddress <= ep < section.VirtualAddress + section.Misc_VirtualSize:
                        ep_sec = section.Name.rstrip(b"\x00").decode("ascii", errors="ignore")
                        for sig, packer in ENTRY_POINT_HEURISTICS.items():
                            if sig.lower() in ep_sec.lower():
                                result["packed"] = True
                                result["packer_name"] = packer
                                break
                        if not result["packed"] and ".text" not in ep_sec.lower():
                            result["packed"] = True
                            result["packer_name"] = f"Unknown({ep_sec})"
                        break
            except Exception:
                pass

        try:
            with open(pe_path, "rb") as handle:
                file_size = handle.seek(0, 2)
                for offset, sig_bytes, name in SFX_SIGNATURES:
                    if name is None:
                        continue
                    handle.seek(max(0, file_size + offset) if offset < 0 else offset)
                    data = handle.read(len(sig_bytes))
                    if sig_bytes in data:
                        result["archive"] = True
                        result["archive_name"] = name
                        break

                if not result["archive"] and hasattr(pe, "OVERLAY") and pe.OVERLAY:
                    overlay_data = pe.OVERLAY
                    if b"Nullsoft" in overlay_data[:4096] or b"NSIS" in overlay_data[:4096]:
                        result["archive"] = True
                        result["archive_name"] = "NSIS"
        except Exception:
            pass

    finally:
        try:
            pe.close()
        except Exception:
            pass

    return result
