import hashlib
import mimetypes
import os
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

from fastapi import UploadFile

from .config import settings


CHUNK_SIZE = 1024 * 1024


def safe_filename(filename: Optional[str]) -> str:
    name = Path(filename or "sample.bin").name
    return name or "sample.bin"


async def save_upload(file: UploadFile) -> Tuple[Path, Dict[str, object]]:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    original_name = safe_filename(file.filename)
    temp_path = settings.upload_dir / f".upload-{os.getpid()}-{uuid.uuid4().hex}"

    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0

    with temp_path.open("wb") as output:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if size > settings.max_upload_size:
                temp_path.unlink(missing_ok=True)
                raise ValueError(f"文件超过大小限制: {settings.max_upload_size} bytes")
            sha256.update(chunk)
            md5.update(chunk)
            output.write(chunk)

    digest = sha256.hexdigest()
    final_path = settings.upload_dir / digest
    if final_path.exists():
        temp_path.unlink(missing_ok=True)
    else:
        temp_path.replace(final_path)

    info = build_file_info(final_path, original_name=original_name, sha256_value=digest, md5_value=md5.hexdigest(), size=size)
    return final_path, info


def build_file_info(
    file_path: Path,
    original_name: Optional[str] = None,
    sha256_value: Optional[str] = None,
    md5_value: Optional[str] = None,
    size: Optional[int] = None,
) -> Dict[str, object]:
    file_path = Path(file_path)
    size = file_path.stat().st_size if size is None else size
    sha256_value, md5_value = _hash_file(file_path, sha256_value, md5_value)
    file_type = detect_file_type(file_path, original_name)
    pe_info = analyze_pe(file_path)

    return {
        "filename": original_name or file_path.name,
        "sha256": sha256_value,
        "md5": md5_value,
        "file_size": size,
        "file_type": file_type,
        "is_pe": pe_info["is_pe"],
        "pe_info": pe_info,
        "query_result": {
            "MD5": md5_value,
            "SHA-256": sha256_value,
            "文件大小": f"{size} bytes",
            "文件类型": file_type,
            "PE结构": "有效" if pe_info["is_pe"] else "无效",
            "架构": pe_info.get("architecture", ""),
            "入口点": pe_info.get("entry_point", ""),
            "镜像基址": pe_info.get("image_base", ""),
            "节数量": pe_info.get("number_of_sections", ""),
            "编译时间": pe_info.get("compile_time", ""),
        },
    }


def _hash_file(file_path: Path, sha256_value: Optional[str], md5_value: Optional[str]) -> Tuple[str, str]:
    if sha256_value and md5_value:
        return sha256_value, md5_value

    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256_value or sha256.hexdigest(), md5_value or md5.hexdigest()


def detect_file_type(file_path: Path, original_name: Optional[str] = None) -> str:
    with file_path.open("rb") as handle:
        head = handle.read(4096)

    if head.startswith(b"MZ"):
        return _detect_pe_type(file_path) or "PE executable"
    if head.startswith(b"%PDF"):
        return "PDF document"
    if head.startswith(b"PK\x03\x04"):
        return "ZIP archive"
    if head.startswith(b"\x7fELF"):
        return "ELF executable"

    guess, _ = mimetypes.guess_type(original_name or file_path.name)
    return guess or "Unknown"


def _detect_pe_type(file_path: Path) -> Optional[str]:
    try:
        with file_path.open("rb") as handle:
            handle.seek(0x3C)
            pe_offset = int.from_bytes(handle.read(4), "little")
            handle.seek(pe_offset)
            if handle.read(4) != b"PE\0\0":
                return None
            machine = int.from_bytes(handle.read(2), "little")
            if machine == 0x8664:
                return "PE32+ executable (x64)"
            if machine == 0x14C:
                return "PE32 executable (x86)"
            return "PE executable"
    except OSError:
        return None


def analyze_pe(file_path: Path) -> Dict[str, Any]:
    try:
        data = Path(file_path).read_bytes()
    except OSError as error:
        return {"is_pe": False, "reason": f"读取文件失败: {error}"}

    if len(data) < 0x40 or data[:2] != b"MZ":
        return {"is_pe": False, "reason": "缺少MZ头"}

    pe_offset = _unpack_from("<I", data, 0x3C)
    if pe_offset is None or pe_offset + 24 > len(data):
        return {"is_pe": False, "reason": "PE头偏移无效"}
    if data[pe_offset:pe_offset + 4] != b"PE\0\0":
        return {"is_pe": False, "reason": "缺少PE签名"}

    file_header_offset = pe_offset + 4
    machine = _unpack_from("<H", data, file_header_offset)
    section_count = _unpack_from("<H", data, file_header_offset + 2)
    timestamp = _unpack_from("<I", data, file_header_offset + 4)
    optional_header_size = _unpack_from("<H", data, file_header_offset + 16)
    characteristics = _unpack_from("<H", data, file_header_offset + 18)
    optional_header_offset = file_header_offset + 20

    if None in (machine, section_count, timestamp, optional_header_size, characteristics):
        return {"is_pe": False, "reason": "COFF文件头不完整"}
    if optional_header_offset + optional_header_size > len(data):
        return {"is_pe": False, "reason": "可选头超出文件范围"}

    magic = _unpack_from("<H", data, optional_header_offset)
    if magic not in (0x10B, 0x20B):
        return {"is_pe": False, "reason": "可选头Magic不是PE32/PE32+"}

    entry_point = _unpack_from("<I", data, optional_header_offset + 16) or 0
    image_base = (
        _unpack_from("<Q", data, optional_header_offset + 24)
        if magic == 0x20B
        else _unpack_from("<I", data, optional_header_offset + 28)
    ) or 0
    subsystem = _unpack_from("<H", data, optional_header_offset + 68) or 0
    size_of_image = _unpack_from("<I", data, optional_header_offset + 56) or 0
    size_of_headers = _unpack_from("<I", data, optional_header_offset + 60) or 0

    sections = _parse_sections(data, optional_header_offset + optional_header_size, section_count)

    return {
        "is_pe": True,
        "format": "PE32+" if magic == 0x20B else "PE32",
        "machine": f"0x{machine:04X}",
        "architecture": _machine_name(machine),
        "number_of_sections": section_count,
        "compile_time": _format_timestamp(timestamp),
        "characteristics": f"0x{characteristics:04X}",
        "entry_point": f"0x{entry_point:X}",
        "image_base": f"0x{image_base:X}",
        "subsystem": _subsystem_name(subsystem),
        "size_of_image": size_of_image,
        "size_of_headers": size_of_headers,
        "sections": sections,
    }


def _parse_sections(data: bytes, section_offset: int, section_count: int) -> List[Dict[str, Any]]:
    sections = []
    for index in range(section_count):
        offset = section_offset + index * 40
        if offset + 40 > len(data):
            break
        raw_name = data[offset:offset + 8].split(b"\0", 1)[0]
        try:
            name = raw_name.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            name = f"section_{index + 1}"
        virtual_size = _unpack_from("<I", data, offset + 8) or 0
        virtual_address = _unpack_from("<I", data, offset + 12) or 0
        raw_size = _unpack_from("<I", data, offset + 16) or 0
        raw_pointer = _unpack_from("<I", data, offset + 20) or 0
        characteristics = _unpack_from("<I", data, offset + 36) or 0
        sections.append({
            "name": name,
            "virtual_address": f"0x{virtual_address:X}",
            "virtual_size": virtual_size,
            "raw_size": raw_size,
            "raw_pointer": f"0x{raw_pointer:X}",
            "characteristics": f"0x{characteristics:08X}",
        })
    return sections


def _unpack_from(fmt: str, data: bytes, offset: int) -> Optional[int]:
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        return None
    return struct.unpack_from(fmt, data, offset)[0]


def _format_timestamp(timestamp: int) -> str:
    if not timestamp:
        return ""
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (OSError, OverflowError, ValueError):
        return ""


def _machine_name(machine: int) -> str:
    names = {
        0x014C: "Intel 386 / x86",
        0x8664: "x64",
        0x01C0: "ARM",
        0x01C4: "ARMv7",
        0xAA64: "ARM64",
        0x0200: "Intel Itanium",
    }
    return names.get(machine, f"Unknown (0x{machine:04X})")


def _subsystem_name(subsystem: int) -> str:
    names = {
        1: "Native",
        2: "Windows GUI",
        3: "Windows Console",
        5: "OS/2 Console",
        7: "POSIX Console",
        9: "Windows CE GUI",
        10: "EFI Application",
        11: "EFI Boot Service Driver",
        12: "EFI Runtime Driver",
        13: "EFI ROM",
        14: "Xbox",
        16: "Windows Boot Application",
    }
    return names.get(subsystem, f"Unknown ({subsystem})")


def stream_hashes(handle: BinaryIO) -> Tuple[str, str, int]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
        size += len(chunk)
        sha256.update(chunk)
        md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest(), size
