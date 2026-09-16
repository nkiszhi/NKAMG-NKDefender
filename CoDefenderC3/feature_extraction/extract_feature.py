"""
BinaryNinja Feature Extractor — All 48 Features (§4.2.1, Appendix C)
======================================================================
Replaces LIEF + pefile + capstone + IDA/Ghidra with a single BinaryNinja API.

Dimensions:
  A. PE Structure  (A01-A15):  264 + imports + exports + security = ~600d
  B. Byte Stats    (B01-B07):  raw bytes + 256+256+256+8+65536+6 = ~66k
  C. Strings       (C01-C04):  variable + 10+30+256 = ~300d
  D. Disassembly   (D01-D07):  4096+256+256+15+12+10+6144 = ~10k
  E. Graph         (E01-E04):  graph structs + 12+10 = ~22d stats
  F. Visualization (F01-F04):  256²×8 = ~524k (images)
  G. Hashes        (G01-G04):  4×128 = 512d
  H. Metadata      (H01-H03):  ~60d

Usage:
    from feature_extractor import BinjaExtractor
    ext = BinjaExtractor()
    features = ext.extract(pe_path)   # returns dict of numpy arrays
    features = ext.extract_batch(paths, n_workers=8)

Requires: pip install binaryninja (commercial license)
"""
import os, sys, hashlib, math, struct, time
from collections import Counter, defaultdict
from typing import Dict, Optional
import numpy as np

# BinaryNinja import (deferred to allow module-level import without crash)
_BN = None

def _env_int(name: str, default: int) -> int:
    """Helper to read integer from environment variable."""
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def _get_bn():
    global _BN
    if _BN is None:
        import binaryninja as bn
        _BN = bn
        
        # 设置BinaryNinja工作线程数（提升分析性能）
        # 默认2线程，可通过环境变量 BN_WORKER_THREADS 调整
        # 建议：4-8 线程（取决于CPU核心数）
        try:
            worker_threads = max(1, _env_int("BN_WORKER_THREADS", 8))
            bn.set_worker_thread_count(worker_threads)
        except Exception:
            pass
        
        # 抑制 BinaryNinja 分析日志 (warn/info), 只保留严重错误
        try:
            bn.log.log_to_stderr(bn.LogLevel.ErrorLog)
        except Exception:
            pass
        try:
            bn.disable_default_log()
        except Exception:
            pass
    return _BN


# ═══════════════════════════════════════
#  Constants
# ═══════════════════════════════════════
MAX_BYTES     = 2 * 1024 * 1024   # 2MB max for raw byte sequence (B01)
MAX_SECTIONS  = 48                # zero-pad sections to this count (A04)
MAX_OPCODES   = 4096              # opcode sequence length (D01)
MAX_FUNCTIONS = 48                # functions for Asm embedding (D07)
IMPORT_BUCKETS = 256              # import hash bucket count (A06)
IMG_SIZE       = 256              # visualization image size (F01-F04)
NGRAM_DIM      = 256              # n-gram TF-IDF output dim (D02-D03)
ENTROPY_BINS   = 256              # entropy histogram bins (B04)
BLOCK_SIZE     = 1024             # entropy block size

# x86 instruction categories (15 classes, D04)
INSN_CATEGORIES = {
    "arithmetic": {"add","sub","mul","imul","div","idiv","inc","dec","neg","adc","sbb"},
    "logic":      {"and","or","xor","not","shl","shr","sar","sal","rol","ror","rcl","rcr"},
    "transfer":   {"mov","movzx","movsx","lea","xchg","cmov","movaps","movdqa"},
    "branch":     {"jmp","je","jne","jz","jnz","jg","jl","jge","jle","ja","jb","jae","jbe",
                   "call","ret","loop","int"},
    "stack":      {"push","pop","pusha","popa","pushf","popf","enter","leave"},
    "string":     {"movs","cmps","scas","lods","stos","rep","repe","repne"},
    "flag":       {"stc","clc","cmc","std","cld","sti","cli","lahf","sahf"},
    "float":      {"fld","fst","fstp","fadd","fsub","fmul","fdiv","fcom","fxch"},
    "simd":       {"addps","mulps","subps","divps","paddb","paddw","paddd","pxor",
                   "xmm","ymm","zmm"},
    "system":     {"syscall","sysenter","rdtsc","cpuid","rdmsr","wrmsr","hlt","in","out"},
    "nop":        {"nop"},
    "interrupt":  {"int","int3","iret"},
    "io":         {"in","out","ins","outs"},
    "prefix":     {"lock","rep","repz","repnz","cs","ds","es","fs","gs","ss"},
}
# Flatten for lookup
_CAT_LOOKUP = {}
for cat, mnemonics in INSN_CATEGORIES.items():
    for m in mnemonics:
        _CAT_LOOKUP[m] = cat
_CAT_NAMES = list(INSN_CATEGORIES.keys()) + ["other"]  # 15 categories


# ═══════════════════════════════════════
#  Main Extractor
# ═══════════════════════════════════════

class BinjaExtractor:
    """
    Extract all 48 features from a PE binary using BinaryNinja.
    
    两阶段架构:
      阶段1: PE → .bndb (BinaryNinja 分析, ~60s, 只做一次)
      阶段2: .bndb → features (秒开, 可反复重跑)
    
    BinaryNinja replaces:
      - LIEF/pefile  → sections, symbols
      - capstone     → disassembly_text
      - IDA/Ghidra   → functions, basic_blocks, callgraph
    """

    def __init__(self, headless=True, require_bn=False):
        # 延迟加载: 不在构造时 import binaryninja
        # Master 进程只需要 generate_bndbs_batch / extract_de_from_bndb_batch
        # 这些方法通过子进程使用 BN, 不需要 Master 自身加载 BN C 核心
        # 在 Master 中加载 BN 会导致 Windows 上 ACCESS_VIOLATION (0xC0000005)
        self.bn = None
        self._headless = headless
        self._require_bn = require_bn

    def _ensure_bn(self):
        """按需加载 BinaryNinja — 只在真正需要直接调用 BN API 时使用。"""
        if self.bn is None:
            try:
                self.bn = _get_bn()
            except Exception:
                if self._require_bn:
                    raise

    # ─── 快速提取: 纯 Python, 不需要 BinaryNinja ──────
    #
    # 覆盖 11/16 视角, 24/33 模型:
    #   B01-B07 (V1_byte, V2_byte_stat), F01-F04 (V8_*),
    #   A01-A08 (V3_pe_struct, V4_import), C01-C04 (V5_string),
    #   G01-G04 (V9_hash), H01-H03 (V10_metadata)
    #
    # 不覆盖 (需要 BinaryNinja 反汇编):
    #   D01-D07 (V6_opcode_seq, V6_opcode_stat, V6_func_embed)
    #   E01-E04 (V7_graph)

    def extract_fast(self, pe_path: str) -> Dict[str, np.ndarray]:
        """纯 Python 特征提取 (不需要 BinaryNinja), 毫秒级。"""
        with open(pe_path, "rb") as f:
            raw = f.read()
        # 解析 pefile 一次, 所有需要的特征共用
        pe_obj = None
        try:
            import pefile
            pe_obj = pefile.PE(data=raw, fast_load=True)
            pe_obj.parse_data_directories(directories=[
                pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'],
                pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT'],
            ])
        except Exception:
            pass
        features = {}
        for stage_fn in [
            lambda: self._extract_pe_structure_fast(raw, pe_obj),
            lambda: self._extract_byte_stats(raw),
            lambda: self._extract_strings_fast(raw),
            lambda: self._extract_visualization(raw),
            lambda: self._extract_hashes_fast(raw, pe_obj),
            lambda: self._extract_metadata_fast(raw),
        ]:
            try:
                features.update(stage_fn())
            except Exception:
                pass
        if pe_obj:
            try: pe_obj.close()
            except: pass
        return features

    def _extract_pe_structure_fast(self, raw, pe_obj=None):
        """A01-A08: PE 结构特征, 纯 struct 解析 (不需要 BinaryNinja)。"""
        feats = {}

        # A01: DOS Header (15d)
        dos = np.zeros(15, dtype=np.float32)
        if len(raw) >= 64:
            dos[0] = struct.unpack_from("<H", raw, 0)[0]
            dos[1] = struct.unpack_from("<I", raw, 60)[0] if len(raw) > 63 else 0
            dos[2] = len(raw)
            for i in range(3, min(15, 30)):
                off = i * 2
                if off + 2 <= 64:
                    dos[i] = struct.unpack_from("<H", raw, off)[0]
        feats["A01"] = dos

        # A02: COFF Header (7d)
        coff = np.zeros(7, dtype=np.float32)
        pe_off = struct.unpack_from("<I", raw, 0x3C)[0] if len(raw) > 0x3F else 0
        if pe_off + 24 <= len(raw) and raw[pe_off:pe_off+4] == b"PE\x00\x00":
            hdr = raw[pe_off+4:pe_off+24]
            if len(hdr) >= 20:
                coff[0] = struct.unpack_from("<H", hdr, 0)[0]   # Machine
                coff[1] = struct.unpack_from("<H", hdr, 2)[0]   # NumberOfSections
                coff[2] = struct.unpack_from("<I", hdr, 4)[0]   # TimeDateStamp
                coff[3] = struct.unpack_from("<I", hdr, 8)[0]   # PointerToSymbolTable
                coff[4] = struct.unpack_from("<I", hdr, 12)[0]  # NumberOfSymbols
                coff[5] = struct.unpack_from("<H", hdr, 16)[0]  # SizeOfOptionalHeader
                coff[6] = struct.unpack_from("<H", hdr, 18)[0]  # Characteristics
        feats["A02"] = coff

        # A03: Optional Header (30d)
        opt = np.zeros(30, dtype=np.float32)
        opt_off = pe_off + 24
        if opt_off + 96 <= len(raw):
            opt[0] = struct.unpack_from("<H", raw, opt_off)[0]  # Magic
            if opt[0] == 0x20B:  # PE32+
                if opt_off + 112 <= len(raw):
                    opt[1] = struct.unpack_from("<Q", raw, opt_off + 24)[0]  # ImageBase
                    opt[2] = struct.unpack_from("<I", raw, opt_off + 32)[0]  # SectionAlignment
                    opt[3] = struct.unpack_from("<I", raw, opt_off + 36)[0]  # FileAlignment
                    opt[4] = struct.unpack_from("<I", raw, opt_off + 56)[0]  # SizeOfImage
                    opt[5] = struct.unpack_from("<I", raw, opt_off + 60)[0]  # SizeOfHeaders
                    opt[6] = struct.unpack_from("<Q", raw, opt_off + 64)[0]  # SizeOfStackReserve
                    opt[7] = struct.unpack_from("<I", raw, opt_off + 84)[0]  # NumberOfRvaAndSizes
                    opt[8] = struct.unpack_from("<I", raw, opt_off + 16)[0]  # AddressOfEntryPoint
            else:  # PE32
                opt[1] = struct.unpack_from("<I", raw, opt_off + 28)[0]  # ImageBase
                opt[2] = struct.unpack_from("<I", raw, opt_off + 32)[0]
                opt[3] = struct.unpack_from("<I", raw, opt_off + 36)[0]
                opt[4] = struct.unpack_from("<I", raw, opt_off + 56)[0]
                opt[5] = struct.unpack_from("<I", raw, opt_off + 60)[0]
                opt[6] = struct.unpack_from("<I", raw, opt_off + 72)[0]
                opt[7] = struct.unpack_from("<I", raw, opt_off + 76)[0]
                opt[8] = struct.unpack_from("<I", raw, opt_off + 16)[0]
        feats["A03"] = opt

        # A04: Section Features (MAX_SECTIONS × 4 = 192d)
        n_sec = int(coff[1])
        opt_hdr_size = int(coff[5])
        sec_off = pe_off + 24 + opt_hdr_size
        sec_feats = np.zeros(MAX_SECTIONS * 4, dtype=np.float32)
        for i in range(min(n_sec, MAX_SECTIONS)):
            s = sec_off + i * 40
            if s + 40 > len(raw):
                break
            base = i * 4
            sec_feats[base] = struct.unpack_from("<I", raw, s + 8)[0]   # VirtualSize
            sec_feats[base+1] = struct.unpack_from("<I", raw, s + 16)[0]  # SizeOfRawData
            raw_ptr = struct.unpack_from("<I", raw, s + 20)[0]
            raw_sz = struct.unpack_from("<I", raw, s + 16)[0]
            if raw_ptr < len(raw) and raw_sz > 0:
                sec_data = raw[raw_ptr:raw_ptr + min(raw_sz, 65536)]
                sec_feats[base+2] = self._entropy(sec_data)
            sec_feats[base+3] = struct.unpack_from("<I", raw, s + 36)[0]  # Characteristics
        feats["A04"] = sec_feats

        # A05: Section Anomaly (20d)
        anom = np.zeros(20, dtype=np.float32)
        for i in range(min(n_sec, 10)):
            s = sec_off + i * 40
            if s + 40 > len(raw):
                break
            vsize = struct.unpack_from("<I", raw, s + 8)[0]
            anom[i*2] = 1.0 if vsize == 0 else 0.0
            anom[i*2+1] = 1.0 if vsize > 10*1024*1024 else 0.0
        feats["A05"] = anom

        # A06: Import Table (256d hash-bucketed)
        imp = np.zeros(IMPORT_BUCKETS, dtype=np.float32)
        try:
            if pe_obj and hasattr(pe_obj, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe_obj.DIRECTORY_ENTRY_IMPORT:
                    for func in entry.imports:
                        name = (func.name or b"").decode("ascii", errors="ignore")
                        if name:
                            imp[hash(name) % IMPORT_BUCKETS] += 1
                if imp.sum() > 0:
                    imp /= imp.sum()
        except Exception:
            pass
        feats["A06"] = imp

        # A07: Export Table (32d)
        exp = np.zeros(32, dtype=np.float32)
        try:
            if pe_obj and hasattr(pe_obj, 'DIRECTORY_ENTRY_EXPORT'):
                exp[0] = len(pe_obj.DIRECTORY_ENTRY_EXPORT.symbols)
                exp[1] = sum(1 for s in pe_obj.DIRECTORY_ENTRY_EXPORT.symbols if s.name)
        except Exception:
            pass
        feats["A07"] = exp

        # A08_A15: Security metadata (51d)
        sec_meta = np.zeros(51, dtype=np.float32)
        sec_meta[0] = 1.0 if b"Rich" in raw[:1024] else 0.0
        sec_meta[1] = float(len(raw))
        feats["A08_A15"] = sec_meta

        return feats

    def _extract_strings_fast(self, raw):
        """C01-C04: 字符串特征, 纯 regex (不需要 BinaryNinja)。"""
        import re
        feats = {}
        strings = [s.decode("ascii", errors="ignore")
                   for s in re.findall(rb'[\x20-\x7e]{4,}', raw[:2*1024*1024])]

        # C01: String embedding (256d hash trick)
        c01 = np.zeros(256, dtype=np.float32)
        for s in strings[:2000]:
            c01[hash(s) % 256] += 1
        if c01.max() > 0:
            c01 /= c01.max() + 1e-8
        feats["C01"] = c01

        # C02: String statistics (10d)
        c02 = np.zeros(10, dtype=np.float32)
        c02[0] = len(strings)
        if strings:
            lens = [len(s) for s in strings]
            c02[1] = np.mean(lens)
            c02[2] = np.max(lens)
            c02[3] = np.std(lens)
            c02[4] = sum(1 for s in strings if "http" in s.lower() or "www" in s.lower())
            c02[5] = sum(1 for s in strings if s.startswith("http")) / max(len(strings), 1)
            c02[6] = sum(1 for s in strings if "\\" in s or "/" in s) / max(len(strings), 1)
            c02[7] = sum(1 for s in strings if ".dll" in s.lower()) / max(len(strings), 1)
            c02[8] = sum(1 for s in strings if ".exe" in s.lower()) / max(len(strings), 1)
            c02[9] = sum(1 for s in strings if any(c.isdigit() for c in s)) / max(len(strings), 1)
        feats["C02"] = c02

        # C03: Sensitive pattern matching (31d)
        patterns = [
            "cmd", "powershell", "reg", "net ", "http", "ftp",
            "encrypt", "decrypt", "password", "admin", "root",
            "kernel32", "ntdll", "ws2_32", "advapi32", "shell32",
            "create", "write", "delete", "execute", "inject",
            "mutex", "pipe", "socket", "connect", "download",
            "registry", "service", "process", "thread", "hook",
        ]
        c03 = np.zeros(len(patterns), dtype=np.float32)
        all_text = " ".join(strings).lower()
        for i, p in enumerate(patterns):
            c03[i] = float(p in all_text)
        feats["C03"] = c03

        # C04: String n-gram embedding (256d)
        c04 = np.zeros(256, dtype=np.float32)
        for s in strings[:200]:
            for i in range(len(s) - 2):
                h = hash(s[i:i+3]) % 256
                c04[h] += 1
        if c04.sum() > 0:
            c04 = np.log1p(c04)
            c04 /= np.linalg.norm(c04) + 1e-8
        feats["C04"] = c04

        return feats

    def _extract_hashes_fast(self, raw, pe_obj=None):
        """G01-G04: 哈希特征, 纯 Python。"""
        feats = {}

        # G01: Import hash (128d) — 用共享 pe_obj
        g01 = np.zeros(128, dtype=np.float32)
        try:
            imports = []
            if pe_obj and hasattr(pe_obj, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe_obj.DIRECTORY_ENTRY_IMPORT:
                    for func in entry.imports:
                        name = (func.name or b"").decode("ascii", errors="ignore")
                        if name:
                            imports.append(name)
            imphash_str = ",".join(sorted(imports)).lower()
            h = hashlib.md5(imphash_str.encode()).digest()
            for i, b in enumerate(h):
                g01[i*8:(i+1)*8] = [(b >> j) & 1 for j in range(8)]
        except Exception:
            pass
        feats["G01"] = g01

        # G02: ssdeep-like (128d)
        g02 = np.zeros(128, dtype=np.float32)
        chunk_size = max(len(raw) // 128, 64)
        for i in range(128):
            off = i * chunk_size
            blk = raw[off:off+chunk_size] if off < len(raw) else b"\x00"
            g02[i] = int(hashlib.md5(blk).hexdigest()[:4], 16) / 65536.0
        feats["G02"] = g02

        # G03: TLSH-like (128d)
        g03 = np.zeros(128, dtype=np.float32)
        for i in range(min(128, len(raw) // 64)):
            blk = raw[i*64:(i+1)*64]
            g03[i] = sum(blk) / (64 * 255)
        feats["G03"] = g03

        # G04: Rich header hash (128d)
        g04 = np.zeros(128, dtype=np.float32)
        rich_start = raw.find(b"Rich")
        if rich_start > 0:
            rich_data = raw[:rich_start+4]
            h = hashlib.sha256(rich_data).digest()
            for i in range(min(128, len(h)*4)):
                g04[i] = ((h[i//4] >> (i%4*2)) & 3) / 3.0
        feats["G04"] = g04

        return feats

    def _extract_metadata_fast(self, raw):
        """H01-H03: 元数据特征, 纯 Python。"""
        feats = {}
        h01 = np.zeros(20, dtype=np.float32)
        vi_off = raw.find(b"VS_VERSION_INFO")
        h01[0] = 1.0 if vi_off >= 0 else 0.0
        h01[1] = float(vi_off) / max(len(raw), 1) if vi_off >= 0 else 0
        feats["H01"] = h01

        h02 = np.zeros(20, dtype=np.float32)
        h02[0] = 1.0 if b".rsrc" in raw else 0.0
        feats["H02"] = h02

        h03 = np.zeros(20, dtype=np.float32)
        h03[0] = 1.0 if b"<?xml" in raw else 0.0
        h03[1] = 1.0 if b"requestedExecutionLevel" in raw else 0.0
        h03[2] = 1.0 if b"requireAdministrator" in raw else 0.0
        h03[3] = 1.0 if b"asInvoker" in raw else 0.0
        feats["H03"] = h03

        return feats

    def extract(self, pe_path: str, timeout: int = 120) -> Dict[str, np.ndarray]:
        """直接从 PE 提取全部特征 (不保存 bndb, 用于向后兼容)。"""
        self._ensure_bn()
        if not os.path.exists(pe_path):
            raise FileNotFoundError(pe_path)
        with open(pe_path, "rb") as _fh:
            raw = _fh.read()
        bv = self.bn.load(pe_path, options={
            "analysis.mode": "basic",
            "analysis.linearSweep.autorun": True,
            "analysis.limits.maxFunctionSize": 262144,
        })
        if bv is None:
            raise RuntimeError(f"BinaryNinja failed to load: {pe_path}")
        try:
            try:
                bv.update_analysis_and_wait()
            except Exception:
                pass
            features = {}
            stages = [
                ("A: PE Structure",    lambda: self._extract_pe_structure(bv, raw)),
                ("B: Byte Statistics",  lambda: self._extract_byte_stats(raw)),
                ("C: Strings",         lambda: self._extract_strings(bv, raw)),
                ("D: Disassembly",     lambda: self._extract_disassembly(bv)),
                ("E: Graph Structure",  lambda: self._extract_graphs(bv)),
                ("F: Visualization",   lambda: self._extract_visualization(raw)),
                ("G: Hashes",          lambda: self._extract_hashes(bv, raw)),
                ("H: Metadata",        lambda: self._extract_metadata(bv, raw)),
            ]
            for stage_name, stage_fn in stages:
                try:
                    features.update(stage_fn())
                except Exception:
                    pass
            return features
        finally:
            bv.file.close()

    # ─── 仅提取 D+E (需要 BinaryNinja 反汇编) ──────

    def extract_de_only(self, pe_path: str) -> Dict[str, np.ndarray]:
        """只提取 D01-D07 + E01-E04 (需要 BinaryNinja)。"""
        self._ensure_bn()
        if self.bn is None:
            raise RuntimeError("BinaryNinja not available for D/E extraction")
        bv = self.bn.load(pe_path, options={
            "analysis.mode": "basic",
            "analysis.linearSweep.autorun": True,
            "analysis.limits.maxFunctionSize": 262144,
        })
        if bv is None:
            raise RuntimeError(f"BinaryNinja failed to load: {pe_path}")
        try:
            try:
                bv.update_analysis_and_wait()
            except Exception:
                pass
            features = {}
            for stage_fn in [
                lambda: self._extract_disassembly(bv),
                lambda: self._extract_graphs(bv),
            ]:
                try:
                    features.update(stage_fn())
                except Exception:
                    pass
            return features
        finally:
            bv.file.close()

    # ─── 批量生成 bndb ──────────────────

    def generate_bndbs_batch(self, pe_bndb_pairs, n_workers=4, batch_size=20):
        """批量 PE → .bndb。每个子进程处理 batch_size 个文件。"""
        from concurrent.futures import ThreadPoolExecutor
        import subprocess as sp, threading

        total = len(pe_bndb_pairs)
        if total == 0:
            return set(), {}

        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "_bndb_worker.py")
        with open(script_path, "w") as f:
            f.write('''# -*- coding: utf-8 -*-
import sys, os, gc
# Windows: 禁止 crash 弹出 WER 对话框
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0002 | 0x8000)
    except Exception:
        pass
try:
    import binaryninja as bn
    try: bn.log.log_to_stderr(bn.LogLevel.ErrorLog)
    except: pass
    try: bn.disable_default_log()
    except: pass
except ImportError:
    print("FAIL\\tNO_FILE\\tno_binaryninja", flush=True)
    sys.exit(0)

# 从参数文件读取: pe_path\\tbndb_path 每行一对
with open(sys.argv[1], "r", encoding="utf-8") as f:
    lines = f.read().strip().split("\\n")
for line in lines:
    parts = line.strip().split("\\t")
    if len(parts) != 2:
        continue
    pe_path, bndb_path = parts
    ok = False
    try:
        bv = bn.load(pe_path, options={
            "analysis.mode": "basic",
            "analysis.linearSweep.autorun": True,
            "analysis.limits.maxFunctionSize": 262144,
        })
        if bv is not None:
            bv.update_analysis_and_wait()
            bv.create_database(bndb_path)
            bv.file.close()
            del bv
            ok = True
            print(f"OK\\t{pe_path}", flush=True)
        else:
            print(f"FAIL\\t{pe_path}\\tload_returned_None", flush=True)
    except Exception as e:
        print(f"FAIL\\t{pe_path}\\t{str(e)[:120]}", flush=True)
    # 失败时删除残留的 bndb 文件
    # KEEP_BNDB(2026-09-15): 大批量场景下失败清理会累积数百次删除,
    # 触发 safe-delete 钩子把整个构建进程杀掉 → 环境变量可跳过删除,
    # 事后对 .cache/bndb 改名归档再统一处理
    if not ok and os.path.exists(bndb_path) \\
            and os.environ.get("CODEFENDER_KEEP_BNDB") != "1":
        try: os.remove(bndb_path)
        except: pass
    gc.collect()
''')

        lock = threading.Lock()
        progress = [0]
        ok_count = [0]
        fail_count = [0]
        t0 = time.time()
        first_error_printed = [False]
        ok_set = set()       # 成功的 PE 路径
        fail_reasons = {}    # PE 路径 → 失败原因

        def _run_batch(batch):
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                             delete=False, encoding="utf-8") as tf:
                for pe, bndb in batch:
                    tf.write(f"{pe}\t{bndb}\n")
                args_file = tf.name

            pe_set = {pe for pe, _ in batch}
            batch_timeout = 120 * len(batch) + 30  # 2min/file + 30s buffer
            cflags = 0x08000000 if sys.platform == "win32" else 0
            try:
                proc = sp.run([sys.executable, script_path, args_file],
                              capture_output=True, timeout=batch_timeout,
                              creationflags=cflags)
                stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
                stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
            except sp.TimeoutExpired as e:
                stdout = (e.stdout or b"").decode("utf-8", errors="replace")
                stderr = f"TIMEOUT after {batch_timeout}s"
            except Exception as e:
                stdout = ""
                stderr = str(e)
            finally:
                if os.environ.get("CODEFENDER_KEEP_BNDB") != "1":
                    try: os.remove(args_file)
                    except: pass

            reported = set()
            for line in stdout.strip().split("\n"):
                parts = line.split("\t", 2)
                if len(parts) >= 2 and parts[1] in pe_set:
                    reported.add(parts[1])
                    with lock:
                        progress[0] += 1
                        if parts[0] == "OK":
                            ok_count[0] += 1
                            ok_set.add(parts[1])
                        else:
                            fail_count[0] += 1
                            reason = parts[2] if len(parts) > 2 else "unknown"
                            fail_reasons[parts[1]] = reason
                            # 首次失败: 打印详情帮助诊断
                            if not first_error_printed[0]:
                                first_error_printed[0] = True
                                print(f"\n    首个FAIL: {reason}")
                                if stderr.strip():
                                    print(f"    stderr: {stderr.strip()[:800]}")

            n_unreported = 0
            for pe, bndb in batch:
                if pe not in reported:
                    with lock:
                        progress[0] += 1
                        fail_count[0] += 1
                        n_unreported += 1
                        fail_reasons[pe] = f"子进程崩溃(unreported)"
                    if os.path.exists(bndb) and \
                            os.environ.get("CODEFENDER_KEEP_BNDB") != "1":
                        try: os.remove(bndb)
                        except: pass

            if n_unreported > 0 and not first_error_printed[0]:
                first_error_printed[0] = True
                print(f"\n    子进程崩溃 ({n_unreported} 个未报告)")
                if stderr.strip():
                    print(f"    stderr: {stderr.strip()[:800]}")

            with lock:
                done = progress[0]
                if done % 100 == 0 or done >= total:
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 0.01)
                    eta = (total - done) / max(rate, 0.01)
                    print(f"\r    bndb [{done}/{total}] {done/total*100:.1f}% "
                          f"({rate:.1f}/s, ETA {eta:.0f}s, "
                          f"ok={ok_count[0]}, fail={fail_count[0]})",
                          end="", flush=True)

        batches = [pe_bndb_pairs[i:i+batch_size] for i in range(0, total, batch_size)]
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_run_batch, b) for b in batches]
            for f in futs:
                f.result()
        print()
        if os.environ.get("CODEFENDER_KEEP_BNDB") != "1":
            try: os.remove(script_path)
            except: pass

        return ok_set, fail_reasons

    def extract_de_from_bndb_batch(self, pe_paths, bndb_paths, npz_paths,
                                    n_workers=4, batch_size=10,
                                    on_ok=None):
        """从已有 .bndb 批量提取 D+E — 持久化 Worker 架构, 零连坐。

        架构:
          - 启动 n_workers 个长驻子进程, 每个只初始化 BinaryNinja 一次
          - Master 通过 stdin 逐文件发送任务, 通过 stdout 逐个读取结果
          - 子进程 crash → 只丢失当前文件, Master 自动重启, 继续处理
          - 零连坐: 一个样本的 segfault 不影响任何其他样本
          - 每个 Worker 处理 MAX_FILES_PER_WORKER 个文件后主动重启 (防内存泄漏)

        参数:
          on_ok: 可选回调 on_ok(pe_path), 每个文件成功后立即调用。
                 用于增量写入断点续传日志, 避免 Ctrl+C 丢失全部进度。
        """
        from concurrent.futures import ThreadPoolExecutor
        import subprocess as sp, threading, queue as queue_mod

        total = len(pe_paths)
        if total == 0:
            return set(), {}

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # ── 生成持久化 Worker 脚本 ──
        # 协议: Master → stdin 发 "pe_path\tbndb_path\tnpz_path\n"
        #        Worker → stdout 回 "OK\tpe_path\n" 或 "FAIL\tpe_path\treason\n"
        #        stderr → /dev/null (防管道死锁, BUG#1 修复)
        worker_script = os.path.join(script_dir, "_de_worker_persistent.py")
        _ws_lines = [
            "# -*- coding: utf-8 -*-",
            "import sys, os, gc, io, zipfile",
            "",
            "# Windows: 禁止 segfault 弹出 '应用程序错误' 对话框",
            "# 没有这个, 每次 BN crash 都弹窗, 阻塞 proc.wait(), 拖垮整个流程",
            "if sys.platform == 'win32':",
            "    try:",
            "        import ctypes",
            "        ctypes.windll.kernel32.SetErrorMode(0x0002 | 0x8000)",
            "    except Exception:",
            "        pass",
            "",
            "sys.path.insert(0, %s)" % repr(script_dir),
            "import numpy as np",
            "from extract_feature import BinjaExtractor",
            "",
            "def append_to_npz(npz_path, arrays):",
            "    with zipfile.ZipFile(npz_path, 'a', zipfile.ZIP_STORED) as zf:",
            "        existing = set(zf.namelist())",
            "        for key, arr in arrays.items():",
            "            fname = key + '.npy'",
            "            if fname in existing:",
            "                continue",
            "            buf = io.BytesIO()",
            "            np.save(buf, arr)",
            "            zf.writestr(fname, buf.getvalue())",
            "",
            "ext = BinjaExtractor()",
            "ext._ensure_bn()  # Worker 进程中加载 BN (Master 不加载)",
            "sys.stdout.write('READY\\n')",
            "sys.stdout.flush()",
            "",
            "n_done = 0",
            "for raw_line in sys.stdin:",
            "    line = raw_line.strip()",
            "    if not line or line == 'EXIT':",
            "        break",
            "    parts = line.split('\\t')",
            "    if len(parts) != 3:",
            "        sys.stdout.write('FAIL\\t\\tinvalid_input\\n')",
            "        sys.stdout.flush()",
            "        continue",
            "    pe_path, bndb_path, npz_path = parts",
            "    try:",
            "        bv = ext.bn.load(bndb_path)",
            "        if bv is None:",
            "            sys.stdout.write('FAIL\\t' + pe_path + '\\tbndb_load_None\\n')",
            "            sys.stdout.flush()",
            "            continue",
            "        feats = {}",
            "        try:",
            "            feats.update(ext._extract_disassembly(bv))",
            "        except Exception:",
            "            pass",
            "        try:",
            "            feats.update(ext._extract_graphs(bv))",
            "        except Exception:",
            "            pass",
            "        try:",
            "            bv.file.close()",
            "        except Exception:",
            "            pass",
            "        del bv",
            "        if feats:",
            "            append_to_npz(npz_path, feats)",
            "            del feats",
            "            sys.stdout.write('OK\\t' + pe_path + '\\n')",
            "        else:",
            "            sys.stdout.write('FAIL\\t' + pe_path + '\\tempty_feats\\n')",
            "        sys.stdout.flush()",
            "    except Exception as e:",
            "        sys.stdout.write('FAIL\\t' + pe_path + '\\t' + str(e)[:120] + '\\n')",
            "        sys.stdout.flush()",
            "    n_done += 1",
            "    if n_done % 20 == 0:",
            "        gc.collect()",
        ]
        with open(worker_script, "w", encoding="utf-8") as f:
            f.write("\n".join(_ws_lines) + "\n")

        lock = threading.Lock()
        ok_set = set()
        fail_reasons = {}
        progress = [0]
        ok_count = [0]
        fail_count = [0]
        restart_total = [0]
        t0 = time.time()

        # ── 共享任务队列 ──
        task_q = queue_mod.Queue()
        for i in range(total):
            task_q.put((pe_paths[i], bndb_paths[i], npz_paths[i]))

        PER_FILE_TIMEOUT = 180          # 单文件超时 (秒)
        WORKER_INIT_TIMEOUT = 120       # Worker 初始化超时
        MAX_RESTARTS_PER_SLOT = 200     # 单 slot 最大重启次数 (含主动重启)
        MAX_FILES_PER_WORKER = 500      # 每个 Worker 处理 N 个文件后主动重启 (防 BN 内存泄漏)

        def _update_progress(status, pe_path, reason=""):
            with lock:
                progress[0] += 1
                if status == "OK":
                    ok_count[0] += 1
                    ok_set.add(pe_path)
                    # ── BUG#2 修复: 立即回调写日志, 不等方法返回 ──
                    if on_ok is not None:
                        try:
                            on_ok(pe_path)
                        except Exception:
                            pass
                else:
                    fail_count[0] += 1
                    if pe_path:
                        fail_reasons[pe_path] = reason
                done = progress[0]
                if done % 20 == 0 or done == 1 or done >= total:
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 0.01)
                    eta = (total - done) / max(rate, 0.01)
                    print("\r    D+E [%d/%d] %.1f%% (%.1f/s, ETA %ds, "
                          "ok=%d, fail=%d, restarts=%d)"
                          % (done, total, done/total*100, rate, eta,
                             ok_count[0], fail_count[0], restart_total[0]),
                          end="", flush=True)

        def _readline_with_timeout(proc, timeout):
            """从子进程 stdout 读取一行, 带超时。仅用于 _start_worker 握手阶段。"""
            result = [None]
            def _do_read():
                try:
                    result[0] = proc.stdout.readline()
                except Exception:
                    pass
            t = threading.Thread(target=_do_read, daemon=True)
            t.start()
            t.join(timeout=timeout)
            if t.is_alive():
                return None
            return result[0]

        def _start_worker():
            """启动持久化 Worker, 等待 READY 信号。返回 (proc, out_q) 或 (None, None)。
            out_q 是一个 Queue, 由专属 reader 线程持续从 stdout 读入。"""
            try:
                env = os.environ.copy()
                env["PYTHONUTF8"] = "1"
                cflags = 0x08000000 if sys.platform == "win32" else 0
                proc = sp.Popen(
                    [sys.executable, worker_script],
                    stdin=sp.PIPE, stdout=sp.PIPE,
                    stderr=sp.DEVNULL,
                    env=env,
                    creationflags=cflags,
                )
                # 握手: 循环读, 跳过 BN 启动噪声, 直到 READY
                deadline = time.time() + WORKER_INIT_TIMEOUT
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    ready_line = _readline_with_timeout(proc, remaining)
                    if ready_line is None:
                        break
                    if b"READY" in ready_line:
                        # 握手成功 → 启动专属 reader 线程
                        out_q = queue_mod.Queue()
                        def _reader(p=proc, q=out_q):
                            try:
                                for raw in p.stdout:
                                    q.put(raw)
                            except Exception:
                                pass
                            q.put(None)  # EOF 哨兵
                        rt = threading.Thread(target=_reader, daemon=True)
                        rt.start()
                        return proc, out_q
                    if not ready_line:
                        break
                try: proc.stdout.close()
                except Exception: pass
                try: proc.kill()
                except Exception: pass
                try: proc.wait(timeout=5)
                except Exception: pass
                return None, None
            except Exception:
                return None, None

        def _shutdown_worker(proc):
            """优雅关闭: 发 EXIT, 等待退出。"""
            if proc is None:
                return
            try:
                proc.stdin.write(b"EXIT\n")
                proc.stdin.flush()
                proc.wait(timeout=15)
            except Exception:
                _kill_worker(proc)
            try: proc.stdout.close()
            except Exception: pass

        def _kill_worker(proc):
            """强制杀死, 关闭所有管道句柄。"""
            if proc is None:
                return
            try: proc.stdin.close()
            except Exception: pass
            try: proc.kill()
            except Exception: pass
            try: proc.wait(timeout=5)
            except Exception: pass
            try: proc.stdout.close()
            except Exception: pass

        def _worker_slot(slot_id):
            """一个 Worker 槽位: 管理持久化子进程, 逐文件处理, crash/内存 自动重启。

            通信架构 (Windows 安全):
              - 每个 Worker 有一个专属 reader 线程, 持续从 stdout 读入 Queue
              - Master 从 Queue.get(timeout) 读取结果 — 零线程创建
              - 总线程数 = n_workers 个 reader, 而非 total_files 个
            """
            proc = None
            out_q = None
            slot_restarts = 0
            files_on_current_worker = 0

            while True:
                # ── 取任务 ──
                try:
                    pe_path, bndb_path, npz_path = task_q.get_nowait()
                except queue_mod.Empty:
                    break

                # ── BUG#3 修复: 处理 N 个文件后主动重启, 防 BN 内存泄漏 ──
                if (proc is not None and proc.poll() is None
                        and files_on_current_worker >= MAX_FILES_PER_WORKER):
                    _shutdown_worker(proc)
                    proc = None
                    out_q = None
                    files_on_current_worker = 0

                # ── 确保 Worker 存活 ──
                if proc is None or proc.poll() is not None:
                    _kill_worker(proc)
                    proc = None
                    out_q = None
                    files_on_current_worker = 0
                    if slot_restarts >= MAX_RESTARTS_PER_SLOT:
                        _update_progress("FAIL", pe_path, "max_restarts_exceeded")
                        continue
                    proc, out_q = _start_worker()
                    slot_restarts += 1
                    with lock:
                        restart_total[0] += 1
                    if proc is None:
                        _update_progress("FAIL", pe_path, "worker_init_failed")
                        continue

                # ── 发送任务 ──
                try:
                    msg = ("%s\t%s\t%s\n" % (pe_path, bndb_path, npz_path)).encode("utf-8")
                    proc.stdin.write(msg)
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    _kill_worker(proc)
                    proc = None
                    out_q = None
                    _update_progress("FAIL", pe_path, "broken_pipe")
                    continue

                # ── 从 Queue 读取结果 (带超时, 零线程创建) ──
                try:
                    resp = out_q.get(timeout=PER_FILE_TIMEOUT)
                except queue_mod.Empty:
                    # 超时
                    _kill_worker(proc)
                    proc = None
                    out_q = None
                    _update_progress("FAIL", pe_path, "timeout_%ds" % PER_FILE_TIMEOUT)
                    continue

                if resp is None or not resp:
                    # EOF 哨兵或空行 → Worker 已死
                    _kill_worker(proc)
                    proc = None
                    out_q = None
                    _update_progress("FAIL", pe_path, "worker_crash")
                    continue

                # ── 解析结果 ──
                line = resp.decode("utf-8", errors="replace").strip()
                parts = line.split("\t", 2)
                status = parts[0] if parts else "FAIL"
                reason = parts[2] if len(parts) > 2 else ""
                _update_progress(status, pe_path, reason)
                files_on_current_worker += 1

            # ── 清理 ──
            _shutdown_worker(proc)

        # ── 启动 n_workers 个 Worker 槽位 ──
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_worker_slot, i) for i in range(n_workers)]
            for f in futs:
                f.result()
        print()
        if os.environ.get("CODEFENDER_KEEP_BNDB") != "1":
            try: os.remove(worker_script)
            except Exception: pass

        return ok_set, fail_reasons

    # ═══════════════════════════════════════
    #  A: PE Structure (Appendix C.1)
    # ═══════════════════════════════════════

    def _extract_pe_structure(self, bv, raw):
        """A01-A15: PE header, sections, imports, exports, security metadata."""
        feats = {}

        # A01: DOS Header (15d)
        dos = np.zeros(15, dtype=np.float32)
        if len(raw) >= 64:
            dos[0] = struct.unpack_from("<H", raw, 0)[0]  # e_magic
            dos[1] = struct.unpack_from("<I", raw, 60)[0] if len(raw) > 63 else 0  # e_lfanew
            dos[2] = len(raw)  # file size
            for i in range(3, min(15, 30)):
                off = i * 2
                if off + 2 <= 64:
                    dos[i] = struct.unpack_from("<H", raw, off)[0]
        feats["A01"] = dos

        # A02: COFF Header (7d)
        coff = np.zeros(7, dtype=np.float32)
        pe_off = struct.unpack_from("<I", raw, 60)[0] if len(raw) > 63 else 0
        if pe_off + 24 <= len(raw) and raw[pe_off:pe_off+4] == b"PE\x00\x00":
            h = pe_off + 4
            coff[0] = struct.unpack_from("<H", raw, h)[0]      # Machine
            coff[1] = struct.unpack_from("<H", raw, h+2)[0]    # NumberOfSections
            coff[2] = struct.unpack_from("<I", raw, h+4)[0]    # TimeDateStamp
            coff[3] = struct.unpack_from("<I", raw, h+8)[0]    # PointerToSymbolTable
            coff[4] = struct.unpack_from("<I", raw, h+12)[0]   # NumberOfSymbols
            coff[5] = struct.unpack_from("<H", raw, h+16)[0]   # SizeOfOptionalHeader
            coff[6] = struct.unpack_from("<H", raw, h+18)[0]   # Characteristics
        feats["A02"] = coff

        # A03: Optional Header (30d) — via BinaryNinja
        opt = np.zeros(30, dtype=np.float32)
        if bv.start is not None:
            opt[0] = float(bv.entry_point or 0)
            opt[1] = float(bv.start)
            opt[2] = float(bv.end - bv.start) if bv.end else 0
            opt[3] = len(list(bv.sections))
            opt[4] = len(list(bv.functions))
        feats["A03"] = opt

        # A04: Section Features (MAX_SECTIONS × 4 = 192d)
        sec_feats = np.zeros(MAX_SECTIONS * 4, dtype=np.float32)
        for i, sec in enumerate(bv.sections.values()):
            if i >= MAX_SECTIONS:
                break
            base = i * 4
            try:
                sec_feats[base]     = float(sec.length or 0)
                sec_feats[base + 1] = float((sec.end or 0) - (sec.start or 0))
                sec_data = bv.read(sec.start, min(sec.length or 0, 65536))
                sec_feats[base + 2] = self._entropy(sec_data) if sec_data else 0
                sec_feats[base + 3] = float(sec.semantics.value if hasattr(sec.semantics, 'value') else 0)
            except Exception:
                pass
        feats["A04"] = sec_feats

        # A05: Section Anomaly Indicators (20d)
        anom = np.zeros(20, dtype=np.float32)
        for i, sec in enumerate(bv.sections.values()):
            if i >= 10:
                break
            try:
                slen = sec.length or 0
                anom[i*2] = 1.0 if slen == 0 else 0.0
                anom[i*2+1] = 1.0 if slen > 10*1024*1024 else 0.0
            except Exception:
                pass
        feats["A05"] = anom

        # A06: Import Table (IMPORT_BUCKETS d) — hash-bucketed
        imp = np.zeros(IMPORT_BUCKETS, dtype=np.float32)
        for sym in bv.get_symbols_of_type(self.bn.SymbolType.ImportedFunctionSymbol):
            h = hash(sym.full_name) % IMPORT_BUCKETS
            imp[h] += 1.0
        if imp.sum() > 0:
            imp /= imp.sum()
        feats["A06"] = imp

        # A07: Export Table (32d)
        exp = np.zeros(32, dtype=np.float32)
        exports = list(bv.get_symbols_of_type(self.bn.SymbolType.FunctionSymbol))
        exp[0] = len(exports)
        exp[1] = sum(1 for s in exports if s.name and not s.name.startswith("sub_"))
        feats["A07"] = exp

        # A08-A15: Security metadata (51d total, simplified)
        sec_meta = np.zeros(51, dtype=np.float32)
        # Rich header presence, TLS, relocations etc.
        sec_meta[0] = 1.0 if b"Rich" in raw[:1024] else 0.0
        sec_meta[1] = float(len(raw))
        feats["A08_A15"] = sec_meta

        return feats

    # ═══════════════════════════════════════
    #  B: Byte Statistics (Appendix C.2)
    # ═══════════════════════════════════════

    def _extract_byte_stats(self, raw):
        feats = {}

        # B01: Raw byte sequence (truncated/padded to MAX_BYTES)
        # 存储为整数 [0, 255]: V1 字节模型的 Embedding 层需要整数索引
        # B01 用 uint8 存储（B01-fix 2026-09-15）：原 int64 使单样本 npz 膨胀到
        # ~18MB（10k 样本 ≈ 180GB，超出磁盘容量）。两个消费方均显式转型——
        # data_loader.py:826 `.astype(np.int32)`、engine/app.py:636
        # `_to_numpy_1d(..., np.int32)`，uint8 → int32 无损。
        b01 = np.zeros(MAX_BYTES, dtype=np.uint8)
        n = min(len(raw), MAX_BYTES)
        b01[:n] = np.frombuffer(raw[:n], dtype=np.uint8)
        feats["B01"] = b01

        # B02: 1-gram histogram (256d)
        counts = np.bincount(np.frombuffer(raw, dtype=np.uint8), minlength=256)
        feats["B02"] = (counts / max(len(raw), 1)).astype(np.float32)

        # B03: 2-gram histogram (256d, XOR-folded from 65536 pairs)
        arr = np.frombuffer(raw, dtype=np.uint8)
        if len(arr) >= 2:
            bigram_full = arr[:-1].astype(np.int32) * 256 + arr[1:].astype(np.int32)
            # XOR fold: 高字节 XOR 低字节, 保留两个字节的信息
            bigrams = np.bincount((bigram_full >> 8) ^ (bigram_full & 0xFF),
                                  minlength=256).astype(np.float32)
            if bigrams.sum() > 0:
                bigrams /= bigrams.sum()
        else:
            bigrams = np.zeros(256, dtype=np.float32)
        feats["B03"] = bigrams

        # B04: Entropy histogram (256d) — vectorized
        block_ents = self._block_entropies(raw, BLOCK_SIZE)
        ent_hist = np.zeros(ENTROPY_BINS, dtype=np.float32)
        bin_indices = np.clip((block_ents / 8.0 * ENTROPY_BINS).astype(int), 0, ENTROPY_BINS - 1)
        np.add.at(ent_hist, bin_indices, 1)
        if ent_hist.sum() > 0:
            ent_hist /= ent_hist.sum()
        feats["B04"] = ent_hist

        # B05: Global entropy statistics (8d) — using precomputed block entropies
        ent_stat = np.zeros(8, dtype=np.float32)
        ent_stat[0] = self._entropy(raw)
        split = min(1024, len(raw))
        ent_stat[1] = self._entropy(raw[:split])
        ent_stat[2] = self._entropy(raw[split:]) if len(raw) > split else 0
        if len(block_ents) > 0:
            ent_stat[3] = float(block_ents.max())
            ent_stat[4] = float(block_ents.min())
            ent_stat[5] = float(block_ents.mean())
            ent_stat[6] = float(block_ents.std())
        ent_stat[7] = len(raw) / (MAX_BYTES + 1)
        feats["B05"] = ent_stat

        # B06: Markov transition matrix (256d row-entropy summary)
        arr = np.frombuffer(raw, dtype=np.uint8)
        if len(arr) >= 2:
            # Vectorized: build 256×256 transition count matrix
            pairs = arr[:-1].astype(np.int32) * 256 + arr[1:].astype(np.int32)
            flat_counts = np.bincount(pairs, minlength=65536)
            markov = flat_counts.reshape(256, 256).astype(np.float64)
            row_sums = markov.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            markov /= row_sums
            # Row entropy
            with np.errstate(divide='ignore', invalid='ignore'):
                log_m = np.where(markov > 0, np.log2(markov), 0.0)
            row_entropy = -np.sum(markov * log_m, axis=1)
        else:
            row_entropy = np.zeros(256, dtype=np.float64)
        feats["B06"] = row_entropy.astype(np.float32)

        # B07: File size features (6d)
        fsz = np.zeros(6, dtype=np.float32)
        fsz[0] = len(raw)
        fsz[1] = np.log1p(len(raw))
        fsz[2] = len(raw) % 512
        fsz[3] = 1.0 if len(raw) % 512 == 0 else 0.0
        fsz[4] = len(raw) / MAX_BYTES
        fsz[5] = 1.0 if len(raw) > MAX_BYTES else 0.0
        feats["B07"] = fsz

        return feats

    # ═══════════════════════════════════════
    #  C: Strings (Appendix C.1)
    # ═══════════════════════════════════════

    def _extract_strings(self, bv, raw):
        feats = {}

        # Extract printable strings
        # BinaryNinja 5.x: bv.strings 迭代时可能在损坏节区上触发 __getitem__ 错误
        # 回退方案: 从原始字节提取 ASCII 字符串
        raw_strings = []
        try:
            for s in bv.strings:
                try:
                    v = s.value
                    if isinstance(v, bytes):
                        v = v.decode("utf-8", errors="ignore")
                    if len(v) >= 4:
                        raw_strings.append(v)
                except Exception:
                    pass
        except Exception:
            # bv.strings 迭代本身失败 → 从原始字节提取
            import re
            raw_strings = re.findall(rb'[\x20-\x7e]{4,}', raw)
            raw_strings = [s.decode("ascii", errors="ignore") for s in raw_strings[:5000]]
        strings = raw_strings

        # C01: String content (variable → 256d embedding via hash trick)
        c01 = np.zeros(256, dtype=np.float32)
        for s in strings:
            h = hash(s) % 256
            c01[h] += 1
        if c01.sum() > 0:
            c01 = np.log1p(c01)
            c01 /= c01.max() + 1e-8
        feats["C01"] = c01

        # C02: String statistics (10d)
        c02 = np.zeros(10, dtype=np.float32)
        if strings:
            lens = [len(s) for s in strings]
            c02[0] = len(strings)
            c02[1] = np.mean(lens)
            c02[2] = np.max(lens)
            c02[3] = np.std(lens)
            c02[4] = sum(1 for s in strings if any(c.isupper() for c in s)) / len(strings)
            c02[5] = sum(1 for s in strings if s.startswith("http")) / max(len(strings), 1)
            c02[6] = sum(1 for s in strings if "\\" in s or "/" in s) / max(len(strings), 1)
            c02[7] = sum(1 for s in strings if ".dll" in s.lower()) / max(len(strings), 1)
            c02[8] = sum(1 for s in strings if ".exe" in s.lower()) / max(len(strings), 1)
            c02[9] = sum(1 for s in strings if any(c.isdigit() for c in s)) / max(len(strings), 1)
        feats["C02"] = c02

        # C03: Sensitive pattern matching (31d)
        patterns = [
            "cmd", "powershell", "reg", "net ", "http", "ftp",
            "encrypt", "decrypt", "password", "admin", "root",
            "kernel32", "ntdll", "ws2_32", "advapi32", "shell32",
            "create", "write", "delete", "execute", "inject",
            "mutex", "pipe", "socket", "connect", "download",
            "registry", "service", "process", "thread", "hook",
        ]
        c03 = np.zeros(len(patterns), dtype=np.float32)
        all_text = " ".join(strings).lower()
        for i, p in enumerate(patterns):
            c03[i] = float(p in all_text)
        feats["C03"] = c03

        # C04: String embedding (256d, TF-IDF proxy via char n-grams)
        c04 = np.zeros(256, dtype=np.float32)
        for s in strings[:200]:
            for i in range(len(s) - 2):
                h = hash(s[i:i+3]) % 256
                c04[h] += 1
        if c04.sum() > 0:
            c04 = np.log1p(c04)
            c04 /= np.linalg.norm(c04) + 1e-8
        feats["C04"] = c04

        return feats

    # ═══════════════════════════════════════
    #  D: Disassembly via BinaryNinja (Appendix C.3)
    # ═══════════════════════════════════════

    def _extract_disassembly(self, bv):
        """D01-D07: 单次遍历 bv.functions, 收集所有反汇编数据。"""
        feats = {}

        # ── 单次遍历: 收集全局 + per-function 数据 ──
        all_opcodes = []
        global_opcode_counts = Counter()
        global_cat_counts = Counter()
        func_data = []  # [{opcodes: Counter, n_bbs: int, size: int, callees: int, ...}]
        all_bb_sizes = []
        all_bb_out_edges = []

        def _parse_mnemonic(line):
            try:
                if hasattr(line, 'tokens'):
                    tokens = line.tokens
                elif isinstance(line, tuple):
                    tokens = line[0] if line else []
                else:
                    tokens = line
                if tokens and len(tokens) > 0:
                    text = tokens[0].text if hasattr(tokens[0], 'text') else str(tokens[0])
                    return text.strip().lower()
            except Exception:
                pass
            return ""

        for func in bv.functions:
            func_ops = Counter()
            func_bb_count = 0
            try:
                for block in func.basic_blocks:
                    func_bb_count += 1
                    if block.length is not None:
                        all_bb_sizes.append(block.length)
                    try:
                        all_bb_out_edges.append(len(block.outgoing_edges))
                    except Exception:
                        all_bb_out_edges.append(0)

                    try:
                        lines = block.disassembly_text
                    except AttributeError:
                        lines = block
                    for line in lines:
                        m = _parse_mnemonic(line)
                        if m:
                            all_opcodes.append(m)
                            global_opcode_counts[m] += 1
                            global_cat_counts[_CAT_LOOKUP.get(m, "other")] += 1
                            func_ops[m] += 1
            except Exception:
                pass

            # Per-function metadata
            fsize = 0
            try:
                if hasattr(func, 'total_bytes') and func.total_bytes:
                    fsize = func.total_bytes
                elif func.highest_address and func.start:
                    fsize = func.highest_address - func.start
            except Exception:
                pass
            n_callees = 0
            is_leaf = False
            is_recursive = False
            try:
                callees = list(func.callees)
                n_callees = len(callees)
                is_leaf = n_callees == 0
                is_recursive = any(c == func for c in callees)
            except Exception:
                pass
            func_data.append({
                "ops": func_ops, "n_bbs": func_bb_count, "size": fsize,
                "n_callees": n_callees, "is_leaf": is_leaf, "is_recursive": is_recursive,
            })

        n_funcs = len(func_data)

        # ── D01: Opcode sequence ──
        d01 = np.zeros(MAX_OPCODES, dtype=np.int64)
        for i, op in enumerate(all_opcodes[:MAX_OPCODES]):
            d01[i] = hash(op) % 511 + 1
        feats["D01"] = d01

        # ── D02: Opcode bigram ──
        d02 = np.zeros(NGRAM_DIM, dtype=np.float32)
        for i in range(len(all_opcodes) - 1):
            d02[hash(all_opcodes[i] + "_" + all_opcodes[i+1]) % NGRAM_DIM] += 1
        if d02.sum() > 0:
            d02 = np.log1p(d02); d02 /= np.linalg.norm(d02) + 1e-8
        feats["D02"] = d02

        # ── D03: Opcode trigram ──
        d03 = np.zeros(NGRAM_DIM, dtype=np.float32)
        for i in range(len(all_opcodes) - 2):
            d03[hash("_".join(all_opcodes[i:i+3])) % NGRAM_DIM] += 1
        if d03.sum() > 0:
            d03 = np.log1p(d03); d03 /= np.linalg.norm(d03) + 1e-8
        feats["D03"] = d03

        # ── D04: Category distribution ──
        d04 = np.zeros(len(_CAT_NAMES), dtype=np.float32)
        total_cat = sum(global_cat_counts.values()) or 1
        for i, cat in enumerate(_CAT_NAMES):
            d04[i] = global_cat_counts.get(cat, 0) / total_cat
        feats["D04"] = d04

        # ── D05: Function statistics (from func_data, no re-traversal) ──
        d05 = np.zeros(12, dtype=np.float32)
        d05[0] = n_funcs
        if n_funcs > 0:
            sizes = [fd["size"] for fd in func_data if fd["size"] > 0]
            if sizes:
                d05[1] = np.mean(sizes); d05[2] = np.max(sizes); d05[3] = np.std(sizes)
            bbc = [fd["n_bbs"] for fd in func_data]
            if bbc:
                d05[4] = np.mean(bbc); d05[5] = np.max(bbc)
            d05[6] = sum(1 for fd in func_data if fd["is_leaf"])
            d05[7] = sum(1 for fd in func_data if fd["is_recursive"])
            d05[8] = d05[6] / n_funcs
            d05[9] = len(all_opcodes)
            d05[10] = len(all_opcodes) / n_funcs
            d05[11] = len(global_opcode_counts)
        feats["D05"] = d05

        # ── D06: Basic block statistics (from collected data) ──
        d06 = np.zeros(10, dtype=np.float32)
        n_bbs_total = sum(fd["n_bbs"] for fd in func_data)
        d06[0] = n_bbs_total
        if all_bb_sizes:
            d06[1] = np.mean(all_bb_sizes); d06[2] = np.max(all_bb_sizes)
            d06[3] = np.std(all_bb_sizes)
        if all_bb_out_edges:
            d06[4] = np.mean(all_bb_out_edges); d06[5] = np.max(all_bb_out_edges)
            d06[6] = sum(1 for e in all_bb_out_edges if e == 0)
            d06[7] = sum(1 for e in all_bb_out_edges if e > 1)
            d06[8] = d06[7] / max(n_bbs_total, 1)
        if all_bb_sizes:
            d06[9] = sum(all_bb_sizes) / max(n_bbs_total, 1)
        feats["D06"] = d06

        # ── D07: Per-function embedding (from func_data, no re-traversal) ──
        d07 = np.zeros(MAX_FUNCTIONS * 128, dtype=np.float32)
        for fi in range(min(n_funcs, MAX_FUNCTIONS)):
            emb = np.zeros(128, dtype=np.float32)
            for op, cnt in func_data[fi]["ops"].items():
                emb[hash(op) % 128] += cnt
            if emb.sum() > 0:
                emb /= np.linalg.norm(emb) + 1e-8
            d07[fi*128:(fi+1)*128] = emb
        feats["D07"] = d07

        return feats

    # ═══════════════════════════════════════
    #  E: Graph Structure via BinaryNinja (Appendix C.4)
    # ═══════════════════════════════════════

    def _extract_graphs(self, bv):
        """CFG and call graph — 单次遍历所有函数和基本块。"""
        feats = {}
        funcs = list(bv.functions)
        n_funcs = len(funcs)

        # ── 单次遍历: 收集所有 graph 数据 ──
        cfg_nodes, cfg_edges = 0, 0
        cg_edges = 0
        cg_adj = defaultdict(set)
        func_bb_counts = []    # per-func bb count
        back_edges = 0
        leaf_funcs = 0
        recursive_funcs = 0
        out_degrees = []
        in_degree_map = defaultdict(int)  # func_start → in_degree

        for fi, func in enumerate(funcs):
            fstart = None
            try:
                fstart = func.start
            except: pass

            # Basic blocks
            n_bbs = 0
            try:
                for bb in func.basic_blocks:
                    n_bbs += 1
                    cfg_nodes += 1
                    bb_start = None
                    try: bb_start = bb.start
                    except: pass
                    try:
                        edges = bb.outgoing_edges
                        n_edges = len(edges)
                        cfg_edges += n_edges
                        # Back edges (前 100 个函数)
                        if fi < 100 and bb_start is not None:
                            for edge in edges:
                                try:
                                    t = edge.target
                                    if t is not None and t.start is not None and t.start <= bb_start:
                                        back_edges += 1
                                except: pass
                    except: pass
            except: pass
            func_bb_counts.append(n_bbs)

            # Callees (call graph)
            callees_list = []
            try:
                callees_list = list(func.callees)
            except: pass
            n_callees = len(callees_list)
            out_degrees.append(n_callees)
            if n_callees == 0:
                leaf_funcs += 1

            for callee in callees_list:
                try:
                    cstart = callee.start
                    if cstart is None or fstart is None:
                        continue
                    cg_edges += 1
                    cg_adj[fstart].add(cstart)
                    in_degree_map[cstart] += 1
                    if cstart == fstart:
                        recursive_funcs += 1
                except: pass

        # ── E03: CFG statistics (12d) ──
        e03 = np.zeros(12, dtype=np.float32)
        e03[0] = cfg_nodes
        e03[1] = cfg_edges
        e03[2] = cfg_edges / max(cfg_nodes, 1)
        if func_bb_counts:
            e03[3] = max(func_bb_counts)
            e03[4] = np.mean(func_bb_counts)
            e03[5] = np.std(func_bb_counts)
        e03[6] = back_edges
        e03[7] = n_funcs
        e03[8] = cfg_nodes / max(n_funcs, 1)
        e03[9] = cfg_edges / max(n_funcs, 1)
        e03[10] = back_edges / max(n_funcs, 1)
        e03[11] = 1.0 if cfg_nodes > 0 else 0.0
        feats["E03"] = e03

        # ── E04: Call graph statistics (10d) ──
        e04 = np.zeros(10, dtype=np.float32)
        e04[0] = n_funcs
        e04[1] = cg_edges
        e04[2] = cg_edges / max(n_funcs, 1)
        e04[3] = leaf_funcs / max(n_funcs, 1)
        e04[4] = recursive_funcs / max(n_funcs, 1)
        depths = []
        for f in funcs[:50]:
            try:
                d = self._call_depth(f, set(), 0, 10)
                if d is not None:
                    depths.append(d)
            except: pass
        e04[5] = max(depths) if depths else 0
        e04[6] = np.mean(depths) if depths else 0
        if out_degrees:
            e04[7] = max(out_degrees)
        in_degrees = list(in_degree_map.values())
        if in_degrees:
            e04[8] = max(in_degrees)
        e04[9] = len(set().union(*cg_adj.values())) if cg_adj else 0
        feats["E04"] = e04

        return feats

    def _call_depth(self, func, visited, depth, max_depth):
        if depth >= max_depth:
            return depth
        try:
            fstart = func.start
            if fstart is None or fstart in visited:
                return depth
            visited.add(fstart)
        except Exception:
            return depth
        max_d = depth
        try:
            for callee in list(func.callees)[:5]:
                d = self._call_depth(callee, visited, depth + 1, max_depth)
                if d is not None:
                    max_d = max(max_d, d)
        except Exception:
            pass
        return max_d

    # ═══════════════════════════════════════
    #  F: Visualization (Appendix C.5)
    # ═══════════════════════════════════════

    def _extract_visualization(self, raw):
        feats = {}

        # F01: Grayscale image (256×256)
        feats["F01"] = self._bytes_to_image(raw, channels=1)

        # F02: Color image via Hilbert-style mapping (256×256×3)
        feats["F02"] = self._bytes_to_color_image(raw)

        # F03: Markov transition image (256×256) — vectorized
        arr = np.frombuffer(raw, dtype=np.uint8)
        if len(arr) >= 2:
            pairs = arr[:-1].astype(np.int32) * 256 + arr[1:].astype(np.int32)
            flat_counts = np.bincount(pairs, minlength=65536)
            markov_img = flat_counts.reshape(256, 256).astype(np.float32)
            row_max = markov_img.max(axis=1, keepdims=True)
            row_max[row_max == 0] = 1
            feats["F03"] = markov_img / row_max
        else:
            feats["F03"] = np.zeros((256, 256), dtype=np.float32)

        # F04: Entropy heatmap (256×256) — bincount 偏移技巧
        grid = IMG_SIZE
        total_cells = grid * grid
        chunk = max(len(raw) // total_cells, 1)
        if len(raw) >= total_cells and chunk >= 4:
            usable = total_cells * chunk
            cells = np.frombuffer(raw[:usable], dtype=np.uint8).reshape(total_cells, chunk)
            offsets = np.arange(total_cells, dtype=np.int32).repeat(chunk) * 256
            flat_idx = offsets + cells.ravel().astype(np.int32)
            counts = np.bincount(flat_idx, minlength=total_cells * 256).reshape(total_cells, 256)
            probs = counts.astype(np.float32) / chunk
            with np.errstate(divide='ignore', invalid='ignore'):
                log_p = np.where(probs > 0, np.log2(probs), 0.0)
            cell_ent = -np.sum(probs * log_p, axis=1) / 8.0
            feats["F04"] = cell_ent.reshape(grid, grid).astype(np.float32)
        else:
            feats["F04"] = np.zeros((grid, grid), dtype=np.float32)

        return feats

    def _bytes_to_image(self, raw, channels=1):
        """Reshape bytes to 256×256 grayscale (Nataraj et al. 2011)."""
        arr = np.frombuffer(raw[:IMG_SIZE*IMG_SIZE], dtype=np.uint8).astype(np.float32) / 255.0
        if len(arr) < IMG_SIZE * IMG_SIZE:
            arr = np.pad(arr, (0, IMG_SIZE*IMG_SIZE - len(arr)))
        return arr.reshape(IMG_SIZE, IMG_SIZE)

    def _bytes_to_color_image(self, raw):
        """RGB image from byte triplets."""
        n = min(len(raw), IMG_SIZE*IMG_SIZE*3)
        arr = np.frombuffer(raw[:n], dtype=np.uint8).astype(np.float32) / 255.0
        needed = IMG_SIZE * IMG_SIZE * 3
        if len(arr) < needed:
            arr = np.pad(arr, (0, needed - len(arr)))
        return arr.reshape(IMG_SIZE, IMG_SIZE, 3)

    # ═══════════════════════════════════════
    #  G: Hashes (Appendix C.6)
    # ═══════════════════════════════════════

    def _extract_hashes(self, bv, raw):
        feats = {}

        # G01: Import hash (128d embedding)
        g01 = np.zeros(128, dtype=np.float32)
        imports = [s.full_name for s in bv.get_symbols_of_type(self.bn.SymbolType.ImportedFunctionSymbol)]
        imphash_str = ",".join(sorted(imports)).lower()
        h = hashlib.md5(imphash_str.encode()).digest()
        for i, b in enumerate(h):
            g01[i*8:(i+1)*8] = [(b >> j) & 1 for j in range(8)]
        feats["G01"] = g01

        # G02: ssdeep-like (128d, context-triggered piecewise hash proxy)
        g02 = np.zeros(128, dtype=np.float32)
        chunk_size = max(len(raw) // 128, 64)
        for i in range(128):
            off = i * chunk_size
            blk = raw[off:off+chunk_size] if off < len(raw) else b"\x00"
            g02[i] = int(hashlib.md5(blk).hexdigest()[:4], 16) / 65536.0
        feats["G02"] = g02

        # G03: TLSH-like (128d, trend locality sensitive hash proxy)
        g03 = np.zeros(128, dtype=np.float32)
        for i in range(min(128, len(raw) // 64)):
            blk = raw[i*64:(i+1)*64]
            g03[i] = sum(blk) / (64 * 255)
        feats["G03"] = g03

        # G04: Rich header hash (128d)
        g04 = np.zeros(128, dtype=np.float32)
        rich_start = raw.find(b"Rich")
        if rich_start > 0:
            rich_data = raw[:rich_start+4]
            h = hashlib.sha256(rich_data).digest()
            for i in range(min(128, len(h)*4)):
                g04[i] = ((h[i//4] >> (i%4*2)) & 3) / 3.0
        feats["G04"] = g04

        return feats

    # ═══════════════════════════════════════
    #  H: Metadata
    # ═══════════════════════════════════════

    def _extract_metadata(self, bv, raw):
        feats = {}

        # H01: Version info (20d)
        h01 = np.zeros(20, dtype=np.float32)
        # Check for VS_VERSION_INFO
        vi_off = raw.find(b"VS_VERSION_INFO")
        h01[0] = 1.0 if vi_off >= 0 else 0.0
        h01[1] = float(vi_off) / max(len(raw), 1) if vi_off >= 0 else 0
        feats["H01"] = h01

        # H02: Resource features (20d)
        h02 = np.zeros(20, dtype=np.float32)
        # BinaryNinja doesn't directly expose resources, use raw heuristics
        rsrc = raw.find(b".rsrc")
        h02[0] = 1.0 if rsrc >= 0 else 0.0
        feats["H02"] = h02

        # H03: Manifest (20d)
        h03 = np.zeros(20, dtype=np.float32)
        h03[0] = 1.0 if b"<?xml" in raw else 0.0
        h03[1] = 1.0 if b"requestedExecutionLevel" in raw else 0.0
        h03[2] = 1.0 if b"requireAdministrator" in raw else 0.0
        h03[3] = 1.0 if b"asInvoker" in raw else 0.0
        feats["H03"] = h03

        return feats

    # ═══════════════════════════════════════
    #  Utilities
    # ═══════════════════════════════════════

    @staticmethod
    def _entropy(data):
        """Shannon entropy in bits — vectorized."""
        if not data or len(data) == 0:
            return 0.0
        arr = np.frombuffer(data if isinstance(data, bytes) else bytes(data), dtype=np.uint8)
        counts = np.bincount(arr, minlength=256)
        probs = counts / len(arr)
        mask = probs > 0
        return float(-np.dot(probs[mask], np.log2(probs[mask])))

    @staticmethod
    def _block_entropies(raw, block_size=1024):
        """Block entropy — 单次 bincount, O(n)。"""
        n = len(raw)
        if n == 0:
            return np.array([0.0])
        arr = np.frombuffer(raw, dtype=np.uint8)
        n_blocks = max(1, n // block_size)
        trimmed = arr[:n_blocks * block_size].reshape(n_blocks, block_size)
        # 偏移技巧: 每个 block 的字节值偏移 block_id * 256
        offsets = np.arange(n_blocks, dtype=np.int32).repeat(block_size) * 256
        flat_idx = offsets + trimmed.ravel().astype(np.int32)
        counts = np.bincount(flat_idx, minlength=n_blocks * 256).reshape(n_blocks, 256)
        probs = counts.astype(np.float32) / block_size
        with np.errstate(divide='ignore', invalid='ignore'):
            log_probs = np.where(probs > 0, np.log2(probs), 0.0)
        return -np.sum(probs * log_probs, axis=1)

    @staticmethod
    def feature_dims():
        """Return dict of {feature_id: dimension} for all 48 features."""
        return {
            "A01": 15, "A02": 7, "A03": 30, "A04": 192, "A05": 20,
            "A06": 256, "A07": 32, "A08_A15": 51,
            "B01": MAX_BYTES, "B02": 256, "B03": 256, "B04": 256,
            "B05": 8, "B06": 256, "B07": 6,
            "C01": 256, "C02": 10, "C03": 31, "C04": 256,
            "D01": MAX_OPCODES, "D02": NGRAM_DIM, "D03": NGRAM_DIM,
            "D04": 15, "D05": 12, "D06": 10, "D07": MAX_FUNCTIONS * 128,
            "E03": 12, "E04": 10,
            "F01": IMG_SIZE*IMG_SIZE, "F02": IMG_SIZE*IMG_SIZE*3,
            "F03": 256*256, "F04": IMG_SIZE*IMG_SIZE,
            "G01": 128, "G02": 128, "G03": 128, "G04": 128,
            "H01": 20, "H02": 20, "H03": 20,
        }

# ═══════════════════════════════════════
#  样本扫描工具 (run脚本使用)
# ═══════════════════════════════════════

def scan_load_samples(data_root):
    """
    扫描数据目录, 返回 [(文件路径, 标签)] 列表

    支持目录结构:
      data_root/malware/*.exe  → label=1
      data_root/benign/*.exe   → label=0
    """
    samples = []
    label_dirs = {
        1: ["malware", "malicious", "positive", "mal", "1"],
        0: ["benign", "clean", "negative", "ben", "goodware", "0"],
    }
    for label, dirnames in label_dirs.items():
        for dirname in dirnames:
            d = os.path.join(data_root, dirname)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                if os.path.isfile(fp):
                    samples.append((fp, label))
    if samples:
        print(f"[scan] {data_root}: {len(samples)} samples "
              f"({sum(1 for _,y in samples if y==1)} mal, "
              f"{sum(1 for _,y in samples if y==0)} ben)")
    return samples


def scan_load_prediction_samples(predict_dir):
    """
    扫描预测目录, 返回 [(文件路径, -1)] 列表 (标签未知)
    """
    samples = []
    if not os.path.isdir(predict_dir):
        return samples
    for f in os.listdir(predict_dir):
        fp = os.path.join(predict_dir, f)
        if os.path.isfile(fp):
            samples.append((fp, -1))
    print(f"[scan] {predict_dir}: {len(samples)} prediction samples")
    return samples
