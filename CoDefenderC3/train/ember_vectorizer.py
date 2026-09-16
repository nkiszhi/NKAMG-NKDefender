"""
EMBER 纯 Numpy 特征向量化器
============================
基于 EMBER JSONL 真实数据结构，将样本 dict 转换为 2381d float32 向量。

子特征        维度   数据源
────────────────────────────
ByteHistogram  256   histogram — 256 个整数
ByteEntropy    256   byteentropy — 256 个浮点数
StringInfo     104   strings — 8 标量 + printabledist(96)
GeneralInfo     10   general — 10 个标量
HeaderInfo      62   header.coff + optional — 混合类型
SectionInfo    255   section.sections — 最多 10 个
ImportsInfo   1280   imports — dll(256d) + func(1024d) hash
ExportsInfo    128   exports — 函数名 hash
DataDirs        30   datadirectories — 15 × (size, vaddr)
────────────────────────────
总计          2381

零 sklearn/lief 依赖。
"""
import hashlib
import numpy as np


def _fast_hash(s, n_features):
    """字符串 hash → 桶索引 + 符号"""
    h = int(hashlib.md5(s.encode("utf-8", errors="replace")).hexdigest()[:8], 16)
    return h % n_features, (1 if h & 0x80000000 else -1)


def _hash_strings(strings, n_features):
    """hashing trick: 字符串列表 → n_features 维向量"""
    vec = np.zeros(n_features, dtype=np.float32)
    for s in strings:
        if not s:
            continue
        col, sign = _fast_hash(s, n_features)
        vec[col] += sign
    return vec


def _sf(val):
    """safe float: 数字直接转, 字符串/列表 hash, 其他 0"""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, str):
        return float(int(hashlib.md5(val.encode()).hexdigest()[:8], 16) % 65536)
    if isinstance(val, (list, tuple)):
        s = ",".join(str(x) for x in val)
        return float(int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 65536)
    return 0.0


class PureNumpyFeatureExtractor:
    """纯 Numpy EMBER 特征提取器 (feature v2, 2381d)"""
    dim = 2381

    def process_raw_features(self, sample):
        parts = []

        # 0. ByteHistogram (256d)
        h = sample.get("histogram")
        if isinstance(h, list) and len(h) >= 256:
            parts.append(np.array(h[:256], dtype=np.float32))
        else:
            parts.append(np.zeros(256, dtype=np.float32))

        # 1. ByteEntropy (256d)
        b = sample.get("byteentropy")
        if isinstance(b, list) and len(b) >= 256:
            parts.append(np.array(b[:256], dtype=np.float32))
        else:
            parts.append(np.zeros(256, dtype=np.float32))

        # 2. StringInfo (104d): 8 标量 + printabledist(96)
        st = sample.get("strings", {})
        if not isinstance(st, dict): st = {}
        sv = np.zeros(104, dtype=np.float32)
        sv[0] = _sf(st.get("numstrings", 0))
        sv[1] = _sf(st.get("avlength", 0))
        sv[2] = _sf(st.get("printables", 0))
        sv[3] = _sf(st.get("entropy", 0))
        sv[4] = _sf(st.get("paths", 0))
        sv[5] = _sf(st.get("urls", 0))
        sv[6] = _sf(st.get("registry", 0))
        sv[7] = _sf(st.get("MZ", 0))
        pd = st.get("printabledist")
        if isinstance(pd, list):
            n = min(len(pd), 96)
            for j in range(n):
                v = pd[j]
                sv[8+j] = float(v) if isinstance(v, (int, float)) else 0.0
        parts.append(sv)

        # 3. GeneralFileInfo (10d)
        g = sample.get("general", {})
        if not isinstance(g, dict): g = {}
        parts.append(np.array([
            _sf(g.get("size", 0)), _sf(g.get("vsize", 0)),
            _sf(g.get("has_debug", 0)), _sf(g.get("exports", 0)),
            _sf(g.get("imports", 0)), _sf(g.get("has_relocations", 0)),
            _sf(g.get("has_resources", 0)), _sf(g.get("has_signature", 0)),
            _sf(g.get("has_tls", 0)), _sf(g.get("symbols", 0)),
        ], dtype=np.float32))

        # 4. HeaderFileInfo (62d)
        hdr = sample.get("header", {})
        if not isinstance(hdr, dict): hdr = {}
        coff = hdr.get("coff", {})
        if not isinstance(coff, dict): coff = {}
        opt = hdr.get("optional", {})
        if not isinstance(opt, dict): opt = {}
        hv = np.zeros(62, dtype=np.float32)
        i = 0
        for f in ["timestamp", "machine", "characteristics"]:
            hv[i] = _sf(coff.get(f, 0)); i += 1
        for f in ["subsystem", "dll_characteristics", "magic",
                   "major_image_version", "minor_image_version",
                   "major_linker_version", "minor_linker_version",
                   "major_operating_system_version", "minor_operating_system_version",
                   "major_subsystem_version", "minor_subsystem_version",
                   "sizeof_code", "sizeof_headers", "sizeof_heap_commit"]:
            hv[i] = _sf(opt.get(f, 0)); i += 1
        parts.append(hv)

        # 5. SectionInfo (255d)
        sec = sample.get("section", {})
        if not isinstance(sec, dict): sec = {}
        secs = sec.get("sections", [])
        if not isinstance(secs, list): secs = []
        sv2 = np.zeros(255, dtype=np.float32)
        sv2[0] = float(len(secs))
        entry = sec.get("entry", "")
        if isinstance(entry, str) and entry:
            sv2[1] = _sf(entry)
        for si, s in enumerate(secs[:10]):
            if not isinstance(s, dict): continue
            base = 5 + si * 25
            if base + 25 > 255: break
            sv2[base] = _sf(s.get("size", 0))
            sv2[base+1] = _sf(s.get("entropy", 0))
            sv2[base+2] = _sf(s.get("vsize", 0))
            nm = s.get("name", "")
            if isinstance(nm, str): sv2[base+3] = _sf(nm)
            props = s.get("props", [])
            if isinstance(props, list):
                for pi, p in enumerate(props[:21]):
                    sv2[base+4+pi] = _sf(p)
        parts.append(sv2)

        # 6. ImportsInfo (1280d): dll names(256d) + func names(1024d)
        imp = sample.get("imports", {})
        if not isinstance(imp, dict): imp = {}
        dlls, funcs = [], []
        for dll, flist in imp.items():
            if isinstance(dll, str): dlls.append(dll.lower())
            if isinstance(flist, list):
                for fn in flist:
                    if isinstance(fn, str): funcs.append(fn.lower())
        iv = np.zeros(1280, dtype=np.float32)
        iv[:256] = _hash_strings(dlls, 256)
        iv[256:] = _hash_strings(funcs, 1024)
        parts.append(iv)

        # 7. ExportsInfo (128d)
        exp = sample.get("exports", [])
        if not isinstance(exp, list): exp = []
        parts.append(_hash_strings([str(e).lower() for e in exp if e], 128))

        # 8. DataDirectories (30d)
        dd = sample.get("datadirectories", [])
        if not isinstance(dd, list): dd = []
        dv = np.zeros(30, dtype=np.float32)
        for di, d in enumerate(dd[:15]):
            if isinstance(d, dict):
                dv[di*2] = _sf(d.get("size", 0))
                dv[di*2+1] = _sf(d.get("virtual_address", 0))
        parts.append(dv)

        result = np.concatenate(parts)
        assert len(result) == self.dim, f"dim {len(result)} != {self.dim}"
        return result
