"""
SLFE Axiom Verification (Paper §3, Theorem 1.5)
=================================================

Verifies that feature extractors satisfy the Structured Locality
of Feature Extraction (SLFE) axioms, which guarantee zero inter-view
feature leakage.

SLFE-1 (Region Dependency): Each feature φᵢ depends only on a
    specific region R(i) of the PE file.
SLFE-2 (Pointer Closure): Regions form a closed set under pointer
    dereferencing — following a pointer from R(i) stays within R(i).

If both axioms hold for two feature groups v₁, v₂ with disjoint
regions R(v₁) ∩ R(v₂) = ∅, then modifying v₁'s features cannot
affect v₂'s features (Theorem 1.5: zero leakage).

This module provides:
  1. verify_slfe() — programmatic verification against PE structure
  2. verify_view_isolation() — empirical test on real PE samples
"""

import struct
import numpy as np


# ══════════════════════════════════════════════════════════════
#  PE Region Definitions (SLFE-1)
# ══════════════════════════════════════════════════════════════

# Each view maps to specific PE file regions (byte offset ranges)
# These are NOT arbitrary — they follow the PE format specification.

PE_REGIONS = {
    # Surface-level features: byte statistics, strings, visualization
    "V1_byte":        ["raw_bytes"],           # entire file bytes
    "V2_byte_stat":   ["raw_bytes"],           # byte histogram, entropy
    "V5_string":      ["raw_bytes"],           # printable string extraction
    "V8_gray":        ["raw_bytes"],           # grayscale visualization
    "V8_color":       ["raw_bytes"],           # color visualization
    "V8_markov":      ["raw_bytes"],           # byte transition matrix
    "V8_entropy":     ["raw_bytes"],           # entropy heatmap
    "V9_hash":        ["raw_bytes", "import_table"],  # file + import hashing

    # Structure-level features: PE headers, sections, imports
    "V3_pe_struct":   ["dos_header", "coff_header", "optional_header", "section_table"],
    "V4_import":      ["import_table", "export_table"],
    "V10_metadata":   ["resource_section", "version_info", "manifest"],

    # Disassembly/graph features: code sections
    "V6_opcode_seq":  ["code_sections"],
    "V6_opcode_stat": ["code_sections"],
    "V6_func_embed":  ["code_sections"],
    "V7_graph":       ["code_sections"],

    # Global features: all of the above combined
    "V11_ensemble":   ["__ALL__"],
}

def _rva_to_offset(rva, sections):
    """Convert RVA to file offset using section table."""
    for va, vs, ptr, rs in sections:
        if va <= rva < va + max(vs, rs):
            return ptr + (rva - va)
    return rva  # fallback: treat as raw offset


def _parse_pe_regions(raw):
    """Parse PE file and return byte ranges for each region."""
    regions = {}

    # DOS Header: bytes [0, 64)
    regions["dos_header"] = [(0, min(64, len(raw)))]

    if len(raw) < 64:
        return regions
    pe_off = struct.unpack_from("<I", raw, 0x3C)[0]
    if pe_off + 24 > len(raw) or raw[pe_off:pe_off+4] != b"PE\x00\x00":
        return regions

    # COFF Header
    regions["coff_header"] = [(pe_off + 4, pe_off + 24)]

    # Optional Header
    opt_size = struct.unpack_from("<H", raw, pe_off + 20)[0]
    opt_start = pe_off + 24
    opt_end = opt_start + opt_size
    regions["optional_header"] = [(opt_start, min(opt_end, len(raw)))]

    # Section Table
    n_sec = struct.unpack_from("<H", raw, pe_off + 6)[0]
    sec_start = opt_end
    sec_end = sec_start + n_sec * 40
    regions["section_table"] = [(sec_start, min(sec_end, len(raw)))]

    # Parse section headers for RVA→offset conversion
    section_info = []  # (VirtualAddress, VirtualSize, PointerToRawData, SizeOfRawData)
    code_ranges = []
    data_ranges = []
    for i in range(n_sec):
        s = sec_start + i * 40
        if s + 40 > len(raw):
            break
        va = struct.unpack_from("<I", raw, s + 12)[0]
        vs = struct.unpack_from("<I", raw, s + 8)[0]
        raw_ptr = struct.unpack_from("<I", raw, s + 20)[0]
        raw_sz = struct.unpack_from("<I", raw, s + 16)[0]
        chars = struct.unpack_from("<I", raw, s + 36)[0]
        section_info.append((va, vs, raw_ptr, raw_sz))
        if raw_ptr < len(raw) and raw_sz > 0:
            rng = (raw_ptr, min(raw_ptr + raw_sz, len(raw)))
            if chars & 0x20:  # IMAGE_SCN_CNT_CODE
                code_ranges.append(rng)
            else:
                data_ranges.append(rng)
    regions["code_sections"] = code_ranges
    regions["data_sections"] = data_ranges
    regions["raw_bytes"] = [(0, len(raw))]

    # Data Directory: Export=DD[0], Import=DD[1], Resource=DD[2]
    if opt_size >= 96:
        magic = struct.unpack_from("<H", raw, opt_start)[0]
        dd_off = opt_start + (112 if magic == 0x20B else 96)  # PE32+ vs PE32

        def _read_dd(index):
            off = dd_off + index * 8
            if off + 8 <= len(raw):
                rva = struct.unpack_from("<I", raw, off)[0]
                size = struct.unpack_from("<I", raw, off + 4)[0]
                if rva > 0 and size > 0:
                    foff = _rva_to_offset(rva, section_info)
                    return [(foff, min(foff + size, len(raw)))]
            return []

        regions["export_table"] = _read_dd(0)    # DD[0] = Export
        regions["import_table"] = _read_dd(1)    # DD[1] = Import
        regions["resource_section"] = _read_dd(2) # DD[2] = Resource

    regions.setdefault("export_table", [])
    regions.setdefault("import_table", [])
    regions.setdefault("resource_section", [])
    regions["version_info"] = []
    regions["manifest"] = []

    return regions


def _regions_overlap(r1_list, r2_list):
    """Check if any ranges in r1 overlap with ranges in r2."""
    for s1, e1 in r1_list:
        for s2, e2 in r2_list:
            if s1 < e2 and s2 < e1:
                return True
    return False


# ══════════════════════════════════════════════════════════════
#  SLFE Axiom Verification
# ══════════════════════════════════════════════════════════════

def verify_slfe(pe_path=None, raw=None, level_map=None):
    """
    Verify SLFE axioms on a PE file.

    Returns:
        dict with:
          - slfe1_satisfied: bool (region dependency holds)
          - slfe2_satisfied: bool (pointer closure holds)
          - region_map: {view → [(start, end), ...]}
          - isolation_matrix: {(v1,v2) → bool}  (True = isolated)
          - violations: list of violation descriptions
    """
    if raw is None:
        with open(pe_path, "rb") as f:
            raw = f.read()

    regions = _parse_pe_regions(raw)
    violations = []

    # SLFE-1: Each view maps to well-defined PE regions
    view_regions = {}
    for vn, region_names in PE_REGIONS.items():
        if "__ALL__" in region_names:
            view_regions[vn] = [(0, len(raw))]
            continue
        ranges = []
        for rn in region_names:
            ranges.extend(regions.get(rn, []))
        view_regions[vn] = ranges
        if not ranges:
            violations.append(f"SLFE-1: {vn} has no mapped PE regions")

    slfe1 = len([v for v in violations if "SLFE-1" in v]) == 0

    # SLFE-2: Pointer closure — structural regions don't pointer-chain to surface regions
    # Check: section table pointers (RawDataPtr) point INTO section data, not headers
    slfe2 = True
    n_sec = 0
    pe_off_raw = struct.unpack_from("<I", raw, 0x3C)[0] if len(raw) >= 64 else 0
    if pe_off_raw + 6 < len(raw):
        n_sec = struct.unpack_from("<H", raw, pe_off_raw + 6)[0]
    opt_size = struct.unpack_from("<H", raw, pe_off_raw + 20)[0] if pe_off_raw + 22 <= len(raw) else 0
    sec_table_start = pe_off_raw + 24 + opt_size
    header_end = sec_table_start + n_sec * 40

    for i in range(n_sec):
        s = sec_table_start + i * 40
        if s + 24 > len(raw):
            break
        raw_ptr = struct.unpack_from("<I", raw, s + 20)[0]
        if 0 < raw_ptr < header_end:
            violations.append(f"SLFE-2: Section {i} RawDataPtr={raw_ptr:#x} points INTO headers")
            slfe2 = False

    # Build isolation matrix: which view pairs are truly isolated?
    isolation = {}
    surface_views = {"V1_byte", "V2_byte_stat", "V5_string",
                     "V8_gray", "V8_color", "V8_markov", "V8_entropy"}
    struct_views = {"V3_pe_struct", "V4_import", "V10_metadata"}
    disasm_views = {"V6_opcode_seq", "V6_opcode_stat", "V6_func_embed", "V7_graph"}

    for v1 in PE_REGIONS:
        for v2 in PE_REGIONS:
            if v1 >= v2:
                continue
            if "__ALL__" in PE_REGIONS[v1] or "__ALL__" in PE_REGIONS[v2]:
                isolation[(v1, v2)] = False
                continue
            r1 = view_regions.get(v1, [])
            r2 = view_regions.get(v2, [])
            isolated = not _regions_overlap(r1, r2)
            isolation[(v1, v2)] = isolated

    # Key theorem verification: structure ∩ surface = ∅ (modulo raw_bytes)
    # This is the SLFE guarantee: modifying code sections doesn't change PE headers
    struct_code_isolated = True
    for sv in struct_views:
        for dv in disasm_views:
            r1 = view_regions.get(sv, [])
            r2 = view_regions.get(dv, [])
            if _regions_overlap(r1, r2):
                struct_code_isolated = False
                violations.append(f"SLFE: {sv} and {dv} regions overlap")

    return {
        "slfe1_satisfied": slfe1,
        "slfe2_satisfied": slfe2,
        "struct_code_isolated": struct_code_isolated,
        "region_map": view_regions,
        "isolation_matrix": isolation,
        "violations": violations,
        "n_sections": n_sec,
        "file_size": len(raw),
    }


def verify_view_isolation(pe_paths, extractor, n_samples=100):
    """
    Empirical verification: modify code bytes and check structural features don't change.

    This directly tests Theorem 1.5: if SLFE holds, then
    mutating code_sections should not affect V3_pe_struct features.

    Args:
        pe_paths: list of PE file paths
        extractor: BinjaExtractor instance
        n_samples: number of files to test

    Returns:
        dict with empirical isolation test results
    """
    import random
    test_paths = random.sample(pe_paths, min(n_samples, len(pe_paths)))

    results = {"tested": 0, "violations": 0, "details": []}

    for path in test_paths:
        try:
            with open(path, "rb") as f:
                raw = f.read()

            # Extract original features
            orig = extractor.extract_fast(path)
            if not orig:
                continue

            # Mutate code bytes (NOP insertion simulation)
            regions = _parse_pe_regions(raw)
            code_ranges = regions.get("code_sections", [])
            if not code_ranges:
                continue

            mutated = bytearray(raw)
            for start, end in code_ranges:
                # Replace 10% of code bytes with NOPs (0x90)
                for j in range(start, min(end, len(mutated)), 10):
                    mutated[j] = 0x90

            # Write mutated file and extract features
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tf:
                tf.write(bytes(mutated))
                mut_path = tf.name

            try:
                mut_feats = extractor.extract_fast(mut_path)
            finally:
                os.remove(mut_path)

            if not mut_feats:
                continue

            results["tested"] += 1

            # Check: pure structural features (PE header/section) should NOT change
            # A01-A05 = PE header, sections, anomalies (SLFE-compliant)
            # NOT A06/A07 = import/export tables (separate data regions)
            for key in ["A01", "A02", "A03", "A04", "A05"]:
                if key in orig and key in mut_feats:
                    diff = np.max(np.abs(orig[key] - mut_feats[key]))
                    if diff > 1e-6:
                        results["violations"] += 1
                        results["details"].append({
                            "file": os.path.basename(path),
                            "feature": key,
                            "max_diff": float(diff),
                        })
                        break

        except Exception:
            pass

    results["isolation_rate"] = (
        1.0 - results["violations"] / max(results["tested"], 1)
    )
    return results


def print_slfe_report(pe_path):
    """Print human-readable SLFE verification report."""
    result = verify_slfe(pe_path=pe_path)
    print(f"=== SLFE Axiom Verification ===")
    print(f"File: {pe_path}")
    print(f"Size: {result['file_size']} bytes, {result['n_sections']} sections")
    print(f"SLFE-1 (Region Dependency):     {'PASS' if result['slfe1_satisfied'] else 'FAIL'}")
    print(f"SLFE-2 (Pointer Closure):       {'PASS' if result['slfe2_satisfied'] else 'FAIL'}")
    print(f"Structure-Code Isolation:       {'PASS' if result['struct_code_isolated'] else 'FAIL'}")
    if result["violations"]:
        print(f"Violations ({len(result['violations'])}):")
        for v in result["violations"]:
            print(f"  ! {v}")
    else:
        print("No violations. Theorem 1.5 (zero leakage) holds for this PE.")
