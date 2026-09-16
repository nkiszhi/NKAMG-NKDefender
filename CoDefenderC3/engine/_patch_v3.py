"""
Patch app.py v3 — Align predict_single output with result_formatter standard format.

Patches:
1. Add _SDD_ENGINE global variable
2. Add _get_sdd_engine() lazy loader
3. Add imports for result_formatter, compute_identifiers, detect_packer
4. Replace predict_single return with format_scan_result output
"""
import pathlib
import tempfile
import subprocess
import sys


def main():
    src = pathlib.Path(r"d:\VM_Share\CoDefenderC3_v7\engine\app.py")
    content = src.read_text(encoding="utf-8")
    original_len = len(content)
    lines = content.splitlines()
    print(f"Source: {len(lines)} lines, {original_len} chars")

    changed = False

    # ── PATCH 1: _SDD_ENGINE global variable ──
    marker1 = "_MALGRAPH_RUNTIME: Optional[Dict[str, Any]] = None\n"
    target1 = "_MALGRAPH_RUNTIME: Optional[Dict[str, Any]] = None\n_SDD_ENGINE: Optional[Any] = None\n"
    if marker1 in content:
        content = content.replace(marker1, target1, 1)
        print("[PATCH 1] Added _SDD_ENGINE global variable")
        changed = True
    else:
        print("[PATCH 1] SKIP (already applied or pattern changed)")

    # ── PATCH 2: _get_sdd_engine() before _get_extractor() ──
    marker2 = "\ndef _get_extractor() -> BinjaExtractor:\n"
    if marker2 in content and "_get_sdd_engine" not in content:
        sdd_func = '''
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

'''
        content = content.replace(marker2, sdd_func + marker2, 1)
        print("[PATCH 2] Added _get_sdd_engine() function")
        changed = True
    else:
        print("[PATCH 2] SKIP")

    # ── PATCH 3: Imports for result_formatter etc. ──
    if "from engine.result_formatter import format_scan_result" not in content:
        imp_marker = "from models.ensemble import MultiModelEnsemble, _input_dtype"
        imp_add = (
            "\nfrom engine.result_formatter import format_scan_result\n"
            "from engine.inference import compute_identifiers\n"
        )
        content = content.replace(imp_marker, imp_marker + imp_marker, 1)
        print("[PATCH 3] Added result_formatter + compute_identifiers imports")
        changed = True
    else:
        print("[PATCH 3] SKIP")

    # ── PATCH 4: Replace predict_single return block ──
    # The current return block (lines 803-819):
    #   ok_count = sum(...)
    #   err_count = sum(...)
    #   skip_count = sum(...)
    #   return {
    #       "filename": ...
    #       ...
    #   }
    old_return = '''        ok_count = sum(1 for r in per_model if r["status"] == "ok")
        err_count = sum(1 for r in per_model if r["status"] == "error")
        skip_count = sum(1 for r in per_model if r["status"] in {"skipped", "unavailable"})

        return {
            "filename": file.filename,
            "bytes": len(content),
            "model_path": MODEL_PATH,
            "summary": {
                "total_models": len(per_model),
                "ok": ok_count,
                "error": err_count,
                "skipped_or_unavailable": skip_count,
            },
            "view_status": view_meta,
            "results": per_model,
        }'''

    if old_return in content:
        new_return = '''        # ── Build (1, K) matrices from flat per_model list ──
        ok_results = [r for r in per_model if r.get("status") == "ok"]
        K = len(ok_results)
        model_names = [r["model"] for r in ok_results]
        scores_arr = np.array(
            [r["score"] for r in ok_results], dtype=np.float64,
        ).reshape(1, K) if K > 0 else np.zeros((1, 1))
        preds_arr = np.array(
            [r["prediction"] for r in ok_results], dtype=np.int64,
        ).reshape(1, K) if K > 0 else np.zeros((1, 1), dtype=int)

        # ── File identifiers + packer detection ──
        identifiers = compute_identifiers(temp_path)
        identifiers["FileName"] = file.filename
        identifiers["FileSize"] = len(content)
        from engine.packer_detector import detect_packer
        pack_info = detect_packer(temp_path)

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

        # ── Build evidence dict for format_scan_result ──
        evidence = {
            "predictions": preds_arr,
            "scores": scores_arr,
            "per_model_scores": scores_arr,
            "per_model_preds": preds_arr,
            "drift_evidence": drift_evidence,
            "sample_drift_type": sample_drift_type,
            "anomaly_score": anomaly_score,
            "is_drift_candidate": is_drift_candidate,
        }

        return format_scan_result(
            evidence=evidence,
            identifiers=identifiers,
            pack_info=pack_info,
            model_names=model_names,
        )'''
        content = content.replace(old_return, new_return, 1)
        print("[PATCH 4] Replaced predict_single return with format_scan_result")
        changed = True
    else:
        if "format_scan_result(" in content and "evidence=evidence" in content:
            print("[PATCH 4] SKIP (already applied)")
        else:
            print("[PATCH 4] FAILED — return marker not found!")
            # Debug: show what's actually around line 803
            for i, line in enumerate(content.splitlines()):
                if "ok_count" in line and i > 750:
                    print(f"  Found ok_count at line {i+1}: {line.rstrip()}")
                    for j in range(i, min(i+20, len(content.splitlines()))):
                        print(f"    {j+1}: {content.splitlines()[j]}")
                    break

    if not changed:
        print("\nNo changes needed — file appears already patched.")
        return

    # ── Write to temp, then copy back ──
    tmp = pathlib.Path(tempfile.gettempdir()) / "app_patched.py"
    tmp.write_text(content, encoding="utf-8")
    print(f"\nPatched file written to: {tmp}")
    print(f"  Size: {len(content)} chars (was {original_len})")

    # Try copy back
    result = subprocess.run(
        ["cmd", "/c", "copy", "/y", str(tmp), str(src)],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        print(f"Copy back SUCCESS")
        # Verify
        v = src.read_text(encoding="utf-8")
        for tag in ["_SDD_ENGINE", "_get_sdd_engine", "format_scan_result", "compute_identifiers"]:
            found = tag in v
            print(f"  {tag}: {'OK' if found else 'MISSING'}")
    else:
        print(f"\nCopy back FAILED (VM Share read-only):")
        print(f"  {result.stdout.strip()}")
        print(f"  {result.stderr.strip()}")
        print(f"\n>>> Please manually copy:")
        print(f"  FROM: {tmp}")
        print(f"  TO:   {src}")


if __name__ == "__main__":
    main()
