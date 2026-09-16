"""Patch app.py: add _SDD_ENGINE + _get_sdd_engine() + fix predict_single output format.
Writes patched content to a temp file, then uses cmd /c copy /y to overwrite.
"""
import pathlib
import tempfile
import shutil
import subprocess
import sys

def main():
    src = pathlib.Path(r"d:\VM_Share\CoDefenderC3_v7\engine\app.py")

    # Step 1: read source
    content = src.read_text(encoding="utf-8")
    print(f"Read source: {len(content)} chars, {content.count(chr(10))+1} lines")

    # Step 2: check if already patched
    if "_SDD_ENGINE" in content:
        print("SKIP: _SDD_ENGINE already exists. Checking predict_single...")

    # --- PATCH 1: Add _SDD_ENGINE global variable ---
    old1 = "_MALGRAPH_RUNTIME: Optional[Dict[str, Any]] = None"
    new1 = "_MALGRAPH_RUNTIME: Optional[Dict[str, Any]] = None\n_SDD_ENGINE: Optional[Any] = None"
    if old1 in content and "_SDD_ENGINE" not in content:
        content = content.replace(old1, new1, 1)
        print("Patch 1 applied: _SDD_ENGINE global var")
    else:
        print("Patch 1 skipped (already applied or pattern not found)")

    # --- PATCH 2: Add _get_sdd_engine() before _get_extractor ---
    old2 = "def _get_extractor() -> BinjaExtractor:"
    if old2 in content and "_get_sdd_engine" not in content:
        sdd_func = '''def _get_sdd_engine():
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
        content = content.replace(old2, sdd_func + old2, 1)
        print("Patch 2 applied: _get_sdd_engine() function")
    else:
        print("Patch 2 skipped (already applied or pattern not found)")

    # --- PATCH 3: Add imports for result_formatter, identifiers, packer ---
    if "from engine.result_formatter import format_scan_result" not in content:
        # Find the last import line and add after it
        import_block_end = content.find("\n_STATE_LOCK")
        if import_block_end > 0:
            new_imports = (
                "\nfrom engine.result_formatter import format_scan_result\n"
                "from engine.inference import compute_identifiers\n"
                "from engine.packer_detector import detect_packer\n"
            )
            content = content[:import_block_end] + new_imports + content[import_block_end:]
            print("Patch 3 applied: new imports (result_formatter, compute_identifiers, detect_packer)")
        else:
            print("Patch 3 FAILED: could not find insertion point for imports")
    else:
        print("Patch 3 skipped (imports already present)")

    # --- PATCH 4: Modify predict_single return value ---
    # Find the predict_single endpoint function and modify its return
    old_return_marker = 'return {"filename": filename, "bytes": len(file_bytes),'
    if old_return_marker in content:
        # Build the new return block
        new_return = '''# --- Build per_model matrices (1, K) from flat result list ---
        model_names = [r["model"] for r in results if r.get("status") == "ok"]
        K = len(model_names)
        scores_arr = numpy.array([r["score"] for r in results if r.get("status") == "ok"], dtype=numpy.float64).reshape(1, K) if K > 0 else numpy.zeros((1, 1))
        preds_arr = numpy.array([r["prediction"] for r in results if r.get("status") == "ok"]).reshape(1, K) if K > 0 else numpy.zeros((1, 1), dtype=int)
        per_model_scores = dict(zip(model_names, scores_arr[0].tolist())) if K > 0 else {}
        per_model_preds = dict(zip(model_names, preds_arr[0].tolist())) if K > 0 else {}

        # --- Compute identifiers and packer ---
        identifiers = compute_identifiers(file_bytes)
        packer_info = detect_packer(file_bytes)

        # --- SDD drift evidence ---
        try:
            sdd = _get_sdd_engine()
            drift_ev = sdd.score_samples(scores_arr)
            drift_evidence = drift_ev.get("drift_evidence", {})
            sample_drift_type = drift_ev.get("sample_drift_type", "normal")
            anomaly_score = drift_ev.get("anomaly_score", 0.0)
            is_drift_candidate = drift_ev.get("is_drift_candidate", False)
        except Exception as exc:
            drift_evidence = {}
            sample_drift_type = "normal"
            anomaly_score = 0.0
            is_drift_candidate = False

        # --- Final ensemble prediction ---
        final_pred = preds_arr[0].mean() >= 0.5 if K > 0 else False
        final_score = float(scores_arr[0].mean()) if K > 0 else 0.0

        evidence = {
            "predictions": preds_arr,
            "scores": scores_arr,
            "per_model_scores": per_model_scores,
            "per_model_preds": per_model_preds,
            "drift_evidence": drift_evidence,
            "sample_drift_type": sample_drift_type,
            "anomaly_score": anomaly_score,
            "is_drift_candidate": is_drift_candidate,
        }

        result = format_scan_result(
            identifiers=identifiers,
            filename=filename,
            file_size=len(file_bytes),
            predictions=preds_arr,
            scores=scores_arr,
            evidence=evidence,
            packer_info=packer_info,
        )
        return result'''

        content = content.replace(old_return_marker, new_return, 1)
        print("Patch 4 applied: predict_single return replaced with format_scan_result")
    else:
        # Check if already patched
        if "format_scan_result(" in content:
            print("Patch 4 skipped (predict_single already uses format_scan_result)")
        else:
            print("Patch 4 FAILED: could not find return marker in predict_single")

    # Add numpy import if missing
    if "import numpy" not in content:
        content = content.replace(
            "import numpy",
            "import numpy", 1
        )
        # Try inserting near top imports
        first_import = content.find("from ")
        if first_import > 0:
            content = "import numpy\n" + content
            print("Patch 5 applied: added 'import numpy'")

    # Write to temp file first
    tmp = pathlib.Path(tempfile.gettempdir()) / "app_patched.py"
    tmp.write_text(content, encoding="utf-8")
    print(f"Wrote patched file to temp: {tmp} ({len(content)} chars)")

    # Copy back using cmd /c copy /y
    result = subprocess.run(
        ["cmd", "/c", "copy", "/y", str(tmp), str(src)],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        print(f"Successfully copied patched file back to {src}")
        # Verify
        verify = src.read_text(encoding="utf-8")
        print(f"Verification - _SDD_ENGINE: {'found' if '_SDD_ENGINE' in verify else 'MISSING'}")
        print(f"Verification - _get_sdd_engine: {'found' if '_get_sdd_engine' in verify else 'MISSING'}")
        print(f"Verification - format_scan_result: {'found' if 'format_scan_result' in verify else 'MISSING'}")
        print(f"Verification - compute_identifiers: {'found' if 'compute_identifiers' in verify else 'MISSING'}")
    else:
        print(f"Copy back FAILED (exit={result.returncode}):")
        print(f"  stdout: {result.stdout.strip()}")
        print(f"  stderr: {result.stderr.strip()}")
        print(f"\nPatched file is at: {tmp}")
        print("Please manually copy it to the target location.")

if __name__ == "__main__":
    main()
