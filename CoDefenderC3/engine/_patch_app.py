"""
_patch_app.py — 给 engine/app.py 添加 _SDD_ENGINE 懒加载并改造 predict_single 输出格式

使用方法: 在 app.py 未被占用时执行:
    python engine/_patch_app.py
"""
import pathlib

BASE = pathlib.Path(__file__).resolve().parent


def main():
    p = BASE / "app.py"
    c = p.read_text(encoding="utf-8")

    # ═══ Step 1: 添加 _SDD_ENGINE 全局变量 ═══
    old1 = "_MALGRAPH_RUNTIME: Optional[Dict[str, Any]] = None\n"
    new1 = "_MALGRAPH_RUNTIME: Optional[Dict[str, Any]] = None\n_SDD_ENGINE: Optional[Any] = None\n"
    if new1 in c:
        print("[skip] _SDD_ENGINE already exists")
    else:
        assert old1 in c, "old1 not found"
        c = c.replace(old1, new1, 1)
        print("[ok] added _SDD_ENGINE global")

    # ═══ Step 2: 添加 _get_sdd_engine() 函数 ═══
    marker = "def _get_extractor() -> BinjaExtractor:\n"
    if "_get_sdd_engine" in c:
        print("[skip] _get_sdd_engine already exists")
    else:
        assert marker in c, "marker not found"
        func = (
            "def _get_sdd_engine():\n"
            '    """\n'
            "    Lazy-load SDDEngine (uncalibrated; score_samples returns zero drift evidence).\n"
            '    """\n'
            "    global _SDD_ENGINE\n"
            "    if _SDD_ENGINE is not None:\n"
            "        return _SDD_ENGINE\n"
            "\n"
            "    with _STATE_LOCK:\n"
            "        if _SDD_ENGINE is not None:\n"
            "            return _SDD_ENGINE\n"
            "\n"
            "        from core.sdd_engine import SDDEngine\n"
            "        ens = _load_ensemble()\n"
            "        _SDD_ENGINE = SDDEngine(\n"
            "            K=ens.K,\n"
            "            perspective_groups=ens.perspective_groups(),\n"
            "        )\n"
            "        return _SDD_ENGINE\n"
            "\n"
            "\n"
        )
        c = c.replace(marker, func + marker, 1)
        print("[ok] added _get_sdd_engine()")

    # ═══ Step 3: 改造 predict_single 输出格式 ═══
    old_return = """\
        ok_count = sum(1 for r in per_model if r["status"] == "ok")
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
        }"""

    new_return = """\
        # -- file identifiers & packer detection --
        from engine.inference import compute_identifiers
        from engine.packer_detector import detect_packer
        identifiers = compute_identifiers(temp_path)
        pack_info = detect_packer(temp_path)

        # -- convert List[Dict] -> (1, K) numpy matrices --
        model_names = ens._model_names  # noqa: SLF001
        per_model_scores = np.array([
            r.get("score", 0.5) if r["status"] == "ok" else 0.5
            for r in per_model
        ], dtype=np.float32).reshape(1, -1)
        per_model_preds = np.array([
            r.get("prediction", 0) if r["status"] == "ok" else 0
            for r in per_model
        ], dtype=np.float32).reshape(1, -1)

        # -- ensemble prediction --
        ens_preds, ens_scores = ens.ensemble_predict(per_model_scores)

        # -- drift evidence (zero values when uncalibrated) --
        sdd = _get_sdd_engine()
        drift = sdd.score_samples(per_model_scores)

        # -- assemble evidence -> format_scan_result --
        from engine.result_formatter import format_scan_result
        evidence = {
            "predictions": ens_preds,
            "scores": ens_scores,
            "per_model_scores": per_model_scores,
            "per_model_preds": per_model_preds,
            "drift_evidence": drift["drift_evidence"],
            "sample_drift_type": drift["sample_drift_type"],
            "anomaly_score": drift["anomaly_score"],
            "is_drift_candidate": drift["is_drift_candidate"],
        }

        return format_scan_result(
            evidence=evidence,
            identifiers=identifiers,
            pack_info=pack_info,
            model_names=model_names,
        )"""

    if "format_scan_result" in c:
        print("[skip] predict_single already uses format_scan_result")
    else:
        assert old_return in c, "old return block not found"
        c = c.replace(old_return, new_return, 1)
        print("[ok] replaced predict_single return with format_scan_result")

    # ═══ Write ═══
    p.write_text(c, encoding="utf-8")
    print(f"[done] app.py updated ({len(c)} chars)")

    # cleanup self
    # try:
    #     __import__("os").remove(__file__)
    # except OSError:
    #     pass


if __name__ == "__main__":
    main()
