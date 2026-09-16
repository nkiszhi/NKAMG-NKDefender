"""
LLM-Assisted Drift Analysis Module (Paper §4.4)
=================================================
Two modes:
  1. LLM mode: Calls Anthropic Claude API for context-aware threat attribution.
  2. Rule mode (fallback): Deterministic analysis for offline/reproducibility.

Usage:
    result = analyze_drift(sdd_result, view_groups=config.VIEW_GROUPS, mode="auto")
"""
import os, json
import config

_ATTCK_DB = {
    "T1027":     {"name": "Obfuscated Files or Information", "tactics": ["Defense Evasion"]},
    "T1027.002": {"name": "Software Packing", "tactics": ["Defense Evasion"]},
    "T1027.004": {"name": "Compile After Delivery", "tactics": ["Defense Evasion"]},
    "T1027.007": {"name": "Dynamic API Resolution", "tactics": ["Defense Evasion"]},
    "T1036":     {"name": "Masquerading", "tactics": ["Defense Evasion"]},
    "T1036.005": {"name": "Match Legitimate Name or Location", "tactics": ["Defense Evasion"]},
    "T1055":     {"name": "Process Injection", "tactics": ["Defense Evasion", "Privilege Escalation"]},
    "Multiple":  {"name": "Multi-vector evasion", "tactics": ["Defense Evasion"]},
}

_REMEDIATION = {
    "V1_byte":        "Byte-level models affected: retrain MalConv on recent samples.",
    "V2_byte_stat":   "Byte distribution shift: new packer/crypter; update baseline.",
    "V3_pe_struct":   "PE structure changes: header/section manipulation for evasion.",
    "V4_import":      "Import table drift: dynamic API resolution or import obfuscation.",
    "V5_string":      "String features shifted: string encryption/obfuscation detected.",
    "V6_opcode_seq":  "Opcode sequence shifted: new compiler or code mutation engine.",
    "V6_opcode_stat": "Opcode distribution changed: instruction-level polymorphism.",
    "V6_func_embed":  "Function embeddings drifted: structural code reorganization.",
    "V7_graph":       "CFG/call graph topology shifted: control flow flattening.",
    "V8_gray":        "Grayscale visualization changed: binary structure reorganization.",
    "V8_color":       "Color visualization changed: section-level byte redistribution.",
    "V8_markov":      "Markov image shifted: byte-pair transition patterns altered.",
    "V8_entropy":     "Entropy heatmap shifted: packing or encryption changes.",
    "V9_hash":        "Hash features drifted: new file structure patterns.",
    "V10_metadata":   "Metadata shifted: version/resource info spoofing.",
    "V11_ensemble":   "Full-feature drift: broad-spectrum malware evolution.",
}

# ── LLM Analysis (Anthropic Claude API) ──

def _call_llm(sdd_result, view_groups):
    """Call Anthropic Claude API for context-aware drift analysis."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    context = _build_llm_context(sdd_result, view_groups)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=(
            "You are a malware analysis expert. Analyze SDD (Spectral Drift Decomposition) "
            "results to attribute concept drift to adversary techniques (MITRE ATT&CK). "
            "Be technically precise: reference eigenvalues, view loadings, z-scores. "
            "Respond in JSON: {severity, drift_type_explanation, "
            "attck_techniques: [{id, name, confidence, evidence}], "
            "affected_components, defense_strategy, label_budget_rationale, narrative_summary}"
        ),
        messages=[{"role": "user", "content": context}],
    )
    text = response.content[0].text
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except (json.JSONDecodeError, IndexError):
        return {"narrative_summary": text, "parse_error": True}

def _build_llm_context(sdd_result, vg):
    lines = [
        "## SDD Analysis Results",
        f"Drift type: {sdd_result.drift_type}",
        f"Mechanisms (r): {sdd_result.num_mechanisms}",
        f"Spectral gap (lambda1/lambda2): {sdd_result.spectral_gap:.3f}",
        f"Novelty (nu): {sdd_result.novelty:.3f}",
        f"Strategy: {sdd_result.strategy}",
        f"Label budget: {sdd_result.label_budget}",
        "", "## Level Z-scores",
    ]
    for ln, z in (sdd_result.level_scores or {}).items():
        lines.append(f"  {ln}: z={z:.2f}")
    lines += ["", "## Affected Views"]
    for v in sdd_result.affected_views:
        if v in vg:
            attck = vg[v].get("attck", "N/A")
            loading = (sdd_result.view_loadings or {}).get(v, 0)
            lines.append(f"  {v} ({vg[v].get('label', v)}): ATT&CK={attck}, loading={loading:.4f}")
    lines += ["", "## Eigenvalues (top 5)"]
    if sdd_result.eigenvalues is not None:
        for i, ev in enumerate(sdd_result.eigenvalues[:5]):
            sig = " *significant*" if ev > (sdd_result.threshold or 0) else ""
            lines.append(f"  lambda_{i+1} = {ev:.4f}{sig}")
    if sdd_result.drift_explanation:
        lines += ["", f"## Engine explanation: {sdd_result.drift_explanation}"]
    lines += ["", "Analyze and respond in JSON."]
    return "\n".join(lines)

# ── Rule-Based Analysis (deterministic fallback) ──

def _rule_analysis(sdd_result, vg):
    r = sdd_result.num_mechanisms
    gap = sdd_result.spectral_gap   # λ₁/λ₂ (>1 = concentrated, ~1 = diffuse)
    aff = sdd_result.affected_views
    strengths = sdd_result.mechanism_strengths or [1.0] * max(r, 1)
    # gap_ratio: healthy_models / K (0~1)
    gap_ratio = len(sdd_result.healthy_models) / max(len(sdd_result.affected_models) + len(sdd_result.healthy_models), 1)

    # severity 基于漂移类型 + 机制数 + 受影响比例
    dt = sdd_result.drift_type
    if dt == "paradigm_shift" or (r >= 4 and gap_ratio < 0.3):
        severity = "CRITICAL"
    elif dt == "structural" or r >= 2:
        severity = "HIGH"
    elif dt == "perturbation":
        severity = "MEDIUM"
    else:
        severity = "LOW"

    attck = []
    for v in aff:
        if v in vg and "attck" in vg[v]:
            tid = vg[v]["attck"]
            info = _ATTCK_DB.get(tid, {"name": "Unknown", "tactics": []})
            # confidence 从 view_loading 的 z-score 推导
            vl = (sdd_result.view_loadings or {}).get(v, 0)
            max_vl = max((sdd_result.view_loadings or {0: 1}).values())
            conf = min(1.0, vl / max(max_vl, 1e-10)) if max_vl > 0 else 0.5
            attck.append({
                "view": v, "label": vg[v].get("label", v),
                "technique_id": tid, "technique_name": info["name"],
                "tactics": info["tactics"],
                "confidence": conf,
            })

    # strategy 文本基于漂移类型 (与 SDD 策略一致)
    strat_map = {
        "auto_adapt": f"Auto-adapt: {gap_ratio:.0%} models unaffected. Pseudo-label safe.",
        "selective_retrain": f"Selective retrain: {1-gap_ratio:.0%} models affected. ~{sdd_result.label_budget} labels needed.",
        "full_retrain": f"Full retrain: paradigm shift detected. ~{sdd_result.label_budget} labels required.",
        "none": "No action needed.",
    }
    strat = strat_map.get(sdd_result.strategy, f"Strategy: {sdd_result.strategy}")

    mechs = sdd_result.mechanism_views or [aff]
    mech_reports = []
    for i, mv in enumerate(mechs):
        labels = [vg[v].get("label", v) for v in mv if v in vg]
        techs = [vg[v]["attck"] for v in mv if v in vg and "attck" in vg[v]]
        strength = strengths[i] if i < len(strengths) else 0.0
        remed = [_REMEDIATION.get(v, "") for v in mv if v in _REMEDIATION]
        mech_reports.append({
            "mechanism": i+1, "strength": strength, "views": mv,
            "labels": labels, "techniques": techs, "remediation": remed,
            "narrative": f"Mechanism {i+1} (str={strength:.2f}): {', '.join(labels)} ({', '.join(techs)})",
        })
    mech_reports.sort(key=lambda m: m["strength"], reverse=True)

    return {
        "status": "drift_detected", "severity": severity,
        "drift_type": sdd_result.drift_type,
        "num_mechanisms": r, "spectral_gap": gap,
        "confidence": sdd_result.confidence,
        "label_budget": sdd_result.label_budget,
        "strategy": sdd_result.strategy,
        "strategy_rationale": strat,
        "attck_mapping": attck, "mechanism_reports": mech_reports,
        "affected_views": aff,
        "level_scores": sdd_result.level_scores,
        "novelty": sdd_result.novelty,
        "drift_explanation": sdd_result.drift_explanation,
        "analysis_mode": "rule",
    }

# ── Public API ──

def analyze_drift(sdd_result, view_groups=None, mode="auto"):
    """
    Generate threat attribution report from SDD result.
    mode: "auto" (LLM if available, else rules), "llm", "rule"
    """
    if not sdd_result.drift_detected:
        return {"status": "clean", "report": "No drift detected.",
                "attck_mapping": [], "mechanism_reports": [], "analysis_mode": "none"}

    vg = view_groups or getattr(config, 'VIEW_GROUPS', {})
    result = _rule_analysis(sdd_result, vg)

    llm_result = None
    if mode in ("auto", "llm"):
        try:
            llm_result = _call_llm(sdd_result, vg)
            result["llm_analysis"] = llm_result
            result["analysis_mode"] = "llm+rule"
        except Exception as e:
            if mode == "llm": raise
            result["llm_error"] = str(e)

    result["report"] = _format_report(result, llm_result)
    return result

def _format_report(result, llm_result=None):
    lines = [
        f"=== SDD Drift Analysis Report ===",
        f"Type: {result.get('drift_type', '?')} | Severity: {result['severity']}",
        f"Mechanisms: {result['num_mechanisms']} | Gap: {result['spectral_gap']:.2f} | "
        f"Novelty: {result.get('novelty', 0):.3f}",
        f"Strategy: {result['strategy']} | Labels: ~{result['label_budget']}",
    ]
    if result.get("drift_explanation"):
        lines += ["", result["drift_explanation"]]
    lines += ["", "-- Mechanisms --"]
    for m in result.get("mechanism_reports", []):
        lines.append(f"  [{m['mechanism']}] {m['narrative']}")
        for rem in m.get("remediation", []):
            if rem: lines.append(f"      > {rem}")
    lines += ["", "-- ATT&CK --"]
    for a in result.get("attck_mapping", []):
        lines.append(f"  {a.get('label',''):>12s}: {a['technique_id']} {a['technique_name']} "
                      f"({a.get('confidence',0):.0%})")
    lines += ["", "-- Strategy --", f"  {result.get('strategy_rationale', '')}"]
    if llm_result and llm_result.get("narrative_summary"):
        lines += ["", "-- LLM Analysis --", f"  {llm_result['narrative_summary']}"]
    return "\n".join(lines)
