from __future__ import annotations

import json
from typing import Any

import numpy as np

from .client import OllamaClient


def _finite(value: Any):
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def build_xai_evidence(results: list[dict[str, Any]], framework: str) -> dict[str, Any]:
    selected = next((r for r in results if str(r.get("framework")) == str(framework)), None)
    if selected is None:
        raise ValueError("Selected framework is not in current run results.")
    exp = dict(selected.get("explainability") or {})
    features = list(exp.get("feature_importance") or [])[:10]
    scalar_params = {}
    for key, value in dict(selected.get("parameters") or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            scalar_params[str(key)] = value
    system = []
    for row in results:
        e = dict(row.get("explainability") or {})
        system.append({
            "framework": row.get("framework"),
            "backend": row.get("backend"),
            "accuracy": _finite(row.get("accuracy")),
            "f1_macro": _finite(row.get("f1_macro")),
            "method": e.get("method"),
            "status": e.get("status"),
            "fidelity": _finite(e.get("fidelity")),
            "stability": _finite(e.get("stability")),
            "consistency": _finite(e.get("consistency")),
            "sensitivity": _finite(e.get("sensitivity")),
            "sparsity": _finite(e.get("sparsity")),
        })
    return {
        "selected": {
            "framework": selected.get("framework"),
            "backend": selected.get("backend"),
            "accuracy": _finite(selected.get("accuracy")),
            "f1_macro": _finite(selected.get("f1_macro")),
            "prediction_diagnostics": selected.get("prediction_diagnostics") or {},
            "xai": {
                "status": exp.get("status"),
                "method": exp.get("method"),
                "fidelity": _finite(exp.get("fidelity")),
                "stability": _finite(exp.get("stability")),
                "consistency": _finite(exp.get("consistency")),
                "sensitivity": _finite(exp.get("sensitivity")),
                "sparsity": _finite(exp.get("sparsity")),
                "multilevel_diagnostics": exp.get("multilevel_diagnostics") or {},
                "top_features": features,
                "method_metadata": exp.get("method_metadata") or {},
            },
            "parameters": scalar_params,
        },
        "system_comparison": system,
        "scientific_boundaries": [
            "Feature importance describes recent final-model replay, not the full prequential history.",
            "Deletion/sufficiency replacement can be out-of-distribution and correlated features can mask effects.",
            "Consistency/stability do not prove internal faithfulness or causal reasoning.",
            "Missing XAI evidence must remain unavailable rather than being converted to zero.",
        ],
    }


def deterministic_xai_summary(evidence: dict[str, Any]) -> str:
    """Human-readable fallback grounded only in recorded evidence."""
    s = evidence["selected"]
    x = s["xai"]
    top = x.get("top_features") or []
    top_rows = [row for row in top[:3] if isinstance(row, dict) and row.get("feature") is not None]
    if top_rows:
        feature_text = ", ".join(
            "**{}** ({:.3g})".format(row.get("feature"), float(row.get("importance", 0.0)))
            for row in top_rows
        )
    else:
        feature_text = "No trustworthy feature ranking was recorded for this run."

    def metric(key: str, digits: int = 3) -> str:
        value = x.get(key)
        return "N/A" if value is None else ("{:.%df}" % digits).format(float(value))

    available = [r for r in evidence["system_comparison"] if r.get("status") == "ok"]
    methods = sorted({str(r.get("method")) for r in available if r.get("method")})
    method_text = ", ".join(methods) if methods else "no trustworthy method recorded"

    return (
        "### What the current explanation says\n"
        "For **{}**, AwareML recorded **{}** as the explanation method. The strongest recent-window features were {}. "
        "These are feature-attribution signals for the replayed recent window; they are not causal effects.\n\n"
        "### How stable the explanation looked\n"
        "Deletion fidelity was **{}** (the replay accuracy drop after deleting important features), stability was **{}**, "
        "consistency was **{}**, sensitivity was **{}** *(lower is better)*, and sparsity was **{}**. "
        "These diagnostics describe different properties, so they should not be collapsed into one invented quality score.\n\n"
        "### System-level context\n"
        "Trustworthy XAI evidence is available for **{} of {}** frameworks in this run. Recorded methods include **{}**. "
        "Cross-framework comparisons are most meaningful when the method and replay conditions are comparable.\n\n"
        "### What this does not prove\n"
        "The explanation is based on final/recent-window replay, not the complete prequential history. Feature deletion can create out-of-distribution inputs, correlated features can share importance, and stability/consistency do not reveal hidden model or LLM reasoning."
    ).format(
        s.get("framework"),
        x.get("method") or "an unavailable method",
        feature_text,
        metric("fidelity"),
        metric("stability"),
        metric("consistency"),
        metric("sensitivity"),
        metric("sparsity"),
        len(available),
        len(evidence["system_comparison"]),
        method_text,
    )


def grounded_xai_summary(results: list[dict[str, Any]], framework: str, model: str, use_llm: bool = True):
    evidence = build_xai_evidence(results, framework)
    fallback = deterministic_xai_summary(evidence)
    if not use_llm:
        return fallback, {"source": "deterministic", "model": model}
    prompt = """You are AwareML's grounded Explainability Explainer. Use ONLY the JSON evidence below.
Write for a researcher who wants a clear, human-readable interpretation rather than a metric dump.
Use exactly four short Markdown sections: 'What the current explanation says', 'How stable the explanation looked', 'System-level context', and 'What this does not prove'.
Explain the selected method and up to three recorded top features in plain language. Explain deletion fidelity as a replay accuracy drop, and state that sensitivity is lower-is-better.
Do not call stability or consistency correctness. Do not claim causality, hidden reasoning, universal faithfulness, or that one framework is globally best because XAI methods may differ.
Never invent missing measurements and never convert missing values to zero. Keep the answer concise.

EVIDENCE:
""" + json.dumps(evidence, ensure_ascii=False, default=str)
    try:
        client = OllamaClient(model=model)
        answer, meta = client.generate_text(prompt)
        return answer, meta
    except Exception as exc:
        return fallback, {
            "source": "deterministic-fallback",
            "model": model,
            "warning": f"{type(exc).__name__}: {exc}",
        }
