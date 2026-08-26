from __future__ import annotations

import hashlib
import html
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .state import ROOT, result_dicts


def _json_safe(value):
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def research_payload(state, faithfulness_report=None):
    ranking = state.get("v2_candidates")
    ranking_rows = ranking.to_dict(orient="records") if isinstance(ranking, pd.DataFrame) else []
    payload = {
        "schema_version": "1.0",
        "exported_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "ui": {"version": state.get("ui_v2_version")},
        "dataset": {
            "name": state.get("dataset_name"),
            "target": state.get("target"),
            "sensitive_attribute": state.get("sensitive"),
            "rows": len(state["dataset"]) if state.get("dataset") is not None else None,
        },
        "preferences": state.get("preference_weights"),
        "ranking_mode": state.get("ranking_mode"),
        "pre_run_profile": state.get("v2_profile"),
        "pre_run_ranking": ranking_rows,
        "pre_run_ranking_meta": state.get("v2_ranking_meta"),
        "copilot": {
            "goal": state.get("copilot_goal"),
            "proposal": state.get("copilot_proposal"),
            "review": state.get("copilot_review"),
        },
        "run_results": result_dicts(state),
        "faithfulness_development_report": faithfulness_report,
        "integrity": {
            "raw_dataset_rows_exported": False,
            "held_out_23_dataset_split_used_by_ui": False,
        },
    }
    return _json_safe(payload)


def results_csv_bytes(state):
    rows = result_dicts(state)
    if not rows:
        return pd.DataFrame().to_csv(index=False).encode("utf-8")
    flat = []
    for r in rows:
        flat.append({
            "framework": r.get("framework"),
            "backend": r.get("backend"),
            "accuracy": r.get("accuracy"),
            "f1_macro": r.get("f1_macro"),
            "runtime_sec": r.get("runtime_sec"),
            "energy_kwh": r.get("energy_kwh"),
            "co2_kg": r.get("co2_kg"),
            "throughput_samples_sec": r.get("throughput_samples_sec"),
            "mean_prediction_latency_ms": r.get("mean_prediction_latency_ms"),
            "p95_prediction_latency_ms": r.get("p95_prediction_latency_ms"),
            "drift_events": len(r.get("drift_events") or []),
        })
    return pd.DataFrame(flat).to_csv(index=False).encode("utf-8")


def html_report_bytes(payload):
    dataset = payload.get("dataset") or {}
    ranking = payload.get("pre_run_ranking") or []
    results = payload.get("run_results") or []
    faith = payload.get("faithfulness_development_report") or {}

    ranking_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{:.4f}</td><td>{}</td></tr>".format(
            html.escape(str(row.get("framework"))),
            html.escape(str(row.get("rank"))),
            float(row.get("utility", 0.0) or 0.0),
            html.escape(str(row.get("pareto_efficient"))),
        )
        for row in ranking
    )
    result_rows = "".join(
        "<tr><td>{}</td><td>{:.4f}</td><td>{:.4f}</td><td>{:.4g}</td><td>{:.4g}</td><td>{:.4g}</td></tr>".format(
            html.escape(str(row.get("framework"))),
            float(row.get("accuracy", 0.0) or 0.0),
            float(row.get("f1_macro", 0.0) or 0.0),
            float(row.get("runtime_sec", 0.0) or 0.0),
            float(row.get("energy_kwh", 0.0) or 0.0),
            float(row.get("co2_kg", 0.0) or 0.0),
        )
        for row in results
    )
    aef = ((faith.get("deterministic") or {}).get("mean_evidence_fidelity_score"))

    doc = """<!doctype html><html><head><meta charset="utf-8">
<title>AwareML Research Evidence Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:42px;color:#18202a}}
h1{{font-size:32px;margin-bottom:4px}}h2{{margin-top:34px;font-size:20px}}
.muted{{color:#65717d}}table{{border-collapse:collapse;width:100%;margin-top:12px}}
th,td{{border-bottom:1px solid #dfe5ea;text-align:left;padding:9px 8px;font-size:13px}}
.callout{{background:#f4f7fa;border-left:3px solid #4b89c8;padding:12px 14px}}
</style></head><body>
<h1>AwareML Research Evidence Report</h1>
<div class="muted">Generated {generated}</div>
<h2>Experiment context</h2>
<table><tr><th>Dataset</th><td>{dataset}</td></tr><tr><th>Rows</th><td>{rows}</td></tr>
<tr><th>Target</th><td>{target}</td></tr><tr><th>Sensitive attribute</th><td>{sensitive}</td></tr></table>
<h2>Pre-run ML recommender</h2>
<table><thead><tr><th>Framework</th><th>Rank</th><th>Utility</th><th>Pareto efficient</th></tr></thead><tbody>{ranking_rows}</tbody></table>
<h2>Observed run results</h2>
<table><thead><tr><th>Framework</th><th>Accuracy</th><th>Macro-F1</th><th>Runtime (s)</th><th>Energy (kWh)</th><th>CO₂ (kg)</th></tr></thead><tbody>{result_rows}</tbody></table>
<h2>Faithfulness development evidence</h2>
<div class="callout">Mean deterministic AEF: <b>{aef}</b>. This is development/meta evidence, not the frozen 23-dataset held-out evaluation.</div>
<h2>Integrity statement</h2>
<p>Raw dataset rows are not included. The Research UI does not use the frozen 23-dataset held-out split.</p>
</body></html>""".format(
        generated=html.escape(str(payload.get("exported_at_utc"))),
        dataset=html.escape(str(dataset.get("name"))),
        rows=html.escape(str(dataset.get("rows"))),
        target=html.escape(str(dataset.get("target"))),
        sensitive=html.escape(str(dataset.get("sensitive_attribute"))),
        ranking_rows=ranking_rows,
        result_rows=result_rows,
        aef="{:.4f}".format(float(aef)) if aef is not None else "N/A",
    )
    return doc.encode("utf-8")


def build_research_zip(state, faithfulness_report=None):
    payload = research_payload(state, faithfulness_report)
    json_bytes = json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    csv_bytes = results_csv_bytes(state)
    report_bytes = html_report_bytes(payload)

    manifest_paths = [
        ROOT / "data" / "meta" / "models" / "recommender_v2" / "release_manifest.json",
        ROOT / "data" / "llm" / "copilot_v1" / "manifest.json",
        ROOT / "data" / "llm" / "faithfulness_v1" / "manifest.json",
    ]

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("awareml_evidence.json", json_bytes)
        archive.writestr("run_metrics.csv", csv_bytes)
        archive.writestr("research_report.html", report_bytes)

        checksums = {
            "awareml_evidence.json": hashlib.sha256(json_bytes).hexdigest(),
            "run_metrics.csv": hashlib.sha256(csv_bytes).hexdigest(),
            "research_report.html": hashlib.sha256(report_bytes).hexdigest(),
        }
        for path in manifest_paths:
            if path.exists():
                data = path.read_bytes()
                name = "provenance/{}_{}".format(path.parent.name, path.name)
                archive.writestr(name, data)
                checksums[name] = hashlib.sha256(data).hexdigest()

        archive.writestr("checksums.json", json.dumps(checksums, indent=2).encode("utf-8"))

    return buffer.getvalue()
