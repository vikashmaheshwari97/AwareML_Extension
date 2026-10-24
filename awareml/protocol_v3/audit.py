from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .common import read_json, write_json


def audit_existing(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("Audit output exists; choose a new path.")
    paths = []
    snapshot = root / "data/meta/snapshots/meta_logs_v2.json"
    if snapshot.exists():
        paths.append(snapshot)
    artifacts = root / "artifacts"
    if artifacts.exists():
        paths.extend(sorted(artifacts.rglob("result.json")))
    records, skipped = [], []
    for path in paths:
        try:
            payload = read_json(path)
            if isinstance(payload, dict) and isinstance(payload.get("records"), list):
                raw = payload["records"]
            elif isinstance(payload, dict) and isinstance(payload.get("result"), dict):
                item = {**payload["result"], **payload.get("task", {})}
                dataset = payload.get("dataset") or {}
                item["available_rows"] = dataset.get("rows_in_evaluation_stream")
                item.setdefault("max_samples_requested", 30000)
                raw = [item]
            else:
                skipped.append({"path": str(path), "reason": "not_a_legacy_result_schema"})
                continue
            for item in raw:
                available = item.get("available_rows") or (item.get("dataset_provenance") or {}).get("rows")
                cap = item.get("max_samples_requested", 30000)
                expected = min(int(available), int(cap)) if available is not None else int(cap)
                samples = int(item.get("samples_processed", item.get("samples", 0)))
                records.append({"source": str(path), "dataset_id": item.get("dataset_id", "unknown"),
                                "framework": item["framework"], "seed": item.get("seed"),
                                "expected_samples": expected, "samples_processed": samples,
                                "runtime_sec": item["runtime_sec"], "completed_30k": expected == 30000 and samples == expected,
                                "incomplete": samples < expected,
                                "inferred_time_limit": samples < expected and float(item["runtime_sec"]) >= float(item.get("time_budget_sec", 60))})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
    output.mkdir(parents=True)
    if not records:
        result = {"status": "no_matching_legacy_results", "records": 0,
                  "searched": [str(p) for p in paths], "skipped": skipped,
                  "next_step": "Check the artifacts location; no runtime table or averages can be calculated from zero records."}
        write_json(output / "audit.json", result)
        return result
    frame = pd.DataFrame(records)
    frame.to_csv(output / "run_audit.csv", index=False)
    # A completed-30k runtime is never imputed from a partial run.
    frame["completed_30k_runtime"] = np.where(frame.completed_30k, frame.runtime_sec, np.nan)
    summary = frame.groupby("framework", as_index=False).agg(
        runs=("framework", "size"), completed_30k=("completed_30k", "sum"), incomplete_runs=("incomplete", "sum"),
        mean_completed_30k_runtime_sec=("completed_30k_runtime", "mean"),
        sd_completed_30k_runtime_sec=("completed_30k_runtime", "std"),
        min_processed=("samples_processed", "min"), mean_processed=("samples_processed", "mean"), max_processed=("samples_processed", "max"))
    summary.to_csv(output / "framework_runtime_summary.csv", index=False)
    result = {"status": "audited", "records": len(records), "skipped": skipped,
              "note": "Rows identify source files. Different campaign copies can repeat keys; do not add them together as independent executions."}
    write_json(output / "audit.json", result)
    return result
