from __future__ import annotations

import csv
import time
from pathlib import Path

from awareml.journal.objective_v31_diagnostic import diagnostic_root, write_json, write_jsonl
from awareml.llm.objective_selection_v31 import HybridEvidenceGroundedObjectiveSelectorV31, SELECTOR_ID

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "journal" / "objective_selection_benchmark_v1"


def main():
    path = BASE / "design" / "adversarial_set.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    selector = HybridEvidenceGroundedObjectiveSelectorV31(root=ROOT)
    outputs = []
    status_ok = 0
    over_selection = 0
    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        result = selector.select(str(row.get("scenario") or ""))
        ok = result.status == row.get("expected_status")
        status_ok += int(ok)
        over_selection += int(bool(result.selected_objectives))
        outputs.append(
            {
                "case_id": row["case_id"],
                "scenario": row.get("scenario"),
                "expected_status": row.get("expected_status"),
                "predicted_status": result.status,
                "selected_objectives": list(result.selected_objectives),
                "status_correct": ok,
                "latency_sec": time.perf_counter() - started,
                "evidence_audit": dict(selector.last_audit or {}),
            }
        )
        print("  [{:02d}/{:02d}] {} expected={} predicted={} selected={}".format(
            index, len(rows), row["case_id"], row.get("expected_status"), result.status, result.selected_objectives
        ))

    metrics = {
        "artifact": "objective_selection_v31_posthoc_development_diagnostic",
        "cases": len(rows),
        "status_accuracy": status_ok / float(len(rows)) if rows else 0.0,
        "over_selection_count": over_selection,
        "over_selection_rate": over_selection / float(len(rows)) if rows else 0.0,
        "selector_id": SELECTOR_ID,
        "evaluation_role": "posthoc_development_diagnostic_not_independent_test",
    }
    out_root = diagnostic_root(ROOT) / "results"
    write_jsonl(out_root / "adversarial_v31_outputs.jsonl", outputs)
    write_json(out_root / "adversarial_v31_metrics.json", metrics)

    print("=" * 76)
    print("AwareML Phase-12 V3.1 ADVERSARIAL diagnostic: COMPLETE")
    print("=" * 76)
    print("Cases:", metrics["cases"])
    print("Status accuracy: {:.4f}".format(metrics["status_accuracy"]))
    print("Over-selection count:", metrics["over_selection_count"])
    print("Over-selection rate: {:.4f}".format(metrics["over_selection_rate"]))
    print("Interpretation: POST-HOC DEVELOPMENT DIAGNOSTIC")
    print("=" * 76)


if __name__ == "__main__":
    main()
