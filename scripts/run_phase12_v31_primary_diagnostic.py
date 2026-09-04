from __future__ import annotations

import time
from pathlib import Path

from awareml.journal.objective_v31_diagnostic import (
    diagnostic_root,
    ground_truth_path,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from awareml.llm.objective_selection_v31 import (
    HybridEvidenceGroundedObjectiveSelectorV31,
    SELECTOR_ID,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    truth_path = ground_truth_path(ROOT)
    if not truth_path.exists():
        raise RuntimeError("Frozen Phase-12 ground truth copy is missing: {}".format(truth_path))

    rows = read_jsonl(truth_path)
    selector = HybridEvidenceGroundedObjectiveSelectorV31(root=ROOT)
    runtime = selector.client.verify_runtime()
    outputs = []

    print("Running V3.1 post-hoc primary development diagnostic on {} frozen Phase-12 cases...".format(len(rows)))
    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        result = selector.select(str(row.get("scenario") or ""))
        elapsed = time.perf_counter() - started
        outputs.append(
            {
                "scenario_id": row["scenario_id"],
                "scenario": row.get("scenario"),
                "benchmark_order": index,
                "status": result.status,
                "selected_objectives": list(result.selected_objectives),
                "uncertainties": list(result.uncertainties),
                "source": result.source,
                "model": result.model,
                "fallback_used": bool(result.fallback_used),
                "latency_sec": elapsed,
                "evidence_audit": dict(selector.last_audit or {}),
            }
        )
        print("  [{:02d}/{:02d}] {} -> {}".format(index, len(rows), row["scenario_id"], result.selected_objectives))

    out_root = diagnostic_root(ROOT) / "results"
    output_path = out_root / "primary_v31_outputs.jsonl"
    write_jsonl(output_path, outputs)
    write_json(
        out_root / "primary_v31_run_metadata.json",
        {
            "artifact": "objective_selection_v31_posthoc_development_diagnostic",
            "selector_id": SELECTOR_ID,
            "evaluation_role": "posthoc_development_diagnostic_not_independent_test",
            "cases": len(outputs),
            "ground_truth_sha256": sha256_file(truth_path),
            "model": runtime.get("model"),
            "model_digest": runtime.get("model_digest"),
            "ollama_version": runtime.get("ollama_version"),
            "phase12_frozen_artifact_modified": False,
            "warning": "V3.1 was developed after inspecting Phase-12 v1/V3 errors; these results are development-only.",
        },
    )

    print("=" * 76)
    print("AwareML Phase-12 V3.1 PRIMARY diagnostic: COMPLETE")
    print("=" * 76)
    print("Outputs:", len(outputs))
    print("Selector:", SELECTOR_ID)
    print("Model:", runtime.get("model"))
    print("Frozen Phase-12 v1 modified: False")
    print("Interpretation: POST-HOC DEVELOPMENT DIAGNOSTIC, not an independent test")
    print("=" * 76)


if __name__ == "__main__":
    main()
