from __future__ import annotations

import json
from pathlib import Path

from awareml.journal.objective_v3_diagnostic import (
    baseline_metrics,
    diagnostic_root,
    ground_truth_path,
    multilabel_metrics,
    read_jsonl,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_optional(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    out_root = diagnostic_root(ROOT) / "results"
    primary_path = out_root / "primary_v3_outputs.jsonl"
    if not primary_path.exists():
        raise RuntimeError(
            "Run scripts/run_phase12_v3_primary_diagnostic.py first."
        )

    truth = read_jsonl(ground_truth_path(ROOT))
    predictions = read_jsonl(primary_path)
    v3 = multilabel_metrics(truth, predictions)
    baseline = baseline_metrics(ROOT)
    paraphrase = _load_optional(out_root / "paraphrase_v3_metrics.json")
    adversarial = _load_optional(out_root / "adversarial_v3_metrics.json")

    comparison = {
        "evaluation_role": "posthoc_development_diagnostic_not_independent_test",
        "warning": (
            "V3 was motivated by inspection of the frozen Phase-12 v1 failure mode. "
            "Therefore improvements on these same 55 cases are development evidence, "
            "not an independent replacement for the frozen v1 journal result."
        ),
        "baseline_v1": baseline,
        "v3_posthoc": v3,
        "delta_v3_minus_v1": {
            "micro_precision": v3["micro_precision"] - float(baseline.get("micro_precision") or 0.0),
            "micro_recall": v3["micro_recall"] - float(baseline.get("micro_recall") or 0.0),
            "micro_f1": v3["micro_f1"] - float(baseline.get("micro_f1") or 0.0),
            "macro_f1": v3["macro_f1"] - float(baseline.get("macro_f1") or 0.0),
            "exact_match_rate": v3["exact_match_rate"] - float(baseline.get("exact_match_rate") or 0.0),
        },
        "paraphrase_v3": paraphrase,
        "adversarial_v3": adversarial,
        "engineering_targets": {
            "micro_f1": 0.75,
            "exact_match_rate": 0.30,
            "paraphrase_consistency_rate": 0.85,
            "note": "Engineering targets only; not universal publication thresholds.",
        },
    }
    write_json(out_root / "v3_comparison_report.json", comparison)
    write_json(out_root / "primary_v3_metrics.json", v3)

    print("=" * 78)
    print("AwareML Phase-12 Objective Selection V3 post-hoc diagnostic")
    print("=" * 78)
    print("                         V1 frozen      V3 post-hoc      delta")
    for key, label in [
        ("micro_precision", "Micro precision"),
        ("micro_recall", "Micro recall"),
        ("micro_f1", "Micro-F1"),
        ("macro_f1", "Macro-F1"),
        ("exact_match_rate", "Exact match"),
    ]:
        left = float(baseline.get(key) or 0.0)
        right = float(v3.get(key) or 0.0)
        print("{:<22} {:>10.4f}      {:>10.4f}      {:+.4f}".format(
            label, left, right, right - left
        ))

    if paraphrase:
        base_consistency = float(baseline.get("paraphrase_consistency_rate") or 0.0)
        v3_consistency = float(paraphrase.get("paraphrase_prediction_consistency_rate") or 0.0)
        print("{:<22} {:>10.4f}      {:>10.4f}      {:+.4f}".format(
            "Paraphrase consist.", base_consistency, v3_consistency, v3_consistency - base_consistency
        ))
    else:
        print("Paraphrase diagnostic: NOT RUN")

    if adversarial:
        baseline_over = int((baseline.get("adversarial_taxonomy_counts") or {}).get("over_selection", 0))
        v3_over = int(adversarial.get("over_selection_count") or 0)
        print("Adversarial over-selection: V1={} / 14, V3={} / {}".format(
            baseline_over, v3_over, int(adversarial.get("cases") or 0)
        ))
    else:
        print("Adversarial diagnostic: NOT RUN")

    print("-" * 78)
    print("IMPORTANT: these V3 numbers are post-hoc development diagnostics.")
    print("Do not replace the frozen Phase-12 v1 journal result with them.")
    print("A fresh independently annotated benchmark is required for a new final claim.")
    print("=" * 78)


if __name__ == "__main__":
    main()
