from __future__ import annotations

import json
from pathlib import Path

from awareml.journal.objective_v31_diagnostic import (
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


def _v3_report():
    path = (
        ROOT
        / "data"
        / "journal"
        / "objective_selection_v3_diagnostic"
        / "results"
        / "v3_comparison_report.json"
    )
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    out_root = diagnostic_root(ROOT) / "results"
    primary_path = out_root / "primary_v31_outputs.jsonl"
    if not primary_path.exists():
        raise RuntimeError("Run scripts/run_phase12_v31_primary_diagnostic.py first.")

    truth = read_jsonl(ground_truth_path(ROOT))
    predictions = read_jsonl(primary_path)
    v31 = multilabel_metrics(truth, predictions)
    baseline = baseline_metrics(ROOT)
    paraphrase = _load_optional(out_root / "paraphrase_v31_metrics.json")
    adversarial = _load_optional(out_root / "adversarial_v31_metrics.json")
    v3_report = _v3_report()
    v3_metrics = dict((v3_report or {}).get("v3_posthoc") or {})

    comparison = {
        "evaluation_role": "posthoc_development_diagnostic_not_independent_test",
        "warning": (
            "V3.1 was designed after inspection of the frozen Phase-12 v1 and V3 errors. "
            "Therefore results on these same cases are development evidence only and cannot replace a fresh independent evaluation."
        ),
        "baseline_v1": baseline,
        "v3_posthoc": v3_metrics or None,
        "v31_posthoc": v31,
        "delta_v31_minus_v1": {
            key: v31[key] - float(baseline.get(key) or 0.0)
            for key in ["micro_precision", "micro_recall", "micro_f1", "macro_f1", "exact_match_rate"]
        },
        "delta_v31_minus_v3": {
            key: v31[key] - float(v3_metrics.get(key) or 0.0)
            for key in ["micro_precision", "micro_recall", "micro_f1", "macro_f1", "exact_match_rate"]
        } if v3_metrics else None,
        "paraphrase_v31": paraphrase,
        "adversarial_v31": adversarial,
        "fresh_independent_benchmark_required": True,
    }
    write_json(out_root / "v31_comparison_report.json", comparison)
    write_json(out_root / "primary_v31_metrics.json", v31)

    print("=" * 92)
    print("AwareML Phase-12 Objective Selection V3.1 post-hoc development diagnostic")
    print("=" * 92)
    if v3_metrics:
        print("                         V1 frozen      V3 post-hoc     V3.1 post-hoc")
    else:
        print("                         V1 frozen      V3.1 post-hoc")
    for key, label in [
        ("micro_precision", "Micro precision"),
        ("micro_recall", "Micro recall"),
        ("micro_f1", "Micro-F1"),
        ("macro_f1", "Macro-F1"),
        ("exact_match_rate", "Exact match"),
    ]:
        v1 = float(baseline.get(key) or 0.0)
        v31v = float(v31.get(key) or 0.0)
        if v3_metrics:
            v3v = float(v3_metrics.get(key) or 0.0)
            print("{:<22} {:>10.4f}      {:>10.4f}      {:>10.4f}".format(label, v1, v3v, v31v))
        else:
            print("{:<22} {:>10.4f}      {:>10.4f}".format(label, v1, v31v))

    if paraphrase:
        v1c = float(baseline.get("paraphrase_consistency_rate") or 0.0)
        v31c = float(paraphrase.get("paraphrase_prediction_consistency_rate") or 0.0)
        print("{:<22} {:>10.4f}      {:>10}      {:>10.4f}".format("Paraphrase consist.", v1c, "—", v31c))
        print("Paraphrase denominator: {} comparisons (60 inference calls = 10 BASE + 50 paraphrases)".format(
            int(paraphrase.get("paraphrase_evaluations") or 0)
        ))
    else:
        print("Paraphrase diagnostic: NOT RUN")

    if adversarial:
        baseline_over = int((baseline.get("adversarial_taxonomy_counts") or {}).get("over_selection", 0))
        print("Adversarial over-selection: V1={} / 14, V3.1={} / {}".format(
            baseline_over,
            int(adversarial.get("over_selection_count") or 0),
            int(adversarial.get("cases") or 0),
        ))
    else:
        print("Adversarial diagnostic: NOT RUN")

    print("-" * 92)
    print("IMPORTANT: V3.1 is POST-HOC DEVELOPMENT evidence on already-inspected Phase-12 material.")
    print("Do not replace the frozen Phase-12 v1 result with these values.")
    print("Freeze the V3.1 method first, then evaluate it once on a fresh independently annotated benchmark.")
    print("=" * 92)


if __name__ == "__main__":
    main()
