from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from awareml.journal.objective_v3_diagnostic import diagnostic_root, write_json, write_jsonl
from awareml.llm.objective_selection_v3 import EvidenceGroundedObjectiveSelectorV3, SELECTOR_ID


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "journal" / "objective_selection_benchmark_v1"


def _truth_by_family():
    path = BASE / "human" / "paraphrase_family_review.csv"
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            labels = []
            for col, label in [
                ("accuracy", "Accuracy"),
                ("runtime", "Runtime"),
                ("energy", "Energy"),
                ("co2", "CO2"),
            ]:
                if str(row.get(col, "")).strip().lower() in {"yes", "1", "true"}:
                    labels.append(label)
            out[row["family_id"]] = labels
    return out


def main():
    family_truth = _truth_by_family()
    path = BASE / "design" / "paraphrase_families.generated.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    selector = EvidenceGroundedObjectiveSelectorV3(root=ROOT)
    outputs = []
    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        result = selector.select(str(row.get("scenario") or ""))
        outputs.append(
            {
                "family_id": row["family_id"],
                "variant_id": row["variant_id"],
                "variant_type": row.get("variant_type"),
                "scenario": row.get("scenario"),
                "selected_objectives": list(result.selected_objectives),
                "status": result.status,
                "latency_sec": time.perf_counter() - started,
                "ground_truth_objectives": family_truth.get(row["family_id"], []),
                "evidence_audit": dict(selector.last_audit or {}),
            }
        )
        print("  [{:02d}/{:02d}] {} {} -> {}".format(index, len(rows), row["family_id"], row["variant_id"], result.selected_objectives))

    by_family = {}
    for row in outputs:
        by_family.setdefault(row["family_id"], []).append(row)
    comparisons = []
    para_total = para_exact = consistent = changed = 0
    base_exact = 0
    for family_id, family_rows in sorted(by_family.items()):
        base = next((r for r in family_rows if str(r["variant_id"]).upper() == "BASE"), None)
        if base is None:
            raise RuntimeError("Missing BASE row for {}".format(family_id))
        gt = set(base.get("ground_truth_objectives") or [])
        base_set = set(base.get("selected_objectives") or [])
        base_exact += int(base_set == gt)
        for row in family_rows:
            if str(row["variant_id"]).upper() == "BASE":
                continue
            para_total += 1
            pred = set(row.get("selected_objectives") or [])
            is_exact = pred == gt
            is_consistent = pred == base_set
            para_exact += int(is_exact)
            consistent += int(is_consistent)
            changed += int(not is_consistent)
            comparisons.append(
                {
                    "family_id": family_id,
                    "variant_id": row["variant_id"],
                    "base_prediction": sorted(base_set),
                    "paraphrase_prediction": sorted(pred),
                    "ground_truth": sorted(gt),
                    "exact_match": is_exact,
                    "consistent_with_base": is_consistent,
                }
            )

    out_root = diagnostic_root(ROOT) / "results"
    write_jsonl(out_root / "paraphrase_v3_outputs.jsonl", outputs)
    write_jsonl(out_root / "paraphrase_v3_comparisons.jsonl", comparisons)
    metrics = {
        "artifact": "objective_selection_v3_posthoc_diagnostic",
        "families": len(by_family),
        "paraphrase_evaluations": para_total,
        "base_exact_match_rate": base_exact / float(len(by_family)) if by_family else 0.0,
        "paraphrase_exact_match_rate": para_exact / float(para_total) if para_total else 0.0,
        "paraphrase_prediction_consistency_rate": consistent / float(para_total) if para_total else 0.0,
        "set_change_rate": changed / float(para_total) if para_total else 0.0,
        "evaluation_role": "posthoc_development_diagnostic_not_independent_test",
        "selector_id": SELECTOR_ID,
    }
    write_json(out_root / "paraphrase_v3_metrics.json", metrics)

    print("=" * 72)
    print("AwareML Phase-12 V3 PARAPHRASE diagnostic: COMPLETE")
    print("=" * 72)
    print("Families:", metrics["families"])
    print("Paraphrase evaluations:", metrics["paraphrase_evaluations"])
    print("Base exact match: {:.4f}".format(metrics["base_exact_match_rate"]))
    print("Paraphrase exact match: {:.4f}".format(metrics["paraphrase_exact_match_rate"]))
    print("Prediction consistency: {:.4f}".format(metrics["paraphrase_prediction_consistency_rate"]))
    print("Set-change rate: {:.4f}".format(metrics["set_change_rate"]))
    print("Interpretation: POST-HOC DEVELOPMENT DIAGNOSTIC")
    print("=" * 72)


if __name__ == "__main__":
    main()
