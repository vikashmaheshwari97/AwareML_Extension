from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "journal" / "objective_selection_benchmark_v1"
OUT = ROOT / "data" / "journal" / "objective_selection_v31_development" / "audit"
OBJECTIVES = ("accuracy", "runtime", "energy", "co2")


def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def yes(value):
    return str(value or "").strip().lower() in {"yes", "true", "1"}


def objective_set(row):
    return tuple(o for o in OBJECTIVES if yes(row.get(o)))


def main():
    candidate = rows(BASE / "design" / "candidate_pool.generated.csv")
    intent = rows(BASE / "design" / "candidate_generation_intent.PRIVATE.csv")
    realism = rows(BASE / "human" / "realism_filter.csv")
    final_pool = rows(BASE / "human" / "final_annotation_pool.csv")
    human_written = rows(BASE / "human" / "human_written_scenarios.csv")
    ann = [rows(BASE / "human" / name) for name in ["annotations_A.csv", "annotations_B.csv", "annotations_C.csv"]]
    para_review = rows(BASE / "human" / "paraphrase_family_review.csv")
    para_families = rows(BASE / "design" / "paraphrase_families.generated.csv")
    adversarial = rows(BASE / "design" / "adversarial_set.csv")

    intent_by_id = {r["scenario_id"]: r for r in intent}
    realism_by_id = {r["scenario_id"]: r for r in realism}
    ann_by_id = [{r["scenario_id"]: r for r in part} for part in ann]

    intended_counts = Counter()
    retained_by_intended_k = Counter()
    for row in intent:
        k = int(row.get("intended_k_prime") or 0)
        intended_counts[k] += 1
        rr = realism_by_id.get(row["scenario_id"])
        if rr and yes(rr.get("keep")):
            retained_by_intended_k[k] += 1

    final_generated = [r for r in final_pool if r.get("source") == "generated_external_llm"]
    final_human = [r for r in final_pool if r.get("source") != "generated_external_llm"]

    majority_sets = {}
    disagreement_counts = Counter()
    unanimous_full = 0
    for sid in [r["scenario_id"] for r in final_pool]:
        votes = {}
        for obj in OBJECTIVES:
            vals = [yes(part[sid].get(obj)) for part in ann_by_id]
            votes[obj] = sum(vals)
            if len(set(vals)) > 1:
                disagreement_counts[obj] += 1
        maj = tuple(obj for obj in OBJECTIVES if votes[obj] >= 2)
        majority_sets[sid] = maj
        sets = [objective_set(part[sid]) for part in ann_by_id]
        unanimous_full += int(sets[0] == sets[1] == sets[2])

    generated_intent_matches = 0
    for row in final_generated:
        iid = row["scenario_id"]
        intended = tuple(
            obj.lower() for obj in str(intent_by_id[iid].get("generation_intent") or "").split("|") if obj
        )
        if tuple(intended) == tuple(majority_sets[iid]):
            generated_intent_matches += 1

    final_k = Counter(len(v) for v in majority_sets.values())
    retained_display = {k: retained_by_intended_k[k] for k in sorted(intended_counts)}
    final_k_display = {k: final_k[k] for k in [1, 2, 3, 4]}
    blank_human_rows = sum(not str(r.get("scenario") or "").strip() for r in human_written)
    para_reviewers = sorted({str(r.get("reviewer_id") or "").strip() for r in para_review if str(r.get("reviewer_id") or "").strip()})
    variants_per_family = Counter(r["family_id"] for r in para_families)

    report = {
        "artifact": "phase12_csv_input_audit_for_v31_development",
        "candidate_rows": len(candidate),
        "candidate_unique_ids": len({r["scenario_id"] for r in candidate}),
        "candidate_duplicate_scenario_texts": len(candidate) - len({r["scenario"] for r in candidate}),
        "intended_k_prime_counts": {str(k): intended_counts[k] for k in sorted(intended_counts)},
        "retained_by_intended_k_prime": {str(k): retained_by_intended_k[k] for k in sorted(intended_counts)},
        "final_primary_rows": len(final_pool),
        "final_generated_rows": len(final_generated),
        "final_human_written_rows": len(final_human),
        "final_majority_k_prime_counts": {str(k): final_k[k] for k in [1, 2, 3, 4]},
        "full_set_unanimous_rows": unanimous_full,
        "full_set_unanimous_rate": unanimous_full / float(len(final_pool)) if final_pool else 0.0,
        "objective_disagreement_rows": dict(disagreement_counts),
        "generated_final_majority_matches_private_generation_intent": generated_intent_matches,
        "generated_final_rows_checked_against_private_intent": len(final_generated),
        "human_written_source_rows": len(human_written),
        "human_written_blank_placeholders": blank_human_rows,
        "paraphrase_families": len(variants_per_family),
        "paraphrase_rows": len(para_families),
        "paraphrase_variants_per_family": dict(variants_per_family),
        "paraphrase_reviewers": para_reviewers,
        "adversarial_cases": len(adversarial),
        "important_interpretation": [
            "The legacy frozen Phase-12 v1 artifact must not be rewritten.",
            "All k_prime=4 generated candidates were removed by realism filtering; restore k_prime=4 only in a fresh natural benchmark, not by resurrecting rejected checklist-like cases.",
            "The legacy human-written source contains blank reserved rows; use the clean derivative for V3.1 development assets.",
            "Energy and CO2 show more human disagreement than Accuracy/Runtime and should remain distinct objectives with explicit semantic evidence.",
            "Private generation intent is provenance/coverage metadata only; it is forbidden as a V3.1 selector dependency.",
            "A fresh final V3.1 benchmark is required because the legacy labels/errors have been inspected during method development."
        ],
        "frozen_legacy_files_modified": False,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase12_csv_input_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "PHASE12_CSV_INPUT_AUDIT.md").write_text(
        "# Phase-12 CSV input audit for V3.1 development\n\n"
        f"- Candidate pool: {len(candidate)} rows, {len({r['scenario_id'] for r in candidate})} unique IDs.\n"
        f"- Final primary benchmark: {len(final_pool)} rows ({len(final_generated)} generated + {len(final_human)} human-written).\n"
        f"- Final majority k′ distribution: {final_k_display}.\n"
        f"- Retained generated cases by intended k′: {retained_display}.\n"
        f"- Full-set unanimous human annotation: {unanimous_full}/{len(final_pool)}.\n"
        f"- Objective disagreement rows: {dict(disagreement_counts)}.\n"
        f"- Blank rows in legacy human-written source: {blank_human_rows}.\n"
        f"- Paraphrase semantic reviewers in legacy artifact: {', '.join(para_reviewers) or 'none'}.\n\n"
        "The audit is descriptive. It does not alter the frozen Phase-12 v1 evidence.\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("Phase-12 CSV input audit: COMPLETE")
    print("=" * 76)
    print("Candidate rows:", len(candidate))
    print("Final rows:", len(final_pool), "generated:", len(final_generated), "human-written:", len(final_human))
    print("Retained by intended k':", retained_display)
    print("Final majority k':", final_k_display)
    print("Full-set unanimous:", "{}/{}".format(unanimous_full, len(final_pool)))
    print("Objective disagreement rows:", dict(disagreement_counts))
    print("Legacy human-written blank placeholders:", blank_human_rows)
    print("Paraphrase reviewers:", para_reviewers)
    print("Frozen legacy files modified: False")
    print("=" * 76)


if __name__ == "__main__":
    main()
