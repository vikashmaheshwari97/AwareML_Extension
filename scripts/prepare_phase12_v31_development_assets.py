from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "data" / "journal" / "objective_selection_benchmark_v1"
OUT = ROOT / "data" / "journal" / "objective_selection_v31_development"


def _clean_human_written():
    src = LEGACY / "human" / "human_written_scenarios.csv"
    dst = OUT / "human" / "human_written_scenarios.clean.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys()) if rows else []
    kept = [row for row in rows if str(row.get("scenario") or "").strip()]
    with dst.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    return src, dst, len(rows), len(kept)


def _legacy_blind_paraphrase_templates():
    src = LEGACY / "design" / "paraphrase_families.generated.csv"
    with src.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    # Legacy re-review aid only. Family IDs and variant IDs are omitted and rows
    # are randomized so future reviewers label each sentence independently.
    blind = [
        {
            "blind_id": "LEGACY-PARA-{:03d}".format(i + 1),
            "scenario": row.get("scenario"),
            "accuracy": "",
            "runtime": "",
            "energy": "",
            "co2": "",
            "ambiguous": "",
            "notes": "",
            "reviewer_id": "",
        }
        for i, row in enumerate(rows)
    ]
    rng = random.Random(31031)
    rng.shuffle(blind)
    out_dir = OUT / "paraphrase_review_templates"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(blind[0].keys()) if blind else []
    for name in ["legacy_paraphrase_blind_review_A.csv", "legacy_paraphrase_blind_review_B.csv"]:
        path = out_dir / name
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(blind)


def _fresh_templates():
    out_dir = OUT / "fresh_benchmark_scaffold"
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_fields = [
        "scenario_id",
        "scenario",
        "author_type",
        "domain",
        "k_prime_design_bucket",
        "collection_status",
    ]
    scenario_path = out_dir / "fresh_scenarios.template.csv"
    with scenario_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=scenario_fields)
        writer.writeheader()

    annotation_fields = [
        "scenario_id",
        "scenario",
        "accuracy",
        "runtime",
        "energy",
        "co2",
        "ambiguous",
        "notes",
        "annotator_id",
    ]
    for suffix in ["A", "B", "C"]:
        path = out_dir / "fresh_annotations_{}.template.csv".format(suffix)
        with path.open("w", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=annotation_fields).writeheader()

    adversarial_fields = [
        "case_id",
        "scenario",
        "expected_status",
        "expected_behavior",
        "notes",
    ]
    with (out_dir / "fresh_adversarial.template.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=adversarial_fields).writeheader()

    (out_dir / "README.md").write_text(
        "# Fresh V3.1 benchmark scaffold\n\n"
        "These files are intentionally empty. Populate them only after the V3.1 method is frozen. "
        "Do not copy the legacy Phase-12 scenarios or private generator-intent labels into the final test set.\n",
        encoding="utf-8",
    )


def main():
    src, dst, original_rows, kept_rows = _clean_human_written()
    _legacy_blind_paraphrase_templates()
    _fresh_templates()
    print("=" * 76)
    print("AwareML V3.1 development assets: PREPARED")
    print("=" * 76)
    print("Legacy human-written source rows:", original_rows)
    print("Non-empty human-written rows copied:", kept_rows)
    print("Clean derivative:", dst)
    print("Legacy frozen source modified: False")
    print("Fresh benchmark scaffold contains labels/cases: False")
    print("=" * 76)


if __name__ == "__main__":
    main()
