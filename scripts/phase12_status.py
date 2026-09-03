from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "journal" / "objective_selection_benchmark_v1"

checks = [
    ("Design assets", BASE / "design" / "candidate_pool.generated.csv"),
    ("Design frozen", BASE / "frozen_design" / "design_manifest.json"),
    ("Realism filter", BASE / "human" / "realism_filter.csv"),
    ("Annotation pool", BASE / "human" / "final_annotation_pool.csv"),
    ("Annotation A", BASE / "human" / "annotations_A.csv"),
    ("Annotation B", BASE / "human" / "annotations_B.csv"),
    ("Annotation C", BASE / "human" / "annotations_C.csv"),
    ("Ground truth", BASE / "results" / "ground_truth.jsonl"),
    ("Agreement", BASE / "results" / "annotation_agreement.json"),
    ("Ground truth frozen", BASE / "frozen_ground_truth" / "ground_truth_manifest.json"),
    ("Main LLaMA outputs", BASE / "results" / "llm_outputs.jsonl"),
    ("Main metrics", BASE / "results" / "metrics.json"),
    ("Paraphrase results", BASE / "results" / "paraphrase_metrics.json"),
    ("Adversarial taxonomy", BASE / "results" / "adversarial_failure_taxonomy.json"),
    ("Final frozen manifest", BASE / "frozen" / "manifest.json"),
]

print("=" * 72)
print("AwareML Phase 12 status")
print("=" * 72)
for label, path in checks:
    print("{:<28} {}".format(label, "READY" if path.exists() else "PENDING"))
print("=" * 72)
