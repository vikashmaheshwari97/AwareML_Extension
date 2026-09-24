#!/usr/bin/env python3
"""
Phase 18 journal-evidence freeze gate.

Run this only AFTER phase18_statistical_analysis.py has passed and the
statistical outputs have been reviewed.

The script deliberately refuses to freeze if:
  * expected statistical outputs are missing;
  * synthetic-demo files are found inside the Phase-18 evidence tree;
  * the 31x5 ground-truth structure is not intact;
  * the known Health Insurance positive-label metadata inconsistency is still
    detectable in the frozen dataset manifest.

It never edits empirical result files. The only output is a freeze manifest.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROTOCOL_ID = "phase18_final_heldout_31_v1"
CONFIRM_TOKEN = "FREEZE_PHASE18_JOURNAL_EVIDENCE"

REQUIRED_STATS = [
    "phase18_framework_descriptive_ci.csv",
    "phase18_omnibus_friedman.csv",
    "phase18_pairwise_framework_comparisons.csv",
    "phase18_effect_sizes.csv",
    "phase18_recommender_confidence_intervals.csv",
    "phase18_objective_prediction_confidence_intervals.csv",
    "phase18_confidence_intervals.csv",
    "phase18_statistical_analysis_manifest.json",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(root):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(root), text=True
    ).strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--confirm", required=True)
    return p.parse_args()


def audit_health_insurance_manifest(repo_root):
    path = (
        repo_root
        / "data"
        / "journal"
        / PROTOCOL_ID
        / "frozen"
        / "dataset_manifest_frozen.tsv"
    )
    if not path.exists():
        return {
            "status": "NOT_CHECKED",
            "reason": "dataset_manifest_frozen.tsv not found",
        }

    df = pd.read_csv(path, sep="\t")
    if "dataset_id" not in df.columns:
        return {
            "status": "NOT_CHECKED",
            "reason": "dataset_id column not found",
        }

    hit = df[df["dataset_id"].astype(str) == "health_insurance_Real"]
    if len(hit) != 1:
        return {
            "status": "NOT_CHECKED",
            "reason": "health_insurance_Real row count=%d" % len(hit),
        }

    row = hit.iloc[0]
    positive_candidates = [
        "positive_label_json",
        "positive_label",
        "resolved_positive_label",
    ]
    class_candidates = [
        "evaluation_target_unique",
        "effective_classes",
        "effective_num_classes",
        "target_unique",
    ]

    positive = None
    positive_col = None
    for c in positive_candidates:
        if c in df.columns and pd.notna(row[c]):
            positive = str(row[c]).strip().strip('"')
            positive_col = c
            break

    class_count = None
    class_col = None
    for c in class_candidates:
        if c in df.columns and pd.notna(row[c]):
            try:
                class_count = int(float(row[c]))
                class_col = c
                break
            except Exception:
                pass

    if class_count == 4 and positive in ("4", "4.0"):
        return {
            "status": "FAIL",
            "reason": (
                "Frozen Health Insurance metadata declares positive label 4 "
                "while the derived evaluation target has four classes (0-3)."
            ),
            "positive_label_column": positive_col,
            "class_count_column": class_col,
        }

    return {
        "status": "PASS",
        "positive_label": positive,
        "positive_label_column": positive_col,
        "class_count": class_count,
        "class_count_column": class_col,
    }


def main():
    args = parse_args()
    if args.confirm != CONFIRM_TOKEN:
        raise SystemExit(
            "Refusing to freeze. Re-run with --confirm %s" % CONFIRM_TOKEN
        )

    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = repo_root / "artifacts" / PROTOCOL_ID
    stats_dir = artifact_root / "evaluation" / "statistical_analysis"

    missing = [name for name in REQUIRED_STATS if not (stats_dir / name).exists()]
    if missing:
        raise SystemExit("Missing required statistical outputs: %s" % missing)

    synthetic = [
        p for p in artifact_root.rglob("*")
        if p.is_file() and "SYNTHETIC_DEMO" in p.name.upper()
    ]
    if synthetic:
        raise SystemExit(
            "Synthetic demo files found inside journal evidence tree; "
            "move them outside before freezing:\n%s"
            % "\n".join(str(p) for p in synthetic)
        )

    gt = artifact_root / "reduced" / "phase18_framework_ground_truth_155.parquet"
    if not gt.exists():
        csv_gt = gt.with_suffix(".csv")
        gt = csv_gt if csv_gt.exists() else gt
    if not gt.exists():
        raise SystemExit("Ground truth 155 file not found.")

    frame = pd.read_parquet(gt) if gt.suffix == ".parquet" else pd.read_csv(gt)
    if len(frame) != 155:
        raise SystemExit("Ground truth row count is not 155.")
    if frame["dataset_id"].nunique() != 31:
        raise SystemExit("Ground truth dataset count is not 31.")
    if frame["framework"].nunique() != 5:
        raise SystemExit("Ground truth framework count is not 5.")

    metadata_audit = audit_health_insurance_manifest(repo_root)
    if metadata_audit["status"] != "PASS":
        raise SystemExit(
            "JOURNAL FREEZE BLOCKED by frozen metadata audit:\n%s"
            % json.dumps(metadata_audit, indent=2)
        )

    evidence_files = [
        gt,
        artifact_root / "evaluation" / "phase18_frozen_v2_predictions_155.parquet",
        artifact_root / "evaluation" / "phase18_preference_eval_primary_3100.csv",
        artifact_root / "evaluation" / "phase18_objective_metrics.csv",
        artifact_root / "evaluation" / "phase18_preference_summary.csv",
    ]
    evidence_files.extend(stats_dir / name for name in REQUIRED_STATS)

    hashes = {}
    for p in evidence_files:
        if p.exists():
            hashes[str(p.relative_to(repo_root))] = sha256_file(p)

    freeze = {
        "schema_version": "1.0",
        "phase": 18,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_JOURNAL_EVIDENCE",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(repo_root),
        "counts": {
            "held_out_datasets": 31,
            "frameworks": 5,
            "seeds": 3,
            "framework_runs": 465,
            "seed_aggregated_dataset_framework_rows": 155,
            "primary_preference_cases": 3100,
        },
        "metadata_audit": metadata_audit,
        "synthetic_demo_files_in_evidence_tree": 0,
        "evidence_sha256": hashes,
        "statement": (
            "Phase-18 held-out empirical results and final statistical analysis "
            "are frozen as journal evidence. No post-outcome tuning or retraining "
            "is represented by this freeze manifest."
        ),
    }

    out = artifact_root / "evaluation" / "PHASE18_FROZEN_JOURNAL_EVIDENCE.json"
    out.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=" * 78)
    print("Phase 18 journal-evidence freeze: PASS")
    print("Manifest:", out)
    print("Git HEAD:", freeze["git_head"])
    print("Evidence files hashed:", len(hashes))
    print("=" * 78)


if __name__ == "__main__":
    main()
