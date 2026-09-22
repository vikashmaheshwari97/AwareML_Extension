from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase18_common import (  # noqa: E402
    EVAL_DIR,
    EXPECTED_AGGREGATED,
    EXPECTED_DATASETS,
    FRAMEWORKS,
    REDUCED_DIR,
    sha256_file,
    utc_now,
    verify_frozen_checksums,
    write_json_atomic,
)

PERFORMANCE_METRICS = [
    "accuracy_mean",
    "f1_macro_mean",
    "runtime_sec_mean",
    "throughput_samples_sec_mean",
    "mean_prediction_latency_ms_mean",
    "p95_prediction_latency_ms_mean",
    "energy_kwh_mean",
    "co2_kg_mean",
]

RESPONSIBLE_AI_METRICS = [
    "drift_event_count_mean",
    "drift_recovery_rate_mean",
    "drift_mean_recovery_samples_mean",
    "drift_max_accuracy_drop_mean",
    "dp_diff_mean",
    "eo_diff_mean",
    "equalized_odds_gap_mean",
    "predictive_parity_diff_mean",
    "group_brier_score_gap_mean",
    "group_ece_gap_mean",
    "worst_group_accuracy_mean",
    "worst_group_macro_f1_mean",
    "xai_stability_mean",
    "xai_fidelity_mean",
    "xai_sensitivity_mean",
    "xai_consistency_mean",
    "xai_sparsity_mean",
    "prediction_coverage_mean",
    "near_constant_prediction_mean",
]

OBJECTIVE_COLUMNS = {
    "accuracy": ("accuracy_mean", "max"),
    "runtime": ("runtime_sec_mean", "min"),
    "energy": ("energy_kwh_mean", "min"),
    "co2": ("co2_kg_mean", "min"),
}


def _finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def _summarize_block(block: pd.DataFrame, metrics: Iterable[str]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "datasets": int(block["dataset_id"].nunique()),
        "dataset_framework_rows": int(len(block)),
    }
    for metric in metrics:
        if metric not in block.columns:
            continue
        values = _finite(block[metric])
        row[metric + "__n"] = int(len(values))
        row[metric + "__mean"] = float(values.mean()) if len(values) else np.nan
        row[metric + "__sd"] = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
        row[metric + "__median"] = float(values.median()) if len(values) else np.nan
    return row


def _with_combined(frame: pd.DataFrame) -> pd.DataFrame:
    combined = frame.copy()
    combined["cohort"] = "combined_31"
    return pd.concat([frame, combined], ignore_index=True)


def build_summary(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    work = _with_combined(frame)
    rows: List[Dict[str, object]] = []
    for (cohort, framework), block in work.groupby(["cohort", "framework"], sort=True):
        row = {"cohort": str(cohort), "framework": str(framework)}
        row.update(_summarize_block(block, metrics))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cohort", "framework"]).reset_index(drop=True)


def build_objective_winner_table(gt: pd.DataFrame, pred: pd.DataFrame | None) -> pd.DataFrame:
    pred_index = None
    if pred is not None:
        pred_index = pred.set_index(["dataset_id", "framework"])

    rows: List[Dict[str, object]] = []
    for dataset_id, block in gt.groupby("dataset_id", sort=True):
        cohort = str(block["cohort"].iloc[0])
        for objective, (column, direction) in OBJECTIVE_COLUMNS.items():
            values = pd.to_numeric(block[column], errors="raise")
            observed_idx = values.idxmax() if direction == "max" else values.idxmin()
            observed = str(block.loc[observed_idx, "framework"])

            predicted = None
            top1_match = None
            if pred_index is not None:
                pred_column = objective
                candidates = []
                for framework in FRAMEWORKS:
                    value = float(pred_index.loc[(str(dataset_id), framework), pred_column])
                    candidates.append((framework, value))
                if direction == "max":
                    predicted = max(candidates, key=lambda item: (item[1], item[0]))[0]
                else:
                    predicted = min(candidates, key=lambda item: (item[1], item[0]))[0]
                top1_match = bool(predicted == observed)

            rows.append({
                "dataset_id": str(dataset_id),
                "cohort": cohort,
                "objective": objective,
                "observed_best_framework": observed,
                "predicted_best_framework": predicted,
                "top1_match": top1_match,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build journal-facing Phase-18 tables from the strict 155-row held-out ground truth."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=REDUCED_DIR / "phase18_framework_ground_truth_155.parquet",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=EVAL_DIR / "phase18_frozen_v2_predictions_155.parquet",
    )
    args = parser.parse_args()

    verify_frozen_checksums()
    gt_path = args.ground_truth.resolve()
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing Phase-18 ground truth: {gt_path}")
    gt = pd.read_parquet(gt_path)
    if len(gt) != EXPECTED_AGGREGATED or int(gt["dataset_id"].nunique()) != EXPECTED_DATASETS:
        raise RuntimeError("Expected exactly 155 ground-truth rows covering 31 datasets.")

    pred = None
    pred_path = args.predictions.resolve()
    if pred_path.exists():
        pred = pd.read_parquet(pred_path)
        if len(pred) != EXPECTED_AGGREGATED:
            raise RuntimeError("Frozen V2 prediction table must contain 155 rows.")

    out_dir = EVAL_DIR / "journal_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    perf = build_summary(gt, PERFORMANCE_METRICS)
    rai = build_summary(gt, RESPONSIBLE_AI_METRICS)
    winners = build_objective_winner_table(gt, pred)

    perf_path = out_dir / "phase18_performance_sustainability_summary.csv"
    rai_path = out_dir / "phase18_responsible_ai_summary.csv"
    winners_path = out_dir / "phase18_predicted_vs_observed_objective_winners.csv"
    perf.to_csv(perf_path, index=False)
    rai.to_csv(rai_path, index=False)
    winners.to_csv(winners_path, index=False)

    manifest = {
        "schema_version": "1.0",
        "phase": 18,
        "created_utc": utc_now(),
        "ground_truth": str(gt_path.relative_to(ROOT)),
        "ground_truth_sha256": sha256_file(gt_path),
        "predictions_used": pred is not None,
        "predictions": str(pred_path.relative_to(ROOT)) if pred is not None else None,
        "outputs": {
            perf_path.name: sha256_file(perf_path),
            rai_path.name: sha256_file(rai_path),
            winners_path.name: sha256_file(winners_path),
        },
        "aggregation_note": (
            "Summary means/SDs are computed across the 31 dataset-level seed-aggregated framework rows, "
            "not across the 465 seed executions as if those were independent datasets. Missing fairness/XAI "
            "metrics are not converted to zero."
        ),
    }
    write_json_atomic(out_dir / "journal_tables_manifest.json", manifest)

    print("=" * 78)
    print("AwareML Phase 18 journal tables: SUCCESS")
    print("=" * 78)
    print("Performance/sustainability:", perf_path)
    print("Responsible-AI summary:", rai_path)
    print("Predicted vs observed winners:", winners_path)
    if pred is None:
        print("Note: frozen recommender predictions were not found; winner table has observed winners only.")


if __name__ == "__main__":
    main()
