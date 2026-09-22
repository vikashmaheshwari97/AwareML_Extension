from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase18_common import (  # noqa: E402
    EXPECTED_AGGREGATED,
    EXPECTED_DATASETS,
    EXPECTED_RUNS,
    FRAMEWORKS,
    REDUCED_DIR,
    RUNS_DIR,
    SEEDS,
    load_frozen_dataset_manifest,
    load_task_manifest,
    sha256_file,
    utc_now,
    verify_frozen_checksums,
    write_json_atomic,
)


def nested_get(mapping: Any, *keys: str, default=None):
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_number(value: Any):
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def load_task_result(task_id: int, task_row: dict) -> Dict[str, Any]:
    task_dir = RUNS_DIR / f"task_{task_id:04d}"
    result_path = task_dir / "result.json"
    success_path = task_dir / "SUCCESS.json"
    if not result_path.exists() or not success_path.exists():
        raise FileNotFoundError(f"Task {task_id} is incomplete: {task_dir}")
    marker = json.loads(success_path.read_text(encoding="utf-8"))
    if marker.get("result_sha256") != sha256_file(result_path):
        raise RuntimeError(f"Task {task_id} result checksum does not match SUCCESS.json")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    task = payload.get("task") or {}
    result = payload.get("result") or {}
    if int(task.get("task_id", -1)) != task_id:
        raise RuntimeError(f"Task {task_id} embedded task_id mismatch")
    if str(task.get("dataset_id")) != str(task_row["dataset_id"]):
        raise RuntimeError(f"Task {task_id} dataset mismatch")
    if str(task.get("framework")) != str(task_row["framework"]):
        raise RuntimeError(f"Task {task_id} framework mismatch")
    if int(task.get("seed", -1)) != int(task_row["seed"]):
        raise RuntimeError(f"Task {task_id} seed mismatch")
    if str(result.get("status")) != "ok":
        raise RuntimeError(f"Task {task_id} status is not ok: {result.get('status')} {result.get('error')}")
    return payload


def flatten_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    task = payload["task"]
    result = payload["result"]
    fairness = result.get("fairness") or {}
    xai = result.get("explainability") or {}
    diag = result.get("prediction_diagnostics") or {}
    drift = result.get("drift_summary") or {}
    sustain = result.get("sustainability") or {}
    points = result.get("points") or []

    return {
        "task_id": int(task["task_id"]),
        "dataset_id": str(task["dataset_id"]),
        "filename": str(task["filename"]),
        "cohort": str(task["cohort"]),
        "dataset_sha256": str(task["dataset_sha256"]),
        "target": str(task["target"]),
        "source_target": str(task.get("source_target") or task["target"]),
        "task_policy": str(task.get("task_policy") or "native_classification"),
        "task_transform_sha256": str(task.get("task_transform_sha256") or ""),
        "sensitive_attribute": str(task.get("sensitive_attribute") or ""),
        "framework": str(task["framework"]),
        "seed": int(task["seed"]),
        "backend": str(result.get("backend") or ""),
        "status": str(result.get("status") or ""),
        "samples_processed": int(result.get("samples") or 0),
        "accuracy": as_number(result.get("accuracy")),
        "f1_macro": as_number(result.get("f1_macro")),
        "runtime_sec": as_number(result.get("runtime_sec")),
        "energy_kwh": as_number(result.get("energy_kwh")),
        "co2_kg": as_number(result.get("co2_kg")),
        "throughput_samples_sec": as_number(result.get("throughput_samples_sec")),
        "mean_prediction_latency_ms": as_number(result.get("mean_prediction_latency_ms")),
        "p95_prediction_latency_ms": as_number(result.get("p95_prediction_latency_ms")),
        "instrumentation_overhead_sec": as_number(result.get("instrumentation_overhead_sec")),
        "window_count": int(len(points)),
        "drift_event_count": int(len(result.get("drift_events") or [])),
        "drift_episode_count": int(drift.get("n_drift_events") or len(drift.get("episodes") or [])),
        "drift_recovery_applicable": int(drift.get("n_recovery_applicable") or 0),
        "drift_no_observed_degradation": int(drift.get("n_no_observed_degradation") or 0),
        "drift_recovered_count": int(drift.get("n_recovered") or 0),
        "drift_recovery_rate": as_number(drift.get("recovery_rate")),
        "drift_mean_recovery_samples": as_number(drift.get("mean_recovery_samples")),
        "drift_median_recovery_samples": as_number(drift.get("median_recovery_samples")),
        "drift_max_accuracy_drop": as_number(drift.get("max_accuracy_drop")),
        "drift_mean_accuracy_drop": as_number(drift.get("mean_accuracy_drop")),
        "fairness_status": str(fairness.get("status") or "not_requested"),
        "dp_diff": as_number(fairness.get("dp_diff")),
        "eo_diff": as_number(fairness.get("equal_opportunity_diff")),
        "equalized_odds_gap": as_number(fairness.get("equalized_odds_gap")),
        "predictive_parity_diff": as_number(fairness.get("predictive_parity_diff")),
        "error_rate_gap": as_number(fairness.get("error_rate_gap")),
        "group_brier_score_gap": as_number(fairness.get("group_brier_score_gap")),
        "group_ece_gap": as_number(fairness.get("group_ece_gap")),
        "fairness_calibration_status": str(fairness.get("calibration_status") or ""),
        "worst_group_accuracy": as_number(fairness.get("worst_group_accuracy")),
        "worst_group_macro_f1": as_number(fairness.get("worst_group_macro_f1")),
        "xai_status": str(xai.get("status") or ""),
        "xai_method": str(xai.get("method") or ""),
        "xai_stability": as_number(xai.get("stability")),
        "xai_fidelity": as_number(xai.get("fidelity")),
        "xai_sensitivity": as_number(xai.get("sensitivity")),
        "xai_consistency": as_number(xai.get("consistency")),
        "xai_sparsity": as_number(xai.get("sparsity")),
        "prediction_pattern": str(diag.get("pattern") or ""),
        "prediction_coverage": as_number(diag.get("prediction_coverage")),
        "unique_predicted_labels": as_number(diag.get("unique_predicted_labels")),
        "majority_prediction_fraction": as_number(diag.get("majority_prediction_fraction")),
        "positive_prediction_rate": as_number(diag.get("positive_prediction_rate")),
        "near_constant_prediction": float(bool(diag.get("near_constant_prediction", False))),
        "sustainability_status": str(sustain.get("status") or ""),
        "codecarbon_version": str(sustain.get("codecarbon_version") or ""),
        "sustainability_backend": str(sustain.get("measurement_backend") or ""),
        "sustainability_country_iso": str(sustain.get("country_iso") or ""),
        "completed_utc": str(payload.get("completed_utc") or ""),
    }


def validate_run_table(df: pd.DataFrame, allow_missing_sustainability: bool) -> None:
    if len(df) != EXPECTED_RUNS:
        raise RuntimeError(f"Collected {len(df)} rows; expected {EXPECTED_RUNS}.")
    keys = ["dataset_id", "framework", "seed"]
    if df[keys].duplicated().any():
        raise RuntimeError("Duplicate dataset/framework/seed rows detected.")
    if int(df["dataset_id"].nunique()) != EXPECTED_DATASETS:
        raise RuntimeError("Run table does not contain 31 datasets.")
    if set(df["framework"].unique()) != set(FRAMEWORKS):
        raise RuntimeError("Run table framework set is incomplete.")
    if set(int(v) for v in df["seed"].unique()) != set(SEEDS):
        raise RuntimeError("Run table seed set is incomplete.")

    acc = pd.to_numeric(df["accuracy"], errors="coerce")
    if acc.isna().any() or ((acc < 0) | (acc > 1)).any():
        raise RuntimeError("Accuracy must be finite and in [0,1] for all 465 runs.")
    rt = pd.to_numeric(df["runtime_sec"], errors="coerce")
    if rt.isna().any() or (rt <= 0).any():
        raise RuntimeError("Runtime must be finite and positive for all 465 runs.")
    if (pd.to_numeric(df["samples_processed"], errors="coerce") <= 0).any():
        raise RuntimeError("Every final run must process at least one sample.")

    for col in ("energy_kwh", "co2_kg"):
        vals = pd.to_numeric(df[col], errors="coerce")
        missing_or_nonpositive = vals.isna() | (vals <= 0)
        if missing_or_nonpositive.any() and not allow_missing_sustainability:
            bad = df.loc[missing_or_nonpositive, ["task_id", "dataset_id", "framework", "seed", col, "sustainability_status"]]
            raise RuntimeError(
                f"Final paper-ready gate failed: {col} is missing/non-positive in "
                f"{int(missing_or_nonpositive.sum())} runs.\n{bad.head(30).to_string(index=False)}"
            )


def aggregate_runs(run_df: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy", "f1_macro", "runtime_sec", "energy_kwh", "co2_kg",
        "samples_processed", "throughput_samples_sec", "mean_prediction_latency_ms",
        "p95_prediction_latency_ms", "instrumentation_overhead_sec", "drift_event_count",
        "drift_episode_count", "drift_recovery_applicable", "drift_no_observed_degradation",
        "drift_recovered_count", "drift_recovery_rate", "drift_mean_recovery_samples",
        "drift_median_recovery_samples", "drift_max_accuracy_drop", "drift_mean_accuracy_drop",
        "dp_diff", "eo_diff", "equalized_odds_gap", "predictive_parity_diff",
        "error_rate_gap", "group_brier_score_gap", "group_ece_gap",
        "worst_group_accuracy", "worst_group_macro_f1", "xai_stability", "xai_fidelity",
        "xai_sensitivity", "xai_consistency", "xai_sparsity",
        "prediction_coverage", "unique_predicted_labels", "majority_prediction_fraction",
        "positive_prediction_rate", "near_constant_prediction",
    ]
    rows: List[Dict[str, Any]] = []
    for (dataset_id, framework), block in run_df.groupby(["dataset_id", "framework"], sort=True):
        seeds = sorted(int(x) for x in block["seed"].tolist())
        if seeds != list(SEEDS):
            raise RuntimeError(f"{dataset_id}/{framework} has seeds {seeds}, expected {list(SEEDS)}")
        row: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "framework": framework,
            "seed_count": len(block),
            "seed_values": ",".join(str(s) for s in seeds),
            "cohort": str(block["cohort"].iloc[0]),
            "filename": str(block["filename"].iloc[0]),
            "target": str(block["target"].iloc[0]),
            "sensitive_attribute": str(block["sensitive_attribute"].iloc[0]),
            "backend_values_json": json.dumps(sorted(set(block["backend"].astype(str).tolist()))),
            "fairness_status_values_json": json.dumps(sorted(set(block["fairness_status"].astype(str).tolist()))),
            "fairness_calibration_status_values_json": json.dumps(sorted(set(block["fairness_calibration_status"].astype(str).tolist()))),
            "xai_status_values_json": json.dumps(sorted(set(block["xai_status"].astype(str).tolist()))),
            "xai_method_values_json": json.dumps(sorted(set(block["xai_method"].astype(str).tolist()))),
        }
        for metric in metrics:
            vals = pd.to_numeric(block[metric], errors="coerce")
            finite = vals[np.isfinite(vals)]
            row[metric + "_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[metric + "_std"] = float(finite.std(ddof=1)) if len(finite) >= 2 else np.nan
            row[metric + "_min"] = float(finite.min()) if len(finite) else np.nan
            row[metric + "_max"] = float(finite.max()) if len(finite) else np.nan
        rows.append(row)

    agg = pd.DataFrame(rows)
    if len(agg) != EXPECTED_AGGREGATED:
        raise RuntimeError(f"Aggregated table has {len(agg)} rows; expected {EXPECTED_AGGREGATED}.")

    profile_cols = [
        "dataset_id", "sha256", "task_policy", "resolved_target", "evaluation_target", "task_transform_sha256", "n_samples_dataset", "n_features", "n_numeric_features",
        "n_categorical_features", "numeric_feature_fraction", "categorical_feature_fraction",
        "missing_fraction", "n_classes", "majority_class_fraction", "minority_class_fraction",
        "class_imbalance_ratio", "class_entropy_normalized", "dataset_family", "source_type", "drift_type",
    ]
    available = [c for c in profile_cols if c in frozen.columns]
    profile = frozen[available].copy().rename(columns={"sha256": "dataset_sha256"})
    agg = agg.merge(profile, on="dataset_id", how="left", validate="many_to_one")
    agg["window_size"] = 1000
    agg["time_budget_sec"] = 60.0
    agg["max_samples_requested"] = 30000
    return agg.sort_values(["dataset_id", "framework"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly collect the 465 Phase-18 framework results and reduce to 155 seed means.")
    parser.add_argument(
        "--allow-missing-sustainability",
        action="store_true",
        help="Debug only. Produces NOT_PAPER_READY outputs if energy/CO2 are incomplete.",
    )
    args = parser.parse_args()

    verify_frozen_checksums()
    tasks = load_task_manifest()
    frozen = load_frozen_dataset_manifest()

    rows = []
    for task_id, task_row in tasks.iterrows():
        payload = load_task_result(int(task_id), task_row.to_dict())
        rows.append(flatten_payload(payload))
    run_df = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)
    validate_run_table(run_df, allow_missing_sustainability=args.allow_missing_sustainability)
    agg = aggregate_runs(run_df, frozen)

    REDUCED_DIR.mkdir(parents=True, exist_ok=True)
    run_csv = REDUCED_DIR / "phase18_runs_465.csv"
    run_parquet = REDUCED_DIR / "phase18_runs_465.parquet"
    agg_csv = REDUCED_DIR / "phase18_framework_ground_truth_155.csv"
    agg_parquet = REDUCED_DIR / "phase18_framework_ground_truth_155.parquet"

    run_df.to_csv(run_csv, index=False)
    run_df.to_parquet(run_parquet, index=False, engine="pyarrow", compression="zstd")
    agg.to_csv(agg_csv, index=False)
    agg.to_parquet(agg_parquet, index=False, engine="pyarrow", compression="zstd")

    cohort_summary = (
        agg.groupby("cohort", dropna=False)
        .agg(datasets=("dataset_id", "nunique"), rows=("dataset_id", "size"))
        .reset_index()
    )
    cohort_summary.to_csv(REDUCED_DIR / "phase18_ground_truth_cohort_counts.csv", index=False)

    complete_sustainability = bool(
        pd.to_numeric(run_df["energy_kwh"], errors="coerce").notna().all()
        and pd.to_numeric(run_df["co2_kg"], errors="coerce").notna().all()
        and (pd.to_numeric(run_df["energy_kwh"], errors="coerce") > 0).all()
        and (pd.to_numeric(run_df["co2_kg"], errors="coerce") > 0).all()
    )
    manifest = {
        "schema_version": "2.0",
        "phase": 18,
        "created_utc": utc_now(),
        "paper_ready": bool(complete_sustainability and not args.allow_missing_sustainability),
        "allow_missing_sustainability": bool(args.allow_missing_sustainability),
        "counts": {
            "run_rows": int(len(run_df)),
            "datasets": int(run_df["dataset_id"].nunique()),
            "frameworks": int(run_df["framework"].nunique()),
            "seeds": int(run_df["seed"].nunique()),
            "aggregated_rows": int(len(agg)),
        },
        "outputs": {
            run_csv.name: sha256_file(run_csv),
            run_parquet.name: sha256_file(run_parquet),
            agg_csv.name: sha256_file(agg_csv),
            agg_parquet.name: sha256_file(agg_parquet),
        },
    }
    write_json_atomic(REDUCED_DIR / "collection_manifest.json", manifest)

    print("=" * 76)
    print("AwareML Phase 18 result collection: SUCCESS")
    print("=" * 76)
    print("Run rows:", len(run_df), "(expected 465)")
    print("Aggregated dataset/framework rows:", len(agg), "(expected 155)")
    print("Datasets:", run_df["dataset_id"].nunique())
    print("Cohorts:")
    print(cohort_summary.to_string(index=False))
    print("Paper-ready sustainability gate:", "PASS" if manifest["paper_ready"] else "NOT PAPER READY")
    print("Wrote:", run_parquet)
    print("Wrote:", agg_parquet)


if __name__ == "__main__":
    main()
