from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from awareml.recommender.v2_evaluation import evaluate_predictions  # noqa: E402
from awareml.recommender.v2_ranking import rank_candidates  # noqa: E402
from awareml.recommender.v2_service import V2Recommender  # noqa: E402
from phase18_common import (  # noqa: E402
    EVAL_DIR,
    EXPECTED_AGGREGATED,
    EXPECTED_DATASETS,
    EXPECTED_PREFERENCE_CASES,
    FRAMEWORKS,
    REDUCED_DIR,
    active_recommender_manifest,
    load_frozen_dataset_manifest,
    load_preference_manifest,
    sha256_file,
    utc_now,
    verify_frozen_checksums,
    write_json_atomic,
)

REAL_COLUMNS = {
    "accuracy": "accuracy_mean",
    "runtime": "runtime_sec_mean",
    "energy": "energy_kwh_mean",
    "co2": "co2_kg_mean",
}


def finite_spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return np.nan
    value = spearmanr(a, b).statistic
    return float(value) if value is not None and np.isfinite(value) else np.nan


def profile_from_frozen_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_family": str(row.get("dataset_family") or "unknown"),
        "source_type": str(row.get("source_type") or "unknown"),
        "drift_type": str(row.get("drift_type") or "unknown"),
        "n_samples_dataset": int(float(row["n_samples_dataset"])),
        "n_features": int(float(row["n_features"])),
        "n_numeric_features": int(float(row["n_numeric_features"])),
        "n_categorical_features": int(float(row["n_categorical_features"])),
        "numeric_feature_fraction": float(row["numeric_feature_fraction"]),
        "categorical_feature_fraction": float(row["categorical_feature_fraction"]),
        "missing_fraction": float(row["missing_fraction"]),
        "n_classes": float(row["n_classes"]),
        "majority_class_fraction": float(row["majority_class_fraction"]),
        "minority_class_fraction": float(row["minority_class_fraction"]),
        "class_imbalance_ratio": float(row["class_imbalance_ratio"]),
        "class_entropy_normalized": float(row["class_entropy_normalized"]),
        "window_size": 1000,
        "time_budget_sec": 60.0,
    }


def build_frozen_predictions(frozen: pd.DataFrame, recommender: V2Recommender) -> pd.DataFrame:
    rows = []
    for _, ds in frozen.iterrows():
        profile = profile_from_frozen_row(ds)
        predicted = recommender.predict_profile(profile, coverage=0.90)
        predicted.insert(0, "dataset_id", str(ds["dataset_id"]))
        predicted.insert(1, "cohort", str(ds["cohort"]))
        rows.append(predicted)
    out = pd.concat(rows, ignore_index=True)
    if len(out) != EXPECTED_AGGREGATED:
        raise RuntimeError(f"Frozen V2 produced {len(out)} rows; expected {EXPECTED_AGGREGATED}.")
    if out[["dataset_id", "framework"]].duplicated().any():
        raise RuntimeError("Frozen V2 predictions are not unique by dataset/framework.")
    return out


def objective_evaluation(gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, true_col in REAL_COLUMNS.items():
        merged = gt[["dataset_id", "framework", true_col]].merge(
            pred[["dataset_id", "framework", target, target + "_lower", target + "_upper"]],
            on=["dataset_id", "framework"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != EXPECTED_AGGREGATED:
            raise RuntimeError(f"{target}: expected 155 matched rows, found {len(merged)}")
        eval_frame = merged.rename(columns={true_col: "y_true", target: "y_pred"})
        metric = evaluate_predictions(eval_frame, target=target, model_name="frozen_v2")
        covered = (
            pd.to_numeric(merged[true_col], errors="raise")
            >= pd.to_numeric(merged[target + "_lower"], errors="raise")
        ) & (
            pd.to_numeric(merged[true_col], errors="raise")
            <= pd.to_numeric(merged[target + "_upper"], errors="raise")
        )
        metric["interval_90_coverage_heldout"] = float(covered.mean())
        metric["interval_90_rows"] = int(len(covered))
        rows.append(metric)
    return pd.DataFrame(rows)


def real_candidates(block: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "framework": block["framework"].astype(str).tolist(),
        "accuracy": pd.to_numeric(block["accuracy_mean"], errors="raise").to_numpy(float),
        "runtime": pd.to_numeric(block["runtime_sec_mean"], errors="raise").to_numpy(float),
        "energy": pd.to_numeric(block["energy_kwh_mean"], errors="raise").to_numpy(float),
        "co2": pd.to_numeric(block["co2_kg_mean"], errors="raise").to_numpy(float),
    })


def preference_weights(row: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "accuracy": float(row["w_accuracy"]),
        "runtime": float(row["w_runtime"]),
        "energy": float(row["w_energy"]),
        "co2": float(row["w_co2"]),
    }


def primary_preference_cases(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    prefs: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    gt_by = {str(ds): block.copy() for ds, block in gt.groupby("dataset_id", sort=False)}
    pred_by = {str(ds): block.copy() for ds, block in pred.groupby("dataset_id", sort=False)}

    for _, pref in prefs.iterrows():
        ds = str(pref["dataset_id"])
        obs = real_candidates(gt_by[ds])
        predicted = pred_by[ds][["framework", "accuracy", "runtime", "energy", "co2"]].copy()
        w = preference_weights(pref)
        pred_rank, _ = rank_candidates(predicted, weights=w, mode="point")
        real_rank, _ = rank_candidates(obs, weights=w, mode="point")

        rec_fw = str(pred_rank.iloc[0]["framework"])
        oracle_fw = str(real_rank.iloc[0]["framework"])
        pred_top3 = pred_rank.head(3)["framework"].astype(str).tolist()
        real_u = real_rank.set_index("framework")["utility"].astype(float)
        pred_u = pred_rank.set_index("framework")["utility"].astype(float)
        u_best = float(real_u.loc[oracle_fw])
        u_rec = float(real_u.loc[rec_fw])
        span = float(real_u.max() - real_u.min())
        regret_span = max(0.0, (u_best - u_rec) / span) if span > 1e-12 else 0.0
        regret_best = max(0.0, (u_best - u_rec) / u_best) if abs(u_best) > 1e-12 else 0.0
        aligned = sorted(set(real_u.index) & set(pred_u.index))
        rho = finite_spearman([real_u[x] for x in aligned], [pred_u[x] for x in aligned])

        rows.append({
            "dataset_id": ds,
            "cohort": str(pref["cohort"]),
            "preference_id": int(pref["preference_id"]),
            "k_prime": int(pref["k_prime"]),
            "active_objectives": str(pref["active_objectives"]),
            **w,
            "oracle_framework": oracle_fw,
            "recommended_framework": rec_fw,
            "top1_match": float(rec_fw == oracle_fw),
            "top3_contains_oracle": float(oracle_fw in pred_top3),
            "oracle_real_utility": u_best,
            "recommended_real_utility": u_rec,
            "normalized_regret_span": float(regret_span),
            "normalized_regret_cikm_denominator": float(regret_best),
            "spearman_utility": rho,
            "evaluation_mode": "primary_frozen_v2_predicted_scale_only",
        })

    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_PREFERENCE_CASES:
        raise RuntimeError(f"Primary preference evaluation has {len(out)} rows; expected 3100.")
    return out


def normalize_with_observed_bounds(values, lo: float, hi: float, maximize: bool, clip: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if abs(hi - lo) <= 1e-12:
        out = np.full_like(arr, 0.5, dtype=float)
    else:
        out = (arr - lo) / (hi - lo)
        if not maximize:
            out = 1.0 - out
    return np.clip(out, 0.0, 1.0) if clip else out


def cikm_compatible_cases(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    prefs: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    gt_by = {str(ds): real_candidates(block) for ds, block in gt.groupby("dataset_id", sort=False)}
    pred_by = {
        str(ds): block[["framework", "accuracy", "runtime", "energy", "co2"]].copy()
        for ds, block in pred.groupby("dataset_id", sort=False)
    }

    for _, pref in prefs.iterrows():
        ds = str(pref["dataset_id"])
        obs = gt_by[ds].set_index("framework").loc[list(FRAMEWORKS)]
        prd = pred_by[ds].set_index("framework").loc[list(FRAMEWORKS)]
        scores_real = {}
        scores_pred = {}
        for objective in ("accuracy", "runtime", "energy", "co2"):
            real_values = obs[objective].to_numpy(float)
            pred_values = prd[objective].to_numpy(float)
            lo, hi = float(real_values.min()), float(real_values.max())
            maximize = objective == "accuracy"
            scores_real[objective] = normalize_with_observed_bounds(real_values, lo, hi, maximize, clip=False)
            scores_pred[objective] = normalize_with_observed_bounds(pred_values, lo, hi, maximize, clip=True)

        w = preference_weights(pref)
        u_real = sum(w[obj] * scores_real[obj] for obj in w)
        u_pred = sum(w[obj] * scores_pred[obj] for obj in w)
        true_idx = int(np.argmax(u_real))
        pred_order = np.argsort(-u_pred)
        rec_idx = int(pred_order[0])
        frameworks = list(FRAMEWORKS)
        oracle_fw = frameworks[true_idx]
        rec_fw = frameworks[rec_idx]
        u_best = float(u_real[true_idx])
        u_rec = float(u_real[rec_idx])
        regret = max(0.0, (u_best - u_rec) / u_best) if abs(u_best) > 1e-12 else 0.0
        rho = finite_spearman(u_real, u_pred)

        rows.append({
            "dataset_id": ds,
            "cohort": str(pref["cohort"]),
            "preference_id": int(pref["preference_id"]),
            "k_prime": int(pref["k_prime"]),
            "active_objectives": str(pref["active_objectives"]),
            **w,
            "oracle_framework": oracle_fw,
            "recommended_framework": rec_fw,
            "top1_match": float(rec_fw == oracle_fw),
            "top3_contains_oracle": float(true_idx in set(pred_order[:3].tolist())),
            "oracle_real_utility": u_best,
            "recommended_real_utility": u_rec,
            "normalized_regret": float(regret),
            "spearman_utility": rho,
            "evaluation_mode": "secondary_historical_cikm_compatible_real_minmax",
        })
    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_PREFERENCE_CASES:
        raise RuntimeError(f"CIKM-compatible preference evaluation has {len(out)} rows; expected 3100.")
    return out


def summarize_cases(frame: pd.DataFrame, regret_col: str, mode: str) -> pd.DataFrame:
    rows = []
    grouping_specs = [
        ("overall", []),
        ("cohort", ["cohort"]),
        ("k_prime", ["k_prime"]),
        ("cohort_x_k_prime", ["cohort", "k_prime"]),
    ]
    for level, cols in grouping_specs:
        iterator = [((), frame)] if not cols else frame.groupby(cols, dropna=False, sort=True)
        for key, block in iterator:
            if not isinstance(key, tuple):
                key = (key,)
            meta = {col: value for col, value in zip(cols, key)}
            rows.append({
                "evaluation_mode": mode,
                "summary_level": level,
                **meta,
                "cases": int(len(block)),
                "datasets": int(block["dataset_id"].nunique()),
                "top1": float(block["top1_match"].mean()),
                "top3": float(block["top3_contains_oracle"].mean()),
                "normalized_regret_mean": float(block[regret_col].mean()),
                "normalized_regret_median": float(block[regret_col].median()),
                "normalized_regret_p90": float(block[regret_col].quantile(0.90)),
                "spearman_mean": float(pd.to_numeric(block["spearman_utility"], errors="coerce").mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the already-frozen V2 recommender on Phase-18 held-out ground truth. "
            "This script never fits/retrains a recommender model."
        )
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=REDUCED_DIR / "phase18_framework_ground_truth_155.parquet",
    )
    args = parser.parse_args()

    verify_frozen_checksums()
    ground_truth_path = args.ground_truth.resolve()
    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Missing reduced ground truth: {ground_truth_path}. Run scripts/phase18_collect_results.py first."
        )

    gt = pd.read_parquet(ground_truth_path)
    if len(gt) != EXPECTED_AGGREGATED or gt[["dataset_id", "framework"]].duplicated().any():
        raise RuntimeError("Ground-truth table must contain exactly 155 unique dataset/framework rows.")
    if int(gt["dataset_id"].nunique()) != EXPECTED_DATASETS:
        raise RuntimeError("Ground-truth table must contain exactly 31 datasets.")
    for col in REAL_COLUMNS.values():
        vals = pd.to_numeric(gt[col], errors="coerce")
        if vals.isna().any():
            raise RuntimeError(f"Ground-truth objective {col} contains missing values.")

    frozen = load_frozen_dataset_manifest()
    prefs = load_preference_manifest()
    recommender_manifest = active_recommender_manifest()
    if recommender_manifest is None:
        raise RuntimeError("Frozen V2 active recommender manifest is unavailable.")

    # Important: instantiate and predict only. There is deliberately no estimator.fit call here.
    recommender = V2Recommender(root=ROOT)
    pred = build_frozen_predictions(frozen, recommender)
    objective_metrics = objective_evaluation(gt, pred)
    primary = primary_preference_cases(gt, pred, prefs)
    compat = cikm_compatible_cases(gt, pred, prefs)
    summaries = pd.concat(
        [
            summarize_cases(primary, "normalized_regret_span", "primary_frozen_v2_predicted_scale_only"),
            summarize_cases(compat, "normalized_regret", "secondary_historical_cikm_compatible_real_minmax"),
        ],
        ignore_index=True,
    )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pred_csv = EVAL_DIR / "phase18_frozen_v2_predictions_155.csv"
    pred_parquet = EVAL_DIR / "phase18_frozen_v2_predictions_155.parquet"
    objective_path = EVAL_DIR / "phase18_objective_metrics.csv"
    primary_path = EVAL_DIR / "phase18_preference_eval_primary_3100.csv"
    compat_path = EVAL_DIR / "phase18_preference_eval_cikm_compat_3100.csv"
    summary_path = EVAL_DIR / "phase18_preference_summary.csv"

    pred.to_csv(pred_csv, index=False)
    pred.to_parquet(pred_parquet, index=False, engine="pyarrow", compression="zstd")
    objective_metrics.to_csv(objective_path, index=False)
    primary.to_csv(primary_path, index=False)
    compat.to_csv(compat_path, index=False)
    summaries.to_csv(summary_path, index=False)

    overall_primary = summaries[
        (summaries["evaluation_mode"] == "primary_frozen_v2_predicted_scale_only")
        & (summaries["summary_level"] == "overall")
    ].iloc[0].to_dict()
    overall_compat = summaries[
        (summaries["evaluation_mode"] == "secondary_historical_cikm_compatible_real_minmax")
        & (summaries["summary_level"] == "overall")
    ].iloc[0].to_dict()

    final = {
        "schema_version": "2.0",
        "phase": 18,
        "created_utc": utc_now(),
        "recommender_retrained": False,
        "recommender_manifest": str(recommender_manifest.relative_to(ROOT)),
        "recommender_manifest_sha256": sha256_file(recommender_manifest),
        "ground_truth_file": str(ground_truth_path.relative_to(ROOT)),
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "counts": {
            "datasets": int(gt["dataset_id"].nunique()),
            "framework_ground_truth_rows": int(len(gt)),
            "frozen_v2_prediction_rows": int(len(pred)),
            "preference_cases_primary": int(len(primary)),
            "preference_cases_cikm_compatible": int(len(compat)),
        },
        "objective_metrics": objective_metrics.to_dict(orient="records"),
        "primary_preference_overall": overall_primary,
        "secondary_cikm_compatible_overall": overall_compat,
        "interpretation": {
            "primary": (
                "Framework recommendations are computed only from frozen V2 predictions; predicted objective "
                "values are normalized across the five predicted candidates. Held-out outcomes are used only "
                "after recommendation to define the oracle and regret."
            ),
            "secondary": (
                "CIKM-compatible analysis applies each held-out dataset's observed objective min/max to both "
                "observed and predicted values, matching the historical evaluator. Because observed scaling "
                "enters the recommendation calculation, this result is secondary rather than the journal-primary estimate."
            ),
        },
        "outputs": {
            pred_csv.name: sha256_file(pred_csv),
            pred_parquet.name: sha256_file(pred_parquet),
            objective_path.name: sha256_file(objective_path),
            primary_path.name: sha256_file(primary_path),
            compat_path.name: sha256_file(compat_path),
            summary_path.name: sha256_file(summary_path),
        },
    }
    final_path = EVAL_DIR / "phase18_final_evaluation_summary.json"
    write_json_atomic(final_path, final)

    print("=" * 78)
    print("AwareML Phase 18 frozen-recommender evaluation: SUCCESS")
    print("=" * 78)
    print("Frozen V2 retrained: NO")
    print("Datasets:", gt["dataset_id"].nunique())
    print("Ground-truth framework rows:", len(gt))
    print("Preference cases:", len(primary))
    print()
    print("Objective metrics:")
    print(
        objective_metrics[
            [
                "target", "top1_accuracy", "top3_accuracy", "normalized_regret", "spearman",
                "mae", "rmse", "interval_90_coverage_heldout",
            ]
        ].to_string(index=False)
    )
    print()
    print("Primary 3,100-case preference evaluation:")
    print(
        "Top-1={:.4f} Top-3={:.4f} Regret={:.4f} Spearman={:.4f}".format(
            float(overall_primary["top1"]),
            float(overall_primary["top3"]),
            float(overall_primary["normalized_regret_mean"]),
            float(overall_primary["spearman_mean"]),
        )
    )
    print("Secondary historical CIKM-compatible result is saved separately.")
    print("Summary:", final_path)


if __name__ == "__main__":
    main()
