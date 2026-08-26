from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from .v2_data import load_recommender_train, validate_recommender_train
from .v2_models import (
    available_model_specs,
    model_feature_columns,
    sanitize_predictions,
    target_column,
    target_direction,
)

PHASE6_TARGETS = ("accuracy", "runtime", "energy", "co2")


@dataclass
class BenchmarkResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selection: Dict[str, Dict[str, object]]


def _direction_sign(target: str) -> float:
    return 1.0 if target_direction(target) == "maximize" else -1.0


def _robust_scale(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    q05, q95 = np.quantile(arr, [0.05, 0.95])
    scale = float(q95 - q05)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(arr))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = max(1.0, abs(float(np.mean(arr))))
    return scale


def _safe_spearman(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return None
    rho = spearmanr(a, b).statistic
    if rho is None or not np.isfinite(rho):
        return None
    return float(rho)


def _per_dataset_ranking_metrics(frame: pd.DataFrame, target: str) -> Dict[str, float]:
    sign = _direction_sign(target)
    top1, top3, regrets, rhos = [], [], [], []

    for _, block in frame.groupby("dataset_id", sort=False):
        true_values = block["y_true"].to_numpy(dtype=float)
        pred_values = block["y_pred"].to_numpy(dtype=float)

        true_u = sign * true_values
        pred_u = sign * pred_values

        true_best = int(np.argmax(true_u))
        pred_order = np.argsort(-pred_u)
        pred_best = int(pred_order[0])

        top1.append(float(true_best == pred_best))
        top3.append(float(true_best in set(pred_order[:3].tolist())))

        span = float(np.max(true_u) - np.min(true_u))
        if span <= 1e-12:
            regret = 0.0
        else:
            regret = max(
                0.0,
                (float(true_u[true_best]) - float(true_u[pred_best])) / span,
            )
        regrets.append(regret)

        rho = _safe_spearman(true_u, pred_u)
        if rho is not None:
            rhos.append(rho)

    return {
        "top1_accuracy": float(np.mean(top1)),
        "top3_accuracy": float(np.mean(top3)),
        "normalized_regret": float(np.mean(regrets)),
        "spearman": float(np.mean(rhos)) if rhos else np.nan,
        "held_out_datasets": int(len(top1)),
        "spearman_defined_datasets": int(len(rhos)),
    }


def evaluate_predictions(predictions: pd.DataFrame, target: str, model_name: str):
    y_true = predictions["y_true"].to_numpy(dtype=float)
    y_pred = predictions["y_pred"].to_numpy(dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    r2 = float(r2_score(y_true, y_pred))
    scale = _robust_scale(y_true)

    ranking = _per_dataset_ranking_metrics(predictions, target)

    return {
        "target": target,
        "model": model_name,
        "direction": target_direction(target),
        "rows": int(len(predictions)),
        "mae": mae,
        "rmse": rmse,
        "normalized_mae": float(mae / scale),
        "normalized_rmse": float(rmse / scale),
        "r2": r2,
        **ranking,
    }


def evaluate_target_model(
    work: pd.DataFrame,
    target: str,
    spec,
    seed: int = 42,
):
    validate_recommender_train(work)

    features = model_feature_columns(True)
    y_col = target_column(target)
    groups = work["dataset_id"].astype(str).to_numpy()
    logo = LeaveOneGroupOut()
    rows = []

    for fold_index, (train_idx, test_idx) in enumerate(logo.split(work, groups=groups)):
        train = work.iloc[train_idx]
        test = work.iloc[test_idx]

        if set(train["dataset_id"].astype(str)) & set(test["dataset_id"].astype(str)):
            raise RuntimeError("Dataset leakage detected.")

        estimator = spec.factory(target, int(seed + fold_index))
        estimator.fit(
            train[features],
            pd.to_numeric(train[y_col], errors="raise").to_numpy(dtype=float),
        )
        pred = sanitize_predictions(target, estimator.predict(test[features]))

        for local_index, (_, row) in enumerate(test.iterrows()):
            rows.append(
                {
                    "fold": int(fold_index),
                    "held_out_dataset": str(row["dataset_id"]),
                    "dataset_id": str(row["dataset_id"]),
                    "framework": str(row["framework"]),
                    "target": target,
                    "model": spec.name,
                    "y_true": float(row[y_col]),
                    "y_pred": float(pred[local_index]),
                }
            )

    pred_frame = pd.DataFrame(rows)
    if len(pred_frame) != len(work):
        raise RuntimeError(
            "{} / {} produced {} predictions; expected {}.".format(
                target, spec.name, len(pred_frame), len(work)
            )
        )

    unique_keys = pred_frame[["dataset_id", "framework"]].drop_duplicates()
    if len(unique_keys) != len(work):
        raise RuntimeError(
            "{} / {} predictions are not unique per dataset/framework.".format(
                target, spec.name
            )
        )

    return pred_frame, evaluate_predictions(pred_frame, target, spec.name)


def select_models(metrics: pd.DataFrame, require_all_targets: bool = True):
    out = {}
    for target, block in metrics.groupby("target", sort=True):
        ranked = block.copy()
        ranked["_regret"] = pd.to_numeric(
            ranked["normalized_regret"], errors="coerce"
        ).fillna(np.inf)
        ranked["_top1"] = pd.to_numeric(
            ranked["top1_accuracy"], errors="coerce"
        ).fillna(-np.inf)
        ranked["_rho"] = pd.to_numeric(
            ranked["spearman"], errors="coerce"
        ).fillna(-np.inf)
        ranked["_nmae"] = pd.to_numeric(
            ranked["normalized_mae"], errors="coerce"
        ).fillna(np.inf)

        ranked = ranked.sort_values(
            ["_regret", "_top1", "_rho", "_nmae", "model"],
            ascending=[True, False, False, True, True],
        )
        best = ranked.iloc[0]
        out[str(target)] = {
            "model": str(best["model"]),
            "direction": str(best["direction"]),
            "normalized_regret": float(best["normalized_regret"]),
            "top1_accuracy": float(best["top1_accuracy"]),
            "top3_accuracy": float(best["top3_accuracy"]),
            "spearman": float(best["spearman"]) if pd.notna(best["spearman"]) else None,
            "mae": float(best["mae"]),
            "rmse": float(best["rmse"]),
            "normalized_mae": float(best["normalized_mae"]),
            "selection_rule": (
                "min normalized_regret; tie-break by max top1_accuracy, "
                "max spearman, min normalized_mae"
            ),
        }

    if require_all_targets:
        missing = sorted(set(PHASE6_TARGETS) - set(out))
        if missing:
            raise ValueError("Missing target selections: {}".format(missing))
    return out


def benchmark_v2_models(
    df: Optional[pd.DataFrame] = None,
    include_xgboost: bool = True,
    seed: int = 42,
    targets: Sequence[str] = PHASE6_TARGETS,
) -> BenchmarkResult:
    """Backward-compatible non-checkpointed API used by package exports/tests.

    The resumable command-line runner calls evaluate_target_model directly and
    checkpoints each target/model job. This function remains available so older
    imports and programmatic callers do not break.
    """
    work = load_recommender_train() if df is None else df.copy()
    validate_recommender_train(work)

    metric_rows = []
    prediction_frames = []
    specs = available_model_specs(include_xgboost=include_xgboost)

    for target in targets:
        for spec in specs:
            predictions, metric = evaluate_target_model(
                work,
                target,
                spec,
                seed=seed,
            )
            metric_rows.append(metric)
            prediction_frames.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    selection = select_models(
        metrics,
        require_all_targets=(set(targets) == set(PHASE6_TARGETS)),
    )

    return BenchmarkResult(
        metrics=metrics,
        predictions=predictions,
        selection=selection,
    )
