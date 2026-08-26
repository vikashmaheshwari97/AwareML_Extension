from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_backend
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.recommender.v2_data import load_recommender_train
from awareml.recommender.v2_models import (
    available_model_specs,
    model_feature_columns,
    sanitize_predictions,
    target_column,
    target_direction,
)

PHASE6_TARGETS = ("accuracy", "runtime", "energy", "co2")
SNAPSHOT_DIR = ROOT / "data" / "meta" / "snapshots"
CHECKPOINT_DIR = SNAPSHOT_DIR / "phase6_2_checkpoints"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_checksum(path: Path) -> None:
    Path(str(path) + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n",
        encoding="utf-8",
    )


def safe_spearman(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return None
    rho = spearmanr(a, b).statistic
    if rho is None or not np.isfinite(rho):
        return None
    return float(rho)


def robust_scale(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    q05, q95 = np.quantile(arr, [0.05, 0.95])
    scale = float(q95 - q05)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(arr))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = max(1.0, abs(float(np.mean(arr))))
    return scale


def evaluate_predictions(predictions: pd.DataFrame, target: str, model_name: str):
    y_true = predictions["y_true"].to_numpy(dtype=float)
    y_pred = predictions["y_pred"].to_numpy(dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = np.nan

    sign = 1.0 if target_direction(target) == "maximize" else -1.0
    top1, top3, regrets, rhos = [], [], [], []

    for _, block in predictions.groupby("dataset_id", sort=False):
        true_values = block["y_true"].to_numpy(dtype=float)
        pred_values = block["y_pred"].to_numpy(dtype=float)
        utility_true = sign * true_values
        utility_pred = sign * pred_values

        true_best = int(np.argmax(utility_true))
        pred_order = np.argsort(-utility_pred)
        pred_best = int(pred_order[0])

        top1.append(float(true_best == pred_best))
        top3.append(float(true_best in set(pred_order[:3].tolist())))

        span = float(np.max(utility_true) - np.min(utility_true))
        if span <= 1e-12:
            regrets.append(0.0)
        else:
            regrets.append(
                max(
                    0.0,
                    float(
                        (utility_true[true_best] - utility_true[pred_best])
                        / span
                    ),
                )
            )

        rho = safe_spearman(utility_true, utility_pred)
        if rho is not None:
            rhos.append(rho)

    scale = robust_scale(y_true)
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
        "top1_accuracy": float(np.mean(top1)),
        "top3_accuracy": float(np.mean(top3)),
        "normalized_regret": float(np.mean(regrets)),
        "spearman": float(np.mean(rhos)) if rhos else np.nan,
        "held_out_datasets": int(len(top1)),
        "spearman_defined_datasets": int(len(rhos)),
    }


def evaluate_target_model(work: pd.DataFrame, target: str, spec, seed: int, threads: int):
    features = model_feature_columns(include_optional_categoricals=True)
    y_col = target_column(target)
    groups = work["dataset_id"].astype(str).to_numpy()
    logo = LeaveOneGroupOut()
    rows = []

    for fold_index, (train_idx, test_idx) in enumerate(logo.split(work, groups=groups)):
        train = work.iloc[train_idx]
        test = work.iloc[test_idx]

        if set(train["dataset_id"].astype(str)) & set(test["dataset_id"].astype(str)):
            raise RuntimeError(f"Dataset leakage detected in fold {fold_index}.")

        estimator = spec.factory(target, int(seed + fold_index))
        with parallel_backend("threading", n_jobs=max(1, int(threads))):
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

    predictions = pd.DataFrame(rows)
    if len(predictions) != 235:
        raise RuntimeError(
            f"{target}/{spec.name} produced {len(predictions)} OOF rows; expected 235."
        )
    if len(predictions[["dataset_id", "framework"]].drop_duplicates()) != 235:
        raise RuntimeError(f"{target}/{spec.name} predictions are not unique.")

    return predictions, evaluate_predictions(predictions, target, spec.name)


def checkpoint_paths(target: str, model: str):
    safe_model = model.lower().replace(" ", "_")
    stem = f"{target}__{safe_model}"
    return (
        CHECKPOINT_DIR / f"{stem}__metrics.json",
        CHECKPOINT_DIR / f"{stem}__predictions.parquet",
    )


def load_checkpoint(target: str, model: str):
    metric_path, pred_path = checkpoint_paths(target, model)
    if not metric_path.exists() or not pred_path.exists():
        return None
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(pred_path)
    if len(predictions) != 235:
        return None
    return metric, predictions


def save_checkpoint(target: str, model: str, metric, predictions: pd.DataFrame):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    metric_path, pred_path = checkpoint_paths(target, model)
    metric_path.write_text(
        json.dumps(metric, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    predictions.to_parquet(
        pred_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )


def select_models(metrics: pd.DataFrame):
    selected = {}
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
        selected[str(target)] = {
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
    if set(selected) != set(PHASE6_TARGETS):
        raise RuntimeError("Model selection is missing one or more primary targets.")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resumable AwareML Phase-6.2 LODO benchmarking."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SNAPSHOT_DIR / "recommender_train_v2.parquet",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-xgboost", action="store_true")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete Phase-6.2 checkpoints and start from zero.",
    )
    args = parser.parse_args()

    # Prevent the harmless but noisy Windows physical-core discovery warning.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, int(args.threads))))

    if args.restart and CHECKPOINT_DIR.exists():
        for path in CHECKPOINT_DIR.glob("*"):
            if path.is_file():
                path.unlink()

    frame = load_recommender_train(args.input.resolve())
    specs = available_model_specs(include_xgboost=(not args.no_xgboost))
    total = len(PHASE6_TARGETS) * len(specs)

    print("=" * 72)
    print("AwareML Phase 6.2 — resumable four-objective benchmark")
    print("=" * 72)
    print("Input:", args.input.resolve())
    print("Rows:", len(frame))
    print("Datasets:", frame["dataset_id"].nunique())
    print("Frameworks:", frame["framework"].nunique())
    print("Evaluation: Leave-One-Dataset-Out")
    print("Models:", [spec.name for spec in specs])
    print("Threads:", max(1, int(args.threads)))
    print("Checkpoint dir:", CHECKPOINT_DIR)
    print()

    metric_rows = []
    prediction_frames = []
    job = 0

    for target in PHASE6_TARGETS:
        for spec in specs:
            job += 1
            cached = load_checkpoint(target, spec.name)
            if cached is not None:
                metric, predictions = cached
                print(
                    f"[{job}/{total}] RESUME target={target} model={spec.name}",
                    flush=True,
                )
            else:
                print(
                    f"[{job}/{total}] RUN    target={target} model={spec.name}",
                    flush=True,
                )
                predictions, metric = evaluate_target_model(
                    frame,
                    target,
                    spec,
                    seed=args.seed,
                    threads=args.threads,
                )
                save_checkpoint(target, spec.name, metric, predictions)
                print(
                    "          DONE regret={:.4f} top1={:.4f} nMAE={:.4f}".format(
                        float(metric["normalized_regret"]),
                        float(metric["top1_accuracy"]),
                        float(metric["normalized_mae"]),
                    ),
                    flush=True,
                )

            metric_rows.append(metric)
            prediction_frames.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    selected = select_models(metrics)

    metrics_path = SNAPSHOT_DIR / "recommender_v2_benchmark_metrics.parquet"
    predictions_path = SNAPSHOT_DIR / "recommender_v2_oof_predictions.parquet"
    selection_path = SNAPSHOT_DIR / "recommender_v2_model_selection.json"

    metrics = metrics.sort_values(
        ["target", "normalized_regret", "top1_accuracy", "spearman", "normalized_mae", "model"],
        ascending=[True, True, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    predictions = predictions.sort_values(
        ["target", "model", "dataset_id", "framework"]
    ).reset_index(drop=True)

    metrics.to_parquet(metrics_path, index=False, engine="pyarrow", compression="zstd")
    predictions.to_parquet(
        predictions_path, index=False, engine="pyarrow", compression="zstd"
    )

    payload = {
        "schema_version": "2.0",
        "phase": "6.2",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_file": args.input.name,
        "input_sha256": sha256_file(args.input.resolve()),
        "evaluation_protocol": "leave_one_dataset_out",
        "checkpoint_resume": True,
        "threads": max(1, int(args.threads)),
        "dataset_count": int(frame["dataset_id"].nunique()),
        "framework_count": int(frame["framework"].nunique()),
        "row_count": int(len(frame)),
        "targets": list(PHASE6_TARGETS),
        "models": sorted(metrics["model"].unique().tolist()),
        "selected_by_target": selected,
        "notes": [
            "All predictions are out-of-fold at the dataset level.",
            "Accuracy is maximized; runtime, energy and CO2 are minimized.",
            "Completed target/model jobs are checkpointed and safely resumed.",
            "Threading backend avoids Windows loky process-spawn overhead.",
        ],
    }
    selection_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    for path in [metrics_path, predictions_path, selection_path]:
        write_checksum(path)

    print()
    print("=" * 72)
    print("AwareML Phase 6.2 benchmark: SUCCESS")
    print("=" * 72)
    for target in PHASE6_TARGETS:
        block = metrics[metrics["target"].eq(target)]
        print()
        print(f"{target.upper()} ({block['direction'].iloc[0]})")
        print(
            block[
                [
                    "model",
                    "normalized_regret",
                    "top1_accuracy",
                    "top3_accuracy",
                    "spearman",
                    "normalized_mae",
                    "mae",
                    "rmse",
                ]
            ].to_string(index=False)
        )
        print("SELECTED ->", selected[target]["model"])

    print()
    for path in [metrics_path, predictions_path, selection_path]:
        print("Wrote:", path)
        print("SHA256:", sha256_file(path))
    print("=" * 72)


if __name__ == "__main__":
    main()
