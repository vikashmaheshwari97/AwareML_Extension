from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.recommender.v2_data import load_recommender_train
from awareml.recommender.v2_evaluation import (
    PHASE6_TARGETS,
    evaluate_target_model,
    select_models,
)
from awareml.recommender.v2_models import available_model_specs

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
        "{}  {}\n".format(sha256_file(path), path.name),
        encoding="utf-8",
    )


def checkpoint_paths(target: str, model: str):
    safe_model = model.lower().replace(" ", "_")
    stem = "{}__{}".format(target, safe_model)
    return (
        CHECKPOINT_DIR / (stem + "__metrics.json"),
        CHECKPOINT_DIR / (stem + "__predictions.parquet"),
    )


def load_checkpoint(target: str, model: str):
    metric_path, pred_path = checkpoint_paths(target, model)
    if not (metric_path.exists() and pred_path.exists()):
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
        json.dumps(metric, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    predictions.to_parquet(
        pred_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resumable AwareML Phase-6.2 LODO model benchmarking."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SNAPSHOT_DIR / "recommender_train_v2.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=SNAPSHOT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-xgboost", action="store_true")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete Phase-6.2 checkpoints and recompute all target/model jobs.",
    )
    args = parser.parse_args()

    if args.restart and CHECKPOINT_DIR.exists():
        for path in CHECKPOINT_DIR.iterdir():
            if path.is_file():
                path.unlink()

    input_path = args.input.resolve()
    frame = load_recommender_train(input_path)
    specs = available_model_specs(include_xgboost=not args.no_xgboost)
    total = len(PHASE6_TARGETS) * len(specs)

    print("=" * 72)
    print("AwareML Phase 6.2 — resumable four-objective benchmark")
    print("=" * 72)
    print("Rows:", len(frame))
    print("Datasets:", frame["dataset_id"].nunique())
    print("Frameworks:", frame["framework"].nunique())
    print("Evaluation: Leave-One-Dataset-Out")
    print("Models:", [spec.name for spec in specs])
    print("Total target/model jobs:", total)
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
                    "[{}/{}] RESUME target={} model={}".format(
                        job, total, target, spec.name
                    ),
                    flush=True,
                )
            else:
                print(
                    "[{}/{}] RUN    target={} model={}".format(
                        job, total, target, spec.name
                    ),
                    flush=True,
                )
                predictions, metric = evaluate_target_model(
                    frame, target, spec, seed=args.seed
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
    selection = select_models(metrics, require_all_targets=True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "recommender_v2_benchmark_metrics.parquet"
    predictions_path = output_dir / "recommender_v2_oof_predictions.parquet"
    selection_path = output_dir / "recommender_v2_model_selection.json"

    metrics = metrics.sort_values(
        ["target", "normalized_regret", "top1_accuracy", "spearman", "normalized_mae", "model"],
        ascending=[True, True, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    predictions = predictions.sort_values(
        ["target", "model", "dataset_id", "framework"]
    ).reset_index(drop=True)

    metrics.to_parquet(
        metrics_path, index=False, engine="pyarrow", compression="zstd"
    )
    predictions.to_parquet(
        predictions_path, index=False, engine="pyarrow", compression="zstd"
    )

    payload = {
        "schema_version": "2.0",
        "phase": "6.2",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_file": input_path.name,
        "input_sha256": sha256_file(input_path),
        "evaluation_protocol": "leave_one_dataset_out",
        "checkpoint_resume": True,
        "dataset_count": int(frame["dataset_id"].nunique()),
        "framework_count": int(frame["framework"].nunique()),
        "row_count": int(len(frame)),
        "targets": list(PHASE6_TARGETS),
        "models": sorted(metrics["model"].unique().tolist()),
        "selected_by_target": selection,
        "notes": [
            "All predictions are out-of-fold at dataset level.",
            "Accuracy is maximized; runtime, energy and CO2 are minimized.",
            "Runtime, energy and CO2 are learned in log1p space.",
            "Tree/XGBoost estimators use n_jobs=1 for Windows stability.",
            "Completed target/model jobs are checkpointed and safely resumed.",
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
        print("{} ({})".format(target.upper(), block["direction"].iloc[0]))
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
        print("SELECTED -> {}".format(selection[target]["model"]))

    print()
    for path in [metrics_path, predictions_path, selection_path]:
        print("Wrote:", path)
        print("SHA256:", sha256_file(path))
    print("=" * 72)


if __name__ == "__main__":
    main()
