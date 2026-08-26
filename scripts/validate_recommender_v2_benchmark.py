from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.recommender.v2_evaluation import (
    PHASE6_TARGETS,
)


SNAPSHOT_DIR = ROOT / "data" / "meta" / "snapshots"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(
            lambda: fh.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def verify_checksum(path: Path) -> None:
    checksum = Path(
        str(path) + ".sha256"
    )
    if not checksum.exists():
        raise RuntimeError(
            "Missing checksum: {}".format(
                checksum
            )
        )

    expected = (
        checksum
        .read_text(encoding="utf-8")
        .strip()
        .split()[0]
    )
    actual = sha256_file(path)
    if expected != actual:
        raise RuntimeError(
            "Checksum mismatch for {}.".format(
                path.name
            )
        )


def main() -> None:
    metrics_path = (
        SNAPSHOT_DIR
        / "recommender_v2_benchmark_metrics.parquet"
    )
    predictions_path = (
        SNAPSHOT_DIR
        / "recommender_v2_oof_predictions.parquet"
    )
    selection_path = (
        SNAPSHOT_DIR
        / "recommender_v2_model_selection.json"
    )

    for path in [
        metrics_path,
        predictions_path,
        selection_path,
    ]:
        if not path.exists():
            raise RuntimeError(
                "Missing Phase-6.2 artifact: {}".format(
                    path
                )
            )
        verify_checksum(path)

    metrics = pd.read_parquet(
        metrics_path
    )
    predictions = pd.read_parquet(
        predictions_path
    )
    selection = json.loads(
        selection_path.read_text(
            encoding="utf-8"
        )
    )

    if set(metrics["target"]) != set(
        PHASE6_TARGETS
    ):
        raise RuntimeError(
            "Metrics do not contain exactly "
            "the four Phase-6 targets."
        )

    model_sets = {
        target: set(
            metrics.loc[
                metrics["target"].eq(target),
                "model",
            ].astype(str)
        )
        for target in PHASE6_TARGETS
    }

    reference = None
    for target, models in model_sets.items():
        if reference is None:
            reference = models
        elif models != reference:
            raise RuntimeError(
                "Model set differs across targets."
            )

    if reference is None or len(reference) < 6:
        raise RuntimeError(
            "Expected at least six benchmark models."
        )

    selected = selection.get(
        "selected_by_target"
    ) or {}
    if set(selected) != set(
        PHASE6_TARGETS
    ):
        raise RuntimeError(
            "Selection file is missing target selections."
        )

    for target in PHASE6_TARGETS:
        model = str(
            selected[target]["model"]
        )
        if model not in model_sets[target]:
            raise RuntimeError(
                "Selected model {} is absent from {} metrics.".format(
                    model,
                    target,
                )
            )

    required_metric_cols = [
        "mae",
        "rmse",
        "normalized_mae",
        "normalized_rmse",
        "top1_accuracy",
        "top3_accuracy",
        "normalized_regret",
        "held_out_datasets",
    ]
    for col in required_metric_cols:
        values = pd.to_numeric(
            metrics[col],
            errors="coerce",
        )
        if values.isna().any():
            raise RuntimeError(
                "{} contains nulls.".format(
                    col
                )
            )

    if not (
        metrics["held_out_datasets"]
        .astype(int)
        .eq(47)
        .all()
    ):
        raise RuntimeError(
            "Every benchmark row must evaluate 47 held-out datasets."
        )

    if not (
        metrics["top1_accuracy"]
        .between(0.0, 1.0)
        .all()
    ):
        raise RuntimeError(
            "top1_accuracy outside [0,1]."
        )

    if not (
        metrics["top3_accuracy"]
        .between(0.0, 1.0)
        .all()
    ):
        raise RuntimeError(
            "top3_accuracy outside [0,1]."
        )

    if not (
        metrics["normalized_regret"] >= 0
    ).all():
        raise RuntimeError(
            "normalized_regret must be non-negative."
        )

    for target in PHASE6_TARGETS:
        for model in sorted(
            model_sets[target]
        ):
            block = predictions[
                predictions["target"].eq(target)
                & predictions["model"].eq(model)
            ]

            if len(block) != 235:
                raise RuntimeError(
                    "{}/{} has {} predictions; expected 235.".format(
                        target,
                        model,
                        len(block),
                    )
                )

            keys = block[
                ["dataset_id", "framework"]
            ].drop_duplicates()
            if len(keys) != 235:
                raise RuntimeError(
                    "{}/{} predictions are not unique.".format(
                        target,
                        model,
                    )
                )

            if block[
                "dataset_id"
            ].nunique() != 47:
                raise RuntimeError(
                    "{}/{} does not cover 47 datasets.".format(
                        target,
                        model,
                    )
                )

            if not np.isfinite(
                block["y_true"].to_numpy(
                    dtype=float
                )
            ).all():
                raise RuntimeError(
                    "Non-finite y_true values."
                )

            if not np.isfinite(
                block["y_pred"].to_numpy(
                    dtype=float
                )
            ).all():
                raise RuntimeError(
                    "Non-finite y_pred values."
                )

    print("=" * 72)
    print(
        "AwareML Phase 6.2 validation: PASS"
    )
    print("=" * 72)
    print(
        "Models:",
        sorted(reference),
    )
    print(
        "Targets:",
        list(PHASE6_TARGETS),
    )
    print(
        "Metric rows:",
        len(metrics),
    )
    print(
        "OOF prediction rows:",
        len(predictions),
    )
    print(
        "Held-out datasets per model/target:",
        47,
    )
    print()
    print("Selected models:")
    for target in PHASE6_TARGETS:
        row = selected[target]
        print(
            "  {:8s} -> {:22s} "
            "regret={:.4f} top1={:.4f} "
            "nMAE={:.4f}".format(
                target,
                row["model"],
                float(
                    row["normalized_regret"]
                ),
                float(
                    row["top1_accuracy"]
                ),
                float(
                    row["normalized_mae"]
                ),
            )
        )
    print()
    print("All checksums: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
