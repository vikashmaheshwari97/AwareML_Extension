from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from awareml.recommender.v2_evaluation import (
    PHASE6_TARGETS,
    evaluate_predictions,
    select_models,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "meta" / "snapshots"


def test_cost_ranking_uses_minimization():
    frame = pd.DataFrame(
        {
            "dataset_id": ["d1"] * 5,
            "framework": ["a", "b", "c", "d", "e"],
            "y_true": [10.0, 20.0, 30.0, 40.0, 50.0],
            "y_pred": [11.0, 21.0, 29.0, 41.0, 49.0],
        }
    )
    metrics = evaluate_predictions(
        frame,
        target="runtime",
        model_name="test",
    )
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["normalized_regret"] == 0.0


def test_accuracy_ranking_uses_maximization():
    frame = pd.DataFrame(
        {
            "dataset_id": ["d1"] * 5,
            "framework": ["a", "b", "c", "d", "e"],
            "y_true": [0.70, 0.75, 0.80, 0.85, 0.90],
            "y_pred": [0.69, 0.76, 0.79, 0.84, 0.91],
        }
    )
    metrics = evaluate_predictions(
        frame,
        target="accuracy",
        model_name="test",
    )
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["normalized_regret"] == 0.0


def test_selection_prefers_lower_regret():
    rows = []
    for target in PHASE6_TARGETS:
        rows.extend(
            [
                {
                    "target": target,
                    "model": "A",
                    "direction": (
                        "maximize"
                        if target == "accuracy"
                        else "minimize"
                    ),
                    "normalized_regret": 0.20,
                    "top1_accuracy": 0.90,
                    "top3_accuracy": 1.0,
                    "spearman": 0.90,
                    "normalized_mae": 0.10,
                    "mae": 0.1,
                    "rmse": 0.2,
                },
                {
                    "target": target,
                    "model": "B",
                    "direction": (
                        "maximize"
                        if target == "accuracy"
                        else "minimize"
                    ),
                    "normalized_regret": 0.10,
                    "top1_accuracy": 0.70,
                    "top3_accuracy": 0.9,
                    "spearman": 0.70,
                    "normalized_mae": 0.20,
                    "mae": 0.2,
                    "rmse": 0.3,
                },
            ]
        )

    selected = select_models(
        pd.DataFrame(rows)
    )
    assert all(
        selected[target]["model"] == "B"
        for target in PHASE6_TARGETS
    )


@pytest.mark.integration
def test_phase62_outputs_if_present():
    metrics_path = (
        SNAPSHOT_DIR
        / "recommender_v2_benchmark_metrics.parquet"
    )
    predictions_path = (
        SNAPSHOT_DIR
        / "recommender_v2_oof_predictions.parquet"
    )

    if not (
        metrics_path.exists()
        and predictions_path.exists()
    ):
        pytest.skip(
            "Phase-6.2 benchmark artifacts not built yet."
        )

    metrics = pd.read_parquet(
        metrics_path
    )
    predictions = pd.read_parquet(
        predictions_path
    )

    assert set(metrics["target"]) == set(
        PHASE6_TARGETS
    )
    assert predictions["dataset_id"].nunique() == 47
    assert np.isfinite(
        predictions["y_pred"].to_numpy(
            dtype=float
        )
    ).all()
