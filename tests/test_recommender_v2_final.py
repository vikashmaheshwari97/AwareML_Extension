from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from awareml.recommender.v2_ranking import (
    normalize_weights,
    rank_candidates,
)
from awareml.recommender.v2_uncertainty import (
    empirical_residual_calibration,
    interval_for_prediction,
)


def test_weight_normalization():
    weights = normalize_weights(
        {
            "accuracy": 2,
            "runtime": 1,
            "energy": 1,
            "co2": 0,
        }
    )
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert weights["accuracy"] == 0.5


def test_empirical_uncertainty_interval():
    pred = pd.DataFrame(
        {
            "y_true": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "y_pred": [
                1.0,
                2.2,
                2.8,
                4.3,
            ],
        }
    )
    cal = empirical_residual_calibration(
        pred
    )
    lo, hi = interval_for_prediction(
        0.8,
        "accuracy",
        cal,
        coverage=0.90,
    )
    assert 0.0 <= lo <= 0.8
    assert 0.8 <= hi <= 1.0


def test_preference_ranking():
    candidates = pd.DataFrame(
        {
            "framework": [
                "accuracy_best",
                "balanced",
                "efficient",
            ],
            "accuracy": [
                0.95,
                0.90,
                0.80,
            ],
            "runtime": [
                5.0,
                2.0,
                1.0,
            ],
            "energy": [
                0.5,
                0.2,
                0.1,
            ],
            "co2": [
                0.5,
                0.2,
                0.1,
            ],
        }
    )
    ranked, _ = rank_candidates(
        candidates,
        weights={
            "accuracy": 1.0,
            "runtime": 0.0,
            "energy": 0.0,
            "co2": 0.0,
        },
    )
    assert ranked.iloc[0]["framework"] == "accuracy_best"


@pytest.mark.integration
def test_frozen_service_if_present():
    root = Path(__file__).resolve().parents[1]
    marker = (
        root
        / "data"
        / "meta"
        / "active_recommender_v2.txt"
    )
    if not marker.exists():
        pytest.skip(
            "Phase 6.5 has not been frozen yet."
        )

    from awareml.recommender.v2_service import (
        V2Recommender,
    )

    service = V2Recommender(root=root)
    assert set(service.models) == {
        "accuracy",
        "runtime",
        "energy",
        "co2",
    }
