from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.recommender.v2_ranking import (
    rank_candidates,
)


SNAPSHOT_DIR = (
    ROOT
    / "data"
    / "meta"
    / "snapshots"
)


def main() -> None:
    summary_path = (
        SNAPSHOT_DIR
        / "recommender_v2_preference_sensitivity.parquet"
    )
    detail_path = (
        SNAPSHOT_DIR
        / "recommender_v2_preference_sensitivity_detail.parquet"
    )
    if not summary_path.exists():
        raise RuntimeError(
            "Missing preference sensitivity summary."
        )
    if not detail_path.exists():
        raise RuntimeError(
            "Missing preference sensitivity detail."
        )

    summary = pd.read_parquet(
        summary_path
    )
    detail = pd.read_parquet(
        detail_path
    )

    expected = {
        "balanced",
        "accuracy_focus",
        "speed_focus",
        "sustainability_focus",
        "energy_only",
        "co2_only",
    }
    if set(summary["scenario"]) != expected:
        raise RuntimeError(
            "Preference scenario set is incomplete."
        )
    if len(detail) != 47 * len(expected):
        raise RuntimeError(
            "Preference detail must contain one row per dataset/scenario."
        )

    for col in [
        "top1_accuracy",
        "top3_accuracy",
    ]:
        if not summary[col].between(
            0.0,
            1.0,
        ).all():
            raise RuntimeError(
                "{} outside [0,1].".format(col)
            )

    # Deterministic unit smoke test of the ranking function.
    candidates = pd.DataFrame(
        {
            "framework": [
                "A",
                "B",
                "C",
            ],
            "accuracy": [
                0.90,
                0.85,
                0.80,
            ],
            "runtime": [
                3.0,
                2.0,
                1.0,
            ],
            "energy": [
                0.3,
                0.2,
                0.1,
            ],
            "co2": [
                0.3,
                0.2,
                0.1,
            ],
        }
    )
    ranked, meta = rank_candidates(
        candidates,
        weights={
            "accuracy": 1.0,
            "runtime": 0.0,
            "energy": 0.0,
            "co2": 0.0,
        },
    )
    if ranked.iloc[0][
        "framework"
    ] != "A":
        raise RuntimeError(
            "Accuracy-focused ranking smoke test failed."
        )

    print("=" * 72)
    print(
        "AwareML Phase 6.4 validation: PASS"
    )
    print("=" * 72)
    print(
        summary[
            [
                "scenario",
                "top1_accuracy",
                "top3_accuracy",
                "normalized_regret",
            ]
        ].to_string(
            index=False
        )
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
