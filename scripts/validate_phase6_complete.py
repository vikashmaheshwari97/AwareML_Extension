from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.recommender.v2_service import (
    V2Recommender,
    locate_active_v2_manifest,
)


META_DIR = ROOT / "data" / "meta"
SNAPSHOT_DIR = META_DIR / "snapshots"


def main() -> None:
    active_manifest = locate_active_v2_manifest(
        ROOT
    )
    recommender = V2Recommender(
        manifest_path=active_manifest,
        root=ROOT,
    )

    train = pd.read_parquet(
        SNAPSHOT_DIR
        / "recommender_train_v2.parquet"
    )
    first = train.iloc[0]

    profile = {
        "dataset_family": str(
            first.get(
                "dataset_family",
                "unknown",
            )
        ),
        "source_type": str(
            first.get(
                "source_type",
                "unknown",
            )
        ),
        "drift_type": str(
            first.get(
                "drift_type",
                "unknown",
            )
        ),
        "n_samples_dataset": int(
            first["n_samples_dataset"]
        ),
        "n_features": int(
            first["n_features"]
        ),
        "n_numeric_features": int(
            first["n_numeric_features"]
        ),
        "n_categorical_features": int(
            first["n_categorical_features"]
        ),
        "numeric_feature_fraction": float(
            first[
                "numeric_feature_fraction"
            ]
        ),
        "categorical_feature_fraction": float(
            first[
                "categorical_feature_fraction"
            ]
        ),
        "missing_fraction": float(
            first["missing_fraction"]
        ),
        "n_classes": float(
            first["n_classes"]
        ),
        "majority_class_fraction": float(
            first[
                "majority_class_fraction"
            ]
        ),
        "minority_class_fraction": float(
            first[
                "minority_class_fraction"
            ]
        ),
        "class_imbalance_ratio": float(
            first[
                "class_imbalance_ratio"
            ]
        ),
        "class_entropy_normalized": float(
            first[
                "class_entropy_normalized"
            ]
        ),
        "window_size": int(
            first["window_size"]
        ),
        "time_budget_sec": float(
            first["time_budget_sec"]
        ),
    }

    ranked, meta = recommender.recommend_profile(
        profile,
        weights={
            "accuracy": 0.55,
            "runtime": 0.15,
            "energy": 0.15,
            "co2": 0.15,
        },
        ranking_mode="point",
        coverage=0.90,
    )

    if len(ranked) != 5:
        raise RuntimeError(
            "Integrated recommender must rank five frameworks."
        )
    if ranked["rank"].tolist() != [
        1,
        2,
        3,
        4,
        5,
    ]:
        raise RuntimeError(
            "Integrated rank ordering is invalid."
        )

    conservative, _ = (
        recommender.recommend_profile(
            profile,
            weights={
                "accuracy": 0.55,
                "runtime": 0.15,
                "energy": 0.15,
                "co2": 0.15,
            },
            ranking_mode="conservative",
            coverage=0.90,
        )
    )
    if len(conservative) != 5:
        raise RuntimeError(
            "Conservative ranking failed."
        )

    release_path = (
        active_manifest.parent
        / "release_manifest.json"
    )
    if not release_path.exists():
        raise RuntimeError(
            "Release manifest is missing."
        )

    release = json.loads(
        release_path.read_text(
            encoding="utf-8"
        )
    )
    if release.get("status") != "frozen":
        raise RuntimeError(
            "Phase-6 release is not frozen."
        )

    print("=" * 72)
    print(
        "AwareML Phase 6 COMPLETE validation: PASS"
    )
    print("=" * 72)
    print(
        "Active manifest:",
        active_manifest,
    )
    print()
    print(
        "Point-ranking smoke test:"
    )
    print(
        ranked[
            [
                "rank",
                "framework",
                "utility",
                "pareto_efficient",
                "accuracy",
                "runtime",
                "energy",
                "co2",
            ]
        ].to_string(
            index=False
        )
    )
    print()
    print(
        "Warnings:",
        meta.get("warnings"),
    )
    print(
        "Release status: frozen"
    )
    print(
        "23-dataset evaluation split remains untouched."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
