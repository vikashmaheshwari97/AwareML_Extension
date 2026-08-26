from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


EXPECTED_RUN_ROWS = 705
EXPECTED_RECOMMENDER_ROWS = 235
EXPECTED_DATASETS = 47
EXPECTED_FRAMEWORKS = (
    "AutoStreamML",
    "AutoClass",
    "EvoAutoML",
    "OAML",
    "ChaCha",
)

V2_INPUT_FEATURES = [
    "framework",
    "n_samples_dataset",
    "n_features",
    "n_numeric_features",
    "n_categorical_features",
    "numeric_feature_fraction",
    "categorical_feature_fraction",
    "missing_fraction",
    "n_classes",
    "majority_class_fraction",
    "minority_class_fraction",
    "class_imbalance_ratio",
    "class_entropy_normalized",
    "window_size",
    "time_budget_sec",
]

V2_OPTIONAL_CATEGORICAL_FEATURES = [
    "dataset_family",
    "source_type",
    "drift_type",
]

V2_TARGETS = {
    "accuracy": "accuracy_mean",
    "runtime": "runtime_sec_mean",
    "energy": "energy_kwh_mean",
    "co2": "co2_kg_mean",
}

V2_UNCERTAINTY_COLUMNS = {
    "accuracy": "accuracy_std",
    "runtime": "runtime_sec_std",
    "energy": "energy_kwh_std",
    "co2": "co2_kg_std",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def snapshot_dir(root: Optional[Path] = None) -> Path:
    base = root if root is not None else project_root()
    return base / "data" / "meta" / "snapshots"


def canonical_runs_path(root: Optional[Path] = None) -> Path:
    return snapshot_dir(root) / "meta_logs_v2.parquet"


def recommender_train_path(root: Optional[Path] = None) -> Path:
    return snapshot_dir(root) / "recommender_train_v2.parquet"


def load_canonical_runs(path: Optional[Path] = None) -> pd.DataFrame:
    p = path if path is not None else canonical_runs_path()
    if not p.exists():
        raise FileNotFoundError(
            "Canonical Meta-Dataset V2 Parquet is missing: {}".format(p)
        )
    df = pd.read_parquet(p)
    validate_canonical_runs(df)
    return df


def load_recommender_train(path: Optional[Path] = None) -> pd.DataFrame:
    p = path if path is not None else recommender_train_path()
    if not p.exists():
        raise FileNotFoundError(
            "Recommender V2 training snapshot is missing: {}".format(p)
        )
    df = pd.read_parquet(p)
    validate_recommender_train(df)
    return df


def validate_canonical_runs(df: pd.DataFrame) -> None:
    required = {
        "dataset_id",
        "framework",
        "seed",
        "accuracy",
        "runtime_sec",
        "energy_kwh",
        "co2_kg",
        "samples_processed",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Canonical V2 run table is missing columns: {}".format(
                ", ".join(missing)
            )
        )

    if len(df) != EXPECTED_RUN_ROWS:
        raise ValueError(
            "Expected {} canonical run rows, got {}.".format(
                EXPECTED_RUN_ROWS, len(df)
            )
        )

    keys = df[["dataset_id", "framework", "seed"]].drop_duplicates()
    if len(keys) != EXPECTED_RUN_ROWS:
        raise ValueError(
            "Canonical V2 run keys are not unique."
        )

    if int(df["dataset_id"].nunique()) != EXPECTED_DATASETS:
        raise ValueError("Canonical V2 must contain 47 datasets.")

    counts = df["framework"].value_counts().to_dict()
    for framework in EXPECTED_FRAMEWORKS:
        if int(counts.get(framework, 0)) != 141:
            raise ValueError(
                "{} must contain 141 rows, got {}.".format(
                    framework, counts.get(framework, 0)
                )
            )


def validate_recommender_train(df: pd.DataFrame) -> None:
    required = {
        "dataset_id",
        "framework",
        "seed_count",
        "accuracy_mean",
        "accuracy_std",
        "runtime_sec_mean",
        "runtime_sec_std",
        "energy_kwh_mean",
        "energy_kwh_std",
        "co2_kg_mean",
        "co2_kg_std",
        "n_features",
        "n_classes",
        "n_samples_dataset",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Recommender V2 snapshot is missing columns: {}".format(
                ", ".join(missing)
            )
        )

    if len(df) != EXPECTED_RECOMMENDER_ROWS:
        raise ValueError(
            "Expected {} dataset/framework rows, got {}.".format(
                EXPECTED_RECOMMENDER_ROWS, len(df)
            )
        )

    keys = df[["dataset_id", "framework"]].drop_duplicates()
    if len(keys) != EXPECTED_RECOMMENDER_ROWS:
        raise ValueError(
            "Recommender V2 dataset/framework keys are not unique."
        )

    if int(df["dataset_id"].nunique()) != EXPECTED_DATASETS:
        raise ValueError("Recommender V2 must contain 47 datasets.")

    if set(df["seed_count"].astype(int).unique().tolist()) != {3}:
        raise ValueError("Every V2 recommender row must aggregate exactly 3 seeds.")

    counts = df["framework"].value_counts().to_dict()
    for framework in EXPECTED_FRAMEWORKS:
        if int(counts.get(framework, 0)) != 47:
            raise ValueError(
                "{} must contain 47 dataset rows, got {}.".format(
                    framework, counts.get(framework, 0)
                )
            )

    for metric, column in V2_TARGETS.items():
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                "{} target contains missing values.".format(metric)
            )


def training_feature_columns(
    include_optional_categoricals: bool = True,
) -> List[str]:
    cols = list(V2_INPUT_FEATURES)
    if include_optional_categoricals:
        cols.extend(V2_OPTIONAL_CATEGORICAL_FEATURES)
    return cols


def target_columns() -> Dict[str, str]:
    return dict(V2_TARGETS)


def uncertainty_columns() -> Dict[str, str]:
    return dict(V2_UNCERTAINTY_COLUMNS)
