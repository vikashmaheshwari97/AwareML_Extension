from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _class_distribution_features(series: pd.Series) -> Dict[str, float]:
    counts = series.value_counts(dropna=False)
    n = float(len(series))
    fractions = [
        float(count) / n
        for count in counts.tolist()
        if n > 0
    ]
    n_classes = int(series.nunique(dropna=False))

    if not fractions:
        return {
            "n_classes": 0.0,
            "majority_class_fraction": np.nan,
            "minority_class_fraction": np.nan,
            "class_imbalance_ratio": np.nan,
            "class_entropy_normalized": np.nan,
        }

    majority = max(fractions)
    minority = min(fractions)
    imbalance = majority / minority if minority > 0 else np.nan

    entropy = -sum(
        p * np.log(p)
        for p in fractions
        if p > 0
    )
    entropy_normalized = (
        float(entropy / np.log(n_classes))
        if n_classes > 1
        else 0.0
    )

    return {
        "n_classes": float(n_classes),
        "majority_class_fraction": float(majority),
        "minority_class_fraction": float(minority),
        "class_imbalance_ratio": float(imbalance),
        "class_entropy_normalized": entropy_normalized,
    }


def profile_from_dataframe_v2(
    df: pd.DataFrame,
    target: str,
    window_size: int = 1000,
    time_budget_sec: float = 60.0,
    dataset_family: str = "unknown",
    source_type: str = "unknown",
    drift_type: str = "unknown",
) -> Dict[str, Any]:
    """Build the Phase-6 recommender meta-feature profile for a new dataset.

    Only dataset/context descriptors are produced. No framework outcome,
    fairness result, XAI result, runtime, energy or carbon value is used as an
    input feature.
    """
    if target not in df.columns:
        raise ValueError(
            "Target column {!r} is missing.".format(target)
        )
    if len(df) == 0:
        raise ValueError("Cannot profile an empty dataframe.")

    feature_frame = df.drop(columns=[target])
    numeric = feature_frame.select_dtypes(
        include=np.number
    ).columns.tolist()
    categorical = [
        str(c)
        for c in feature_frame.columns
        if c not in set(numeric)
    ]

    n_features = int(feature_frame.shape[1])
    n_numeric = int(len(numeric))
    n_categorical = int(len(categorical))

    out = {
        "dataset_family": str(dataset_family or "unknown"),
        "source_type": str(source_type or "unknown"),
        "drift_type": str(drift_type or "unknown"),
        "n_samples_dataset": int(len(df)),
        "n_features": n_features,
        "n_numeric_features": n_numeric,
        "n_categorical_features": n_categorical,
        "numeric_feature_fraction": (
            float(n_numeric / n_features)
            if n_features
            else 0.0
        ),
        "categorical_feature_fraction": (
            float(n_categorical / n_features)
            if n_features
            else 0.0
        ),
        "missing_fraction": float(
            df.isna().mean().mean()
        ),
        "window_size": int(window_size),
        "time_budget_sec": float(time_budget_sec),
    }
    out.update(
        _class_distribution_features(df[target])
    )
    return out


def candidate_rows_from_profile(
    profile: Dict[str, Any],
    frameworks,
) -> pd.DataFrame:
    rows = []
    for framework in frameworks:
        row = dict(profile)
        row["framework"] = str(framework)
        rows.append(row)
    return pd.DataFrame(rows)
