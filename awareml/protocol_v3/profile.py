from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .common import NUMERIC_FEATURES


def prefix_profile(frame, target, prefix_size=1000):
    if int(prefix_size) < 1:
        raise ValueError("prefix_size must be positive")
    data = frame.iloc[:int(prefix_size)].copy()
    if data.empty or target not in data.columns or data.columns.duplicated().any():
        raise ValueError("A nonempty prefix, unique columns and existing target are required.")
    if data[target].isna().any():
        raise ValueError("Profiling needs labels for the calibration prefix; missing labels are not a class.")
    x = data.drop(columns=[target])
    n = x.shape[1]
    if n == 0:
        raise ValueError("No model input features remain.")
    numeric = x.select_dtypes(include=np.number).shape[1]
    fractions = data[target].value_counts().to_numpy(dtype=float) / len(data)
    classes = len(fractions)
    entropy = -sum(float(p) * math.log(float(p)) for p in fractions)
    result = {"profile_sample_size": len(data), "n_features": n,
              "n_numeric_features": numeric, "n_categorical_features": n - numeric,
              "numeric_feature_fraction": numeric / n, "categorical_feature_fraction": (n - numeric) / n,
              "missing_fraction": float(x.isna().to_numpy().mean()), "n_classes": classes,
              "majority_class_fraction": float(max(fractions)), "minority_class_fraction": float(min(fractions)),
              "class_imbalance_ratio": float(max(fractions) / min(fractions)),
              "class_entropy_normalized": entropy / math.log(classes) if classes > 1 else 0.0}
    assert set(result) == set(NUMERIC_FEATURES)
    return result


def profile_csv(path, target, prefix_size=1000):
    # Infer dtypes on the prefix itself, never on the unobserved suffix.
    frame = pd.read_csv(path, nrows=int(prefix_size))
    if target == "__LAST__":
        target = str(frame.columns[-1])
    return prefix_profile(frame, target, prefix_size)


def ordinal_thresholds(prefix, max_classes=5):
    values = pd.to_numeric(prefix, errors="raise").to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or len(np.unique(values)) < 2:
        raise ValueError("Derived classification requires a finite, varying labeled prefix.")
    quantiles = np.arange(1, max_classes) / float(max_classes)
    thresholds = np.unique(np.quantile(values, quantiles)).tolist()
    if len(np.unique(np.digitize(values, thresholds, right=True))) < 2:
        raise ValueError("Prefix quantiles collapse to one class; resolve this task before collection.")
    return thresholds
