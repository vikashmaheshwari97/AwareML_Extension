from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import math
import numpy as np
import pandas as pd


@dataclass
class DatasetProfile:
    name: str
    n_samples: int
    n_features: int
    n_classes: int
    target: str
    numeric_features: int
    categorical_features: int
    missing_fraction: float


def profile_dataset(df: pd.DataFrame, target: str, name: str = "dataset") -> DatasetProfile:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' is not present in the dataset.")
    X = df.drop(columns=[target])
    return DatasetProfile(
        name=name,
        n_samples=len(df),
        n_features=X.shape[1],
        n_classes=int(df[target].nunique(dropna=True)),
        target=target,
        numeric_features=len(X.select_dtypes(include=np.number).columns),
        categorical_features=X.shape[1] - len(X.select_dtypes(include=np.number).columns),
        missing_fraction=float(df.isna().mean().mean()),
    )


def load_csv(path_or_buffer) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer)


def make_drift_stream(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """A deterministic synthetic stream with two concept changes and one protected attribute."""
    rng = np.random.default_rng(seed)
    n = max(500, int(n))
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    group = rng.integers(0, 2, size=n)
    t = np.arange(n)

    score = np.zeros(n)
    cut1, cut2 = n // 3, 2 * n // 3
    score[:cut1] = 1.6 * x1[:cut1] + 0.6 * x2[:cut1] - 0.15 * group[:cut1]
    score[cut1:cut2] = -0.5 * x1[cut1:cut2] + 1.8 * x2[cut1:cut2] + 0.4 * x3[cut1:cut2] - 0.15 * group[cut1:cut2]
    score[cut2:] = 0.4 * x1[cut2:] - 0.8 * x2[cut2:] + 1.7 * x3[cut2:] - 0.15 * group[cut2:]
    noise = rng.normal(scale=0.7, size=n)
    y = (score + noise > 0).astype(int)

    return pd.DataFrame({
        "x_signal_1": x1,
        "x_signal_2": x2,
        "x_signal_3": x3,
        "group": group,
        "time_index": t,
        "target": y,
    })


class StreamingEncoder:
    """Online, deterministic conversion of mixed tabular rows into numeric River dictionaries."""

    def __init__(self):
        self.category_maps: dict[str, dict[str, int]] = {}

    def transform_row(self, row: pd.Series | dict) -> dict[str, float]:
        data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        out: dict[str, float] = {}
        for key, value in data.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                out[str(key)] = 0.0
                continue
            if isinstance(value, (bool, np.bool_)):
                out[str(key)] = float(value)
                continue
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                text = str(value)
                mapping = self.category_maps.setdefault(str(key), {})
                if text not in mapping:
                    mapping[text] = len(mapping) + 1
                out[str(key)] = float(mapping[text])
        return out

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([self.transform_row(row) for _, row in df.iterrows()], index=df.index)
