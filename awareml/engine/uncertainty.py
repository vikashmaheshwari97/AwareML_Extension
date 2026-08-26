from __future__ import annotations

from typing import Iterable
import numpy as np


def bootstrap_mean_ci(values: Iterable[float], confidence: float = 0.95, n_boot: int = 1000, seed: int = 42):
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": None, "lower": None, "upper": None, "n": 0}
    if len(arr) == 1:
        v = float(arr[0])
        return {"mean": v, "lower": v, "upper": v, "n": 1}
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "mean": float(arr.mean()),
        "lower": float(np.quantile(samples, alpha / 2)),
        "upper": float(np.quantile(samples, 1 - alpha / 2)),
        "n": int(len(arr)),
    }
