from __future__ import annotations

import copy
import random
from typing import Any, Callable
import numpy as np


def clone_model(model):
    if hasattr(model, "clone"):
        try:
            return model.clone()
        except Exception:
            pass
    return copy.deepcopy(model)


def safe_metric_value(metric) -> float:
    try:
        return float(metric.get())
    except Exception:
        return 0.0


def seeded_choice(items, rng: random.Random):
    if not items:
        raise ValueError("Cannot choose from an empty sequence.")
    return items[rng.randrange(len(items))]


def normalize_importance(values: dict[str, float] | None) -> dict[str, float]:
    if not values:
        return {}
    clean = {}
    for k, v in values.items():
        try:
            f = abs(float(v))
            if np.isfinite(f):
                clean[str(k)] = f
        except Exception:
            continue
    total = sum(clean.values())
    return {k: v / total for k, v in clean.items()} if total > 0 else {}
