from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


PARETO_SPEC_ID = "epsilon_pareto_v1"
CANONICAL_EPSILON = 0.05
NORMALIZATION_ID = "robust_quantile_05_95_all_higher_v1"


def robust_unit(series: pd.Series, direction: str) -> pd.Series:
    """Normalize one objective to [0, 1], where higher is always better.

    The canonical journal transform clips at the within-candidate 5th and 95th
    percentiles before unit scaling. A cost/minimize objective is then inverted.
    Missing values remain missing. A constant objective receives 0.5.
    """
    if direction not in {"max", "min"}:
        raise ValueError("direction must be 'max' or 'min'.")

    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([np.nan] * len(s), index=s.index, dtype=float)

    lo = float(s.quantile(0.05))
    hi = float(s.quantile(0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(
            [0.5 if pd.notna(v) else np.nan for v in s],
            index=s.index,
            dtype=float,
        )

    unit = ((s.clip(lo, hi) - lo) / (hi - lo)).clip(0.0, 1.0)
    if direction == "min":
        unit = 1.0 - unit
    return unit.astype(float)


def normalize_all_higher(
    frame: pd.DataFrame,
    directions: Mapping[str, str],
) -> pd.DataFrame:
    """Return the available objectives in canonical all-higher-is-better space."""
    out = pd.DataFrame(index=frame.index)
    for column, direction in directions.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            out[column] = robust_unit(values, direction)
    return out


def epsilon_nondominated_mask(
    normalized: pd.DataFrame,
    epsilon: float = CANONICAL_EPSILON,
) -> pd.Series:
    """Return epsilon-nondominated rows from normalized higher-is-better values.

    For candidates i and j with normalized desirability vectors z_i and z_j,
    j epsilon-dominates i iff:

      z_jk >= z_ik - epsilon  for every jointly available objective k, and
      z_jk >  z_ik + epsilon  for at least one jointly available objective k.

    A candidate is epsilon-nondominated when no other candidate satisfies this
    relation. epsilon=0 recovers ordinary Pareto nondominance.
    """
    eps = max(0.0, float(epsilon))
    if normalized.empty or len(normalized.columns) == 0:
        return pd.Series([True] * len(normalized), index=normalized.index, dtype=bool)

    keep = []
    for i in normalized.index:
        dominated = False
        for j in normalized.index:
            if i == j:
                continue
            pair = normalized.loc[[i, j]].dropna(axis=1)
            if len(pair.columns) == 0:
                continue
            a = pair.loc[i].to_numpy(dtype=float)
            b = pair.loc[j].to_numpy(dtype=float)
            if np.all(b >= a - eps) and np.any(b > a + eps):
                dominated = True
                break
        keep.append(not dominated)
    return pd.Series(keep, index=normalized.index, dtype=bool)


def epsilon_pareto_mask(
    frame: pd.DataFrame,
    directions: Mapping[str, str],
    epsilon: float = CANONICAL_EPSILON,
) -> pd.Series:
    """Canonical AwareML epsilon-Pareto mask for raw objective values."""
    normalized = normalize_all_higher(frame, directions)
    return epsilon_nondominated_mask(normalized, epsilon=epsilon)


def specification_dict(epsilon: float = CANONICAL_EPSILON) -> Dict[str, object]:
    eps = max(0.0, float(epsilon))
    return {
        "spec_id": PARETO_SPEC_ID,
        "epsilon": eps,
        "normalization_id": NORMALIZATION_ID,
        "space": "normalized all-higher-is-better objective space",
        "normalization": (
            "Per candidate set and objective: clip to the 5th/95th percentile, "
            "scale to [0,1], and invert minimize objectives so higher is better."
        ),
        "dominance_rule": (
            "j epsilon-dominates i iff j_k >= i_k - epsilon for all jointly "
            "available objectives and j_k > i_k + epsilon for at least one objective."
        ),
        "nondominated_rule": (
            "A candidate is epsilon-Pareto / near-Pareto when no other candidate "
            "epsilon-dominates it."
        ),
        "missing_objectives": (
            "Dominance is evaluated only on objectives jointly available for the "
            "candidate pair; a pair with no common objective is not comparable."
        ),
        "epsilon_zero_relation": "epsilon=0 reduces to ordinary Pareto nondominance.",
    }
