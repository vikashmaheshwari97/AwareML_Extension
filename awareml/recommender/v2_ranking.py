from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


OBJECTIVES = (
    "accuracy",
    "runtime",
    "energy",
    "co2",
)

DEFAULT_WEIGHTS = {
    "accuracy": 0.55,
    "runtime": 0.15,
    "energy": 0.15,
    "co2": 0.15,
}


def normalize_weights(
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    raw = dict(
        DEFAULT_WEIGHTS
        if weights is None
        else weights
    )
    clean = {
        objective: max(
            0.0,
            float(raw.get(objective, 0.0)),
        )
        for objective in OBJECTIVES
    }
    total = float(sum(clean.values()))
    if total <= 0:
        raise ValueError(
            "At least one Phase-6 objective weight must be positive."
        )
    return {
        k: v / total
        for k, v in clean.items()
    }


def _minmax(
    values: pd.Series,
    maximize: bool,
) -> pd.Series:
    s = pd.to_numeric(
        values,
        errors="coerce",
    )
    if s.isna().any():
        raise ValueError(
            "Candidate objective values contain nulls."
        )
    lo = float(s.min())
    hi = float(s.max())

    if abs(hi - lo) <= 1e-12:
        return pd.Series(
            [0.5] * len(s),
            index=s.index,
            dtype=float,
        )

    scaled = (s - lo) / (hi - lo)
    if not maximize:
        scaled = 1.0 - scaled
    return scaled.clip(0.0, 1.0)


def pareto_efficient_mask(
    frame: pd.DataFrame,
) -> np.ndarray:
    """Return non-dominated candidates for accuracy↑ and costs↓."""
    values = np.column_stack(
        [
            -pd.to_numeric(
                frame["accuracy"],
                errors="raise",
            ).to_numpy(dtype=float),
            pd.to_numeric(
                frame["runtime"],
                errors="raise",
            ).to_numpy(dtype=float),
            pd.to_numeric(
                frame["energy"],
                errors="raise",
            ).to_numpy(dtype=float),
            pd.to_numeric(
                frame["co2"],
                errors="raise",
            ).to_numpy(dtype=float),
        ]
    )

    n = len(values)
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            no_worse = np.all(values[j] <= values[i])
            strictly_better = np.any(values[j] < values[i])
            if no_worse and strictly_better:
                efficient[i] = False
                break
    return efficient


def rank_candidates(
    candidates: pd.DataFrame,
    weights: Optional[Mapping[str, float]] = None,
    mode: str = "point",
    objective_correlations: Optional[Mapping[str, Mapping[str, float]]] = None,
    correlation_warning: float = 0.90,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Rank five framework candidates under explicit user preferences.

    mode="point" uses point predictions.
    mode="conservative" uses lower accuracy and upper cost interval bounds.
    """
    if mode not in {"point", "conservative"}:
        raise ValueError(
            "mode must be 'point' or 'conservative'."
        )

    required = {
        "framework",
        "accuracy",
        "runtime",
        "energy",
        "co2",
    }
    missing = sorted(
        required - set(candidates.columns)
    )
    if missing:
        raise ValueError(
            "Candidate table missing: {}".format(missing)
        )

    work = candidates.copy()
    normalized = normalize_weights(weights)

    source_columns = {
        "accuracy": "accuracy",
        "runtime": "runtime",
        "energy": "energy",
        "co2": "co2",
    }

    if mode == "conservative":
        conservative = {
            "accuracy": "accuracy_lower",
            "runtime": "runtime_upper",
            "energy": "energy_upper",
            "co2": "co2_upper",
        }
        missing_bounds = [
            col
            for col in conservative.values()
            if col not in work.columns
        ]
        if missing_bounds:
            raise ValueError(
                "Conservative ranking requires uncertainty bounds: {}".format(
                    missing_bounds
                )
            )
        source_columns = conservative

    for objective in OBJECTIVES:
        maximize = objective == "accuracy"
        work[
            objective + "_score"
        ] = _minmax(
            work[source_columns[objective]],
            maximize=maximize,
        )

    work["utility"] = 0.0
    for objective in OBJECTIVES:
        work["utility"] += (
            normalized[objective]
            * work[objective + "_score"]
        )

    work["pareto_efficient"] = pareto_efficient_mask(
        work
    )

    work = work.sort_values(
        ["utility", "accuracy"],
        ascending=[False, False],
    ).reset_index(drop=True)
    work["rank"] = np.arange(
        1,
        len(work) + 1,
    )

    warnings = []
    correlations = objective_correlations or {}
    try:
        energy_co2 = float(
            correlations.get("energy", {}).get(
                "co2",
                np.nan,
            )
        )
    except Exception:
        energy_co2 = np.nan

    if (
        np.isfinite(energy_co2)
        and abs(energy_co2)
        >= float(correlation_warning)
        and normalized["energy"] > 0
        and normalized["co2"] > 0
    ):
        warnings.append(
            "Energy and CO2 outcomes are strongly correlated "
            "(Spearman rho={:.3f}). Giving both large weights can "
            "double-count sustainability efficiency.".format(
                energy_co2
            )
        )

    margin = (
        float(
            work.loc[0, "utility"]
            - work.loc[1, "utility"]
        )
        if len(work) > 1
        else 1.0
    )

    meta = {
        "weights": normalized,
        "ranking_mode": mode,
        "utility_margin": margin,
        "warnings": warnings,
        "pareto_count": int(
            work["pareto_efficient"].sum()
        ),
    }
    return work, meta
