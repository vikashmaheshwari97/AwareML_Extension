from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from awareml.engine.pareto_spec import (
    CANONICAL_EPSILON,
    PARETO_SPEC_ID,
    epsilon_pareto_mask as canonical_epsilon_pareto_mask,
)


OBJECTIVES = (
    "accuracy",
    "runtime",
    "energy",
    "co2",
)

OBJECTIVE_DIRECTIONS = {
    "accuracy": "max",
    "runtime": "min",
    "energy": "min",
    "co2": "min",
}

DEFAULT_WEIGHTS = {
    "accuracy": 0.55,
    "runtime": 0.15,
    "energy": 0.15,
    "co2": 0.15,
}


def normalize_weights(
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    raw = dict(DEFAULT_WEIGHTS if weights is None else weights)
    clean = {
        objective: max(0.0, float(raw.get(objective, 0.0)))
        for objective in OBJECTIVES
    }
    total = float(sum(clean.values()))
    if total <= 0:
        raise ValueError("At least one Phase-6 objective weight must be positive.")
    return {key: value / total for key, value in clean.items()}


def _minmax(values: pd.Series, maximize: bool) -> pd.Series:
    """Utility normalization retained from frozen ML Recommender V2."""
    s = pd.to_numeric(values, errors="coerce")
    if s.isna().any():
        raise ValueError("Candidate objective values contain nulls.")
    lo = float(s.min())
    hi = float(s.max())

    if abs(hi - lo) <= 1e-12:
        return pd.Series([0.5] * len(s), index=s.index, dtype=float)

    scaled = (s - lo) / (hi - lo)
    if not maximize:
        scaled = 1.0 - scaled
    return scaled.clip(0.0, 1.0)


def pareto_efficient_mask(frame: pd.DataFrame) -> np.ndarray:
    """Legacy exact-Pareto API; epsilon=0 recovers ordinary nondominance."""
    required = set(OBJECTIVES)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Candidate table missing: {}".format(missing))
    mask = canonical_epsilon_pareto_mask(
        frame[list(OBJECTIVES)],
        directions=OBJECTIVE_DIRECTIONS,
        epsilon=0.0,
    )
    return mask.to_numpy(dtype=bool)


def near_pareto_mask(
    frame: pd.DataFrame,
    epsilon: float = CANONICAL_EPSILON,
) -> np.ndarray:
    """Canonical Phase-13 epsilon-Pareto / near-Pareto mask."""
    required = set(OBJECTIVES)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Candidate table missing: {}".format(missing))
    mask = canonical_epsilon_pareto_mask(
        frame[list(OBJECTIVES)],
        directions=OBJECTIVE_DIRECTIONS,
        epsilon=epsilon,
    )
    return mask.to_numpy(dtype=bool)


def rank_candidates(
    candidates: pd.DataFrame,
    weights: Optional[Mapping[str, float]] = None,
    mode: str = "point",
    objective_correlations: Optional[Mapping[str, Mapping[str, float]]] = None,
    correlation_warning: float = 0.90,
    epsilon: float = CANONICAL_EPSILON,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Rank framework candidates under explicit user preferences.

    Weighted utility preserves the frozen V2 point/conservative ranking rule.
    Phase 13 standardizes the frontier marker to canonical epsilon-Pareto with
    epsilon=0.05 in normalized all-higher-is-better space.
    """
    if mode not in {"point", "conservative"}:
        raise ValueError("mode must be 'point' or 'conservative'.")

    required = {"framework", "accuracy", "runtime", "energy", "co2"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError("Candidate table missing: {}".format(missing))

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
            column for column in conservative.values() if column not in work.columns
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
        work[objective + "_score"] = _minmax(
            work[source_columns[objective]],
            maximize=maximize,
        )

    work["utility"] = 0.0
    for objective in OBJECTIVES:
        work["utility"] += normalized[objective] * work[objective + "_score"]

    pareto_evidence = pd.DataFrame(index=work.index)
    for objective in OBJECTIVES:
        pareto_evidence[objective] = pd.to_numeric(
            work[source_columns[objective]],
            errors="raise",
        )

    work["near_pareto"] = near_pareto_mask(
        pareto_evidence,
        epsilon=epsilon,
    )
    # Backward compatibility: existing plots/tables read this column. From Phase 13
    # onward it intentionally represents the canonical epsilon-Pareto set.
    work["pareto_efficient"] = work["near_pareto"]
    work["pareto_spec_id"] = PARETO_SPEC_ID
    work["pareto_epsilon"] = float(epsilon)

    work = work.sort_values(
        ["utility", "accuracy"],
        ascending=[False, False],
    ).reset_index(drop=True)
    work["rank"] = np.arange(1, len(work) + 1)

    warnings = []
    correlations = objective_correlations or {}
    try:
        energy_co2 = float(
            correlations.get("energy", {}).get("co2", np.nan)
        )
    except Exception:
        energy_co2 = np.nan

    if (
        np.isfinite(energy_co2)
        and abs(energy_co2) >= float(correlation_warning)
        and normalized["energy"] > 0
        and normalized["co2"] > 0
    ):
        warnings.append(
            "Energy and CO2 outcomes are strongly correlated "
            "(Spearman rho={:.3f}). Giving both positive weights can "
            "double-count sustainability efficiency; see the Phase-13 "
            "Energy/CO2 sensitivity report.".format(energy_co2)
        )

    margin = (
        float(work.loc[0, "utility"] - work.loc[1, "utility"])
        if len(work) > 1
        else 1.0
    )

    meta = {
        "weights": normalized,
        "ranking_mode": mode,
        "utility_margin": margin,
        "warnings": warnings,
        "near_pareto_count": int(work["near_pareto"].sum()),
        "pareto_count": int(work["near_pareto"].sum()),
        "pareto_spec_id": PARETO_SPEC_ID,
        "pareto_epsilon": float(epsilon),
    }
    return work, meta
