from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from awareml.engine.pareto_spec import (
    CANONICAL_EPSILON,
    PARETO_SPEC_ID,
    epsilon_pareto_mask as canonical_epsilon_pareto_mask,
    robust_unit,
)
from awareml.types import FrameworkResult, ObjectiveWeights, Recommendation


METRIC_SPECS = {
    "accuracy": "max",
    "runtime_sec": "min",
    "energy_kwh": "min",
    "co2_kg": "min",
    "fairness_score": "max",
    "interpretability_score": "max",
}

FAIRNESS_GAP_KEYS = {
    "composite": None,
    "demographic_parity": "dp_diff",
    "equal_opportunity": "equal_opportunity_diff",
    "equalized_odds": "equalized_odds_gap",
    "predictive_parity": "predictive_parity_diff",
    "error_rate": "error_rate_gap",
}

COMPOSITE_FAIRNESS_GAP_KEYS = (
    "dp_diff",
    "equal_opportunity_diff",
    "equalized_odds_gap",
    "predictive_parity_diff",
    "error_rate_gap",
)

COMPOSITE_FAIRNESS_LABELS = {
    "dp_diff": "demographic parity",
    "equal_opportunity_diff": "equal opportunity",
    "equalized_odds_gap": "equalized odds",
    "predictive_parity_diff": "predictive parity",
    "error_rate_gap": "error-rate parity",
}


def _robust_unit(series: pd.Series, direction: str) -> pd.Series:
    """Backward-compatible wrapper around the canonical Phase-13 transform."""
    return robust_unit(series, direction)


def _finite_number(value) -> bool:
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def common_composite_fairness_keys(
    fairness_records: Sequence[Mapping[str, object]],
) -> list:
    """Return disparity criteria available for every valid framework result.

    A composite fairness score is only cross-framework comparable if every
    framework is scored from the same criterion set. Missing values are not
    silently converted to zero and do not give a framework a smaller divisor.
    """
    valid = [
        dict(record or {})
        for record in fairness_records
        if record and record.get("status") == "ok"
    ]
    if not valid:
        return []
    return [
        key
        for key in COMPOSITE_FAIRNESS_GAP_KEYS
        if all(_finite_number(record.get(key)) for record in valid)
    ]


def fairness_score_from_result(
    fair: dict,
    fairness_metric: str = "composite",
    composite_keys: Optional[Sequence[str]] = None,
):
    """Convert a disparity gap to an all-higher-is-better score.

    Named criteria use ``1 - gap``. For ``composite`` the caller can provide a
    shared ``composite_keys`` set so every framework is compared on exactly the
    same available criteria. When omitted, the legacy per-result behavior is
    retained for backward compatibility with older callers.
    """
    if not fair or fair.get("status") != "ok":
        return None

    key = FAIRNESS_GAP_KEYS.get(fairness_metric)
    if key:
        gap = fair.get(key)
        return None if not _finite_number(gap) else max(0.0, 1.0 - float(gap))

    keys = list(composite_keys) if composite_keys is not None else [
        k for k in COMPOSITE_FAIRNESS_GAP_KEYS if _finite_number(fair.get(k))
    ]
    gaps = [float(fair.get(k)) for k in keys if _finite_number(fair.get(k))]
    if not gaps or len(gaps) != len(keys):
        return None
    return max(0.0, min(1.0, 1.0 - float(np.mean(gaps))))


def results_frame(
    results: Iterable[FrameworkResult],
    fairness_metric: str = "composite",
) -> pd.DataFrame:
    result_list = list(results)
    fairness_records = [dict((r.fairness or {})) for r in result_list]

    common_keys = None
    if fairness_metric == "composite":
        common_keys = common_composite_fairness_keys(fairness_records)

    rows = []
    for r, fair in zip(result_list, fairness_records):
        exp = r.explainability or {}
        available_count = sum(
            1 for key in COMPOSITE_FAIRNESS_GAP_KEYS if _finite_number(fair.get(key))
        )
        if fairness_metric == "composite":
            criteria_used = list(common_keys or [])
        else:
            metric_key = FAIRNESS_GAP_KEYS.get(fairness_metric)
            criteria_used = [metric_key] if metric_key else []

        rows.append(
            {
                "framework": r.framework,
                "accuracy": r.accuracy,
                "runtime_sec": r.runtime_sec,
                "energy_kwh": r.energy_kwh,
                "co2_kg": r.co2_kg,
                "fairness_score": fairness_score_from_result(
                    fair,
                    fairness_metric,
                    composite_keys=common_keys,
                ),
                "fairness_criteria_used": ", ".join(
                    COMPOSITE_FAIRNESS_LABELS.get(key, key) for key in criteria_used
                ),
                "fairness_criteria_count": len(criteria_used),
                "fairness_criteria_total": len(COMPOSITE_FAIRNESS_GAP_KEYS),
                "fairness_metric_coverage": available_count
                / float(len(COMPOSITE_FAIRNESS_GAP_KEYS)),
                "interpretability_score": exp.get("consistency"),
            }
        )
    return pd.DataFrame(rows)


def epsilon_pareto_mask(
    df: pd.DataFrame,
    epsilon: float = CANONICAL_EPSILON,
) -> pd.Series:
    """Canonical epsilon-Pareto mask in normalized all-higher-is-better space."""
    directions = {
        column: direction
        for column, direction in METRIC_SPECS.items()
        if column in df.columns
    }
    return canonical_epsilon_pareto_mask(
        df,
        directions=directions,
        epsilon=epsilon,
    )


def objective_correlations(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df[
        [c for c in METRIC_SPECS if c in df.columns]
    ].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method="spearman")


def rank_results(
    results: Iterable[FrameworkResult],
    weights: ObjectiveWeights,
    epsilon: float = CANONICAL_EPSILON,
    fairness_metric: str = "composite",
):
    df = results_frame(results, fairness_metric=fairness_metric)
    w = weights.as_dict()
    mapping = {
        "accuracy": "accuracy",
        "runtime": "runtime_sec",
        "energy": "energy_kwh",
        "co2": "co2_kg",
        "fairness": "fairness_score",
        "interpretability": "interpretability_score",
    }

    norm = pd.DataFrame(index=df.index)
    available_weight = pd.Series(0.0, index=df.index)
    utility = pd.Series(0.0, index=df.index)
    for weight_key, column in mapping.items():
        if column not in df.columns:
            continue
        values = _robust_unit(df[column], METRIC_SPECS[column])
        norm[column] = values
        valid = values.notna().astype(float)
        utility += values.fillna(0.0) * float(w[weight_key])
        available_weight += valid * float(w[weight_key])

    df["utility"] = utility / available_weight.replace(0, np.nan)
    df["near_pareto"] = epsilon_pareto_mask(df, epsilon=epsilon)
    # Backward-compatibility field for exports/internal consumers only. Journal UI
    # should expose ``near_pareto`` and the canonical epsilon, not this alias.
    df["pareto_efficient"] = df["near_pareto"]
    df["fairness_metric"] = fairness_metric
    df["pareto_spec_id"] = PARETO_SPEC_ID
    df["pareto_epsilon"] = float(epsilon)
    df = df.sort_values(
        ["utility", "accuracy"],
        ascending=[False, False],
    ).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    recs = []
    for _, row in df.iterrows():
        rationale = ["weighted utility={:.3f}".format(row["utility"])]
        if row["near_pareto"]:
            rationale.append(
                "epsilon-Pareto candidate (epsilon={:.2f})".format(float(epsilon))
            )
        if pd.notna(row.get("accuracy")):
            rationale.append("accuracy={:.3f}".format(row["accuracy"]))
        if pd.notna(row.get("fairness_score")):
            if fairness_metric == "composite":
                rationale.append(
                    "composite fairness score={:.3f} (1 - mean common disparity)".format(
                        row["fairness_score"]
                    )
                )
            else:
                rationale.append(
                    "{} fairness score={:.3f}".format(
                        fairness_metric,
                        row["fairness_score"],
                    )
                )
        recs.append(
            Recommendation(
                framework=str(row["framework"]),
                utility=float(row["utility"]),
                rank=int(row["rank"]),
                rationale=rationale,
                near_pareto=bool(row["near_pareto"]),
            )
        )
    return df, recs
