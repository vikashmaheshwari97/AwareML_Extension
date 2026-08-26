from __future__ import annotations

from typing import Iterable, Mapping, Tuple

import numpy as np
import pandas as pd

from awareml.recommender.v2_ranking import (
    rank_candidates,
)


def _raw_candidate_table(
    ranked: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "framework",
        "accuracy",
        "runtime",
        "energy",
        "co2",
    ]
    missing = [
        col
        for col in required
        if col not in ranked.columns
    ]
    if missing:
        raise ValueError(
            "Ranking missing objective columns: {}".format(
                missing
            )
        )
    return ranked[required].copy()


def _extreme_pair(
    series: pd.Series,
    maximize: bool,
):
    values = pd.to_numeric(
        series,
        errors="raise",
    ).astype(float)

    lo = float(values.min())
    hi = float(values.max())
    span = max(
        1e-12,
        hi - lo,
    )

    if maximize:
        worse = lo - 0.20 * span
        better = hi + 0.20 * span
    else:
        worse = hi + 0.50 * span
        better = max(
            0.0,
            lo - 0.50 * span,
        )

    return worse, better


def build_objective_counterfactual(
    ranked: pd.DataFrame,
    objective: str,
    weights: Mapping[str, float],
    objective_correlations=None,
):
    if objective not in {
        "accuracy",
        "runtime",
        "energy",
        "co2",
    }:
        raise ValueError(
            "Unknown counterfactual objective: {}".format(
                objective
            )
        )

    base = _raw_candidate_table(
        ranked
    )

    top = str(
        ranked.iloc[0]["framework"]
    )
    challenger = str(
        ranked.iloc[1]["framework"]
    )

    maximize = (
        objective == "accuracy"
    )
    worse, better = _extreme_pair(
        base[objective],
        maximize=maximize,
    )

    modified = base.copy()
    modified.loc[
        modified["framework"].eq(top),
        objective,
    ] = worse
    modified.loc[
        modified["framework"].eq(challenger),
        objective,
    ] = better

    reranked, meta = rank_candidates(
        modified,
        weights=weights,
        mode="point",
        objective_correlations=objective_correlations,
    )

    return reranked, meta, {
        "scenario": (
            "{}_evidence_flip".format(
                objective
            )
        ),
        "changed_objectives": [
            objective
        ],
        "intervention": {
            "degraded_framework": top,
            "improved_framework": challenger,
            "objective": objective,
            "degraded_value": float(worse),
            "improved_value": float(better),
        },
    }


def build_sustainability_counterfactual(
    ranked: pd.DataFrame,
    weights: Mapping[str, float],
    objective_correlations=None,
):
    base = _raw_candidate_table(
        ranked
    )

    top = str(
        ranked.iloc[0]["framework"]
    )
    challenger = str(
        ranked.iloc[1]["framework"]
    )

    modified = base.copy()

    interventions = {}

    for objective in [
        "energy",
        "co2",
    ]:
        worse, better = _extreme_pair(
            base[objective],
            maximize=False,
        )
        modified.loc[
            modified["framework"].eq(top),
            objective,
        ] = worse
        modified.loc[
            modified["framework"].eq(
                challenger
            ),
            objective,
        ] = better
        interventions[objective] = {
            "degraded_value": float(
                worse
            ),
            "improved_value": float(
                better
            ),
        }

    reranked, meta = rank_candidates(
        modified,
        weights=weights,
        mode="point",
        objective_correlations=objective_correlations,
    )

    return reranked, meta, {
        "scenario": (
            "sustainability_joint_evidence_flip"
        ),
        "changed_objectives": [
            "energy",
            "co2",
        ],
        "intervention": {
            "degraded_framework": top,
            "improved_framework": challenger,
            "objectives": interventions,
        },
    }
