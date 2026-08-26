from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from awareml.recommender.v2_ranking import (
    OBJECTIVES,
    rank_candidates,
)


def cited_objectives(
    evidence_keys: Iterable[str],
) -> List[str]:
    found = []
    for objective in OBJECTIVES:
        token = "." + objective
        if any(
            token in key
            for key in evidence_keys
        ):
            found.append(objective)
    return found


def objective_influence(
    candidates: pd.DataFrame,
    weights: Mapping[str, float],
    objective_correlations=None,
) -> Dict[str, float]:
    """Estimate recommendation evidence influence by objective neutralization.

    This is an external, model-agnostic intervention over the recommender's
    predicted objective table. It is not an internal LLM attribution method.

    For each objective we remove its cross-framework discriminating signal by
    setting it to the median for every candidate, rerank, and measure:
      - loss in the original winner's utility, plus
      - an additional penalty if the winner changes.

    Scores are normalized across the four objectives.
    """
    ranked, _ = rank_candidates(
        candidates,
        weights=weights,
        mode="point",
        objective_correlations=objective_correlations,
    )

    original_top = str(
        ranked.iloc[0]["framework"]
    )
    original_utility = float(
        ranked.iloc[0]["utility"]
    )

    raw = {}

    for objective in OBJECTIVES:
        modified = candidates.copy()
        values = pd.to_numeric(
            modified[objective],
            errors="raise",
        )
        modified[objective] = float(
            values.median()
        )

        reranked, _ = rank_candidates(
            modified,
            weights=weights,
            mode="point",
            objective_correlations=objective_correlations,
        )

        new_top = str(
            reranked.iloc[0]["framework"]
        )
        same_row = reranked[
            reranked["framework"].eq(
                original_top
            )
        ]
        if same_row.empty:
            retained_utility = 0.0
        else:
            retained_utility = float(
                same_row.iloc[0]["utility"]
            )

        utility_loss = abs(
            original_utility
            - retained_utility
        )
        flip_bonus = (
            1.0
            if new_top != original_top
            else 0.0
        )
        raw[objective] = (
            utility_loss
            + flip_bonus
        )

    total = float(sum(raw.values()))
    if total <= 1e-12:
        return {
            objective: 0.25
            for objective in OBJECTIVES
        }

    return {
        objective: float(
            value / total
        )
        for objective, value in raw.items()
    }
