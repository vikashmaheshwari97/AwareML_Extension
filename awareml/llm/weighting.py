from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from .schemas import PrimaryObjectiveWeights


POLICY_ID = "equal_selected_v1"

DISPLAY_TO_KEY = {
    "Accuracy": "accuracy",
    "Runtime": "runtime",
    "Energy": "energy",
    "CO2": "co2",
}
CANONICAL_ORDER = tuple(DISPLAY_TO_KEY)


class WeightingPolicyError(ValueError):
    pass


def normalize_selected_objectives(selected: Iterable[str]) -> List[str]:
    selected_list = list(selected or [])
    unknown = sorted(set(selected_list) - set(CANONICAL_ORDER))
    if unknown:
        raise WeightingPolicyError(
            "Unknown journal objective(s): {}".format(", ".join(unknown))
        )

    ordered = [objective for objective in CANONICAL_ORDER if objective in selected_list]
    if not ordered:
        raise WeightingPolicyError(
            "At least one selected objective is required before weighting."
        )
    return ordered


def equal_weights_for_selected(
    selected: Iterable[str],
) -> Tuple[PrimaryObjectiveWeights, Dict[str, object]]:
    """Problem B: selected objective subset -> documented recommender weights."""

    ordered = normalize_selected_objectives(selected)
    share = 1.0 / float(len(ordered))

    values = {
        "accuracy": 0.0,
        "runtime": 0.0,
        "energy": 0.0,
        "co2": 0.0,
    }
    for objective in ordered:
        values[DISPLAY_TO_KEY[objective]] = share

    weights = PrimaryObjectiveWeights(**values)
    return weights, {
        "policy_id": POLICY_ID,
        "selected_objectives": ordered,
        "rule": "Equal weight among selected objectives; all unselected objectives receive zero.",
        "weights": weights.normalized_dict(),
    }
