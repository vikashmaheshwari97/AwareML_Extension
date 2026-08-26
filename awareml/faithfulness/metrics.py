from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, Mapping, Sequence, Tuple


EVIDENCE_KEY_PATTERN = re.compile(
    r"\[(evidence\.[A-Za-z0-9_.-]+)\]"
)


def extract_evidence_keys(
    text: str,
):
    return EVIDENCE_KEY_PATTERN.findall(
        text or ""
    )


def citation_validity(
    cited_keys: Iterable[str],
    valid_keys: Iterable[str],
) -> float:
    cited = list(cited_keys or [])
    valid = set(valid_keys or [])

    if not cited:
        return 0.0

    correct = sum(
        1
        for key in cited
        if key in valid
    )
    return float(
        correct / len(cited)
    )


def text_similarity(
    left: str,
    right: str,
) -> float:
    return float(
        SequenceMatcher(
            None,
            (left or "").strip().lower(),
            (right or "").strip().lower(),
        ).ratio()
    )


def set_distance(
    left: Iterable[str],
    right: Iterable[str],
) -> float:
    a = set(left or [])
    b = set(right or [])

    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    intersection = a & b
    return float(
        1.0
        - len(intersection) / len(union)
    )


def evidence_fidelity_score(
    grounding_validity: float,
    decision_alignment: float,
    attribution_alignment: float,
    counterfactual_sensitivity: float,
    irrelevant_invariance: float,
) -> float:
    """AwareML Evidence Fidelity (AEF).

    This is a project-defined transparent composite, not a metric claimed by
    FaithLM or Faithfulness Serum.

    Components:
      25% citation grounding validity
      20% decision/rationale alignment
      20% evidence-attribution alignment
      25% counterfactual sensitivity
      10% irrelevant-evidence invariance
    """
    values = {
        "grounding": float(
            grounding_validity
        ),
        "decision": float(
            decision_alignment
        ),
        "attribution": float(
            attribution_alignment
        ),
        "counterfactual": float(
            counterfactual_sensitivity
        ),
        "invariance": float(
            irrelevant_invariance
        ),
    }

    for key, value in values.items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                "{} must be in [0,1].".format(
                    key
                )
            )

    return float(
        0.25 * values["grounding"]
        + 0.20 * values["decision"]
        + 0.20 * values["attribution"]
        + 0.25 * values["counterfactual"]
        + 0.10 * values["invariance"]
    )
