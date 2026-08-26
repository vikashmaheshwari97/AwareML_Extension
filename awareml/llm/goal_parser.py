from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from pydantic import ValidationError

from .client import OllamaClient
from .schemas import (
    GoalInterpretation,
    PrimaryObjectiveWeights,
)


def _runtime_constraint(text: str):
    match = re.search(
        r"(?:under|within|max(?:imum)?|no more than)"
        r"[^0-9]{0,18}"
        r"(\d+(?:\.\d+)?)"
        r"\s*(?:s|sec|secs|second|seconds)\b",
        text.lower(),
    )
    return (
        float(match.group(1))
        if match
        else None
    )


def deterministic_goal_parse(
    text: str,
) -> GoalInterpretation:
    t = (text or "").lower()

    weights = {
        "accuracy": 0.55,
        "runtime": 0.15,
        "energy": 0.15,
        "co2": 0.15,
    }

    cues = {
        "accuracy": [
            "accur",
            "performance",
            "quality",
            "correct",
        ],
        "runtime": [
            "fast",
            "latency",
            "runtime",
            "speed",
            "quick",
        ],
        "energy": [
            "energy",
            "power",
            "efficient",
        ],
        "co2": [
            "co2",
            "carbon",
            "emission",
            "sustainable",
            "sustainability",
        ],
    }

    for objective, words in cues.items():
        hits = sum(
            1
            for word in words
            if word in t
        )
        weights[objective] += (
            0.20 * hits
        )

    drift_hits = any(
        phrase in t
        for phrase in [
            "concept drift",
            "drift",
            "non-stationary",
            "nonstationary",
            "adapt quickly",
            "recover quickly",
        ]
    )
    drift_sensitivity = (
        "high"
        if drift_hits
        else "moderate"
    )

    fairness_required = any(
        word in t
        for word in [
            "fair",
            "fairness",
            "bias",
            "equity",
            "protected group",
        ]
    )

    fairness_metric = None
    if "equal opportunity" in t:
        fairness_metric = "equal_opportunity"
    elif "equalized odds" in t:
        fairness_metric = "equalized_odds"
    elif "demographic parity" in t:
        fairness_metric = "demographic_parity"

    explainability_level = (
        "high"
        if any(
            word in t
            for word in [
                "explain",
                "interpretable",
                "interpretability",
                "transparent",
                "understandable",
            ]
        )
        else "moderate"
    )

    if any(
        phrase in t
        for phrase in [
            "strict energy",
            "very low energy",
            "minimize energy",
            "lowest energy",
        ]
    ):
        energy_constraint = "strict"
    elif "energy" in t or "power" in t:
        energy_constraint = "moderate"
    else:
        energy_constraint = "low"

    primary = PrimaryObjectiveWeights(
        **weights
    )
    normalized = primary.normalized_dict()

    summary_parts = [
        "streaming classification",
        "{} drift sensitivity".format(
            drift_sensitivity
        ),
        "{} explainability".format(
            explainability_level
        ),
    ]
    if fairness_required:
        summary_parts.append(
            "fairness required"
        )
    if energy_constraint != "low":
        summary_parts.append(
            "{} energy constraint".format(
                energy_constraint
            )
        )

    return GoalInterpretation(
        primary_weights=PrimaryObjectiveWeights(
            **normalized
        ),
        drift_sensitivity=drift_sensitivity,
        energy_constraint=energy_constraint,
        fairness_required=fairness_required,
        fairness_metric=fairness_metric,
        explainability_level=explainability_level,
        max_runtime_sec=_runtime_constraint(
            t
        ),
        summary="; ".join(summary_parts)
        + ".",
        uncertainties=[],
    )


class GoalParser:
    """Human-goal interpreter.

    This component is deliberately separate from the empirical ML recommender.
    The LLM interprets intent; the V2 meta-model supplies measured framework
    evidence later in CopilotService.
    """

    def __init__(
        self,
        client: Optional[
            OllamaClient
        ] = None,
    ):
        self.client = (
            client or OllamaClient()
        )

    def parse(
        self,
        text: str,
        use_llm: bool = False,
    ) -> Tuple[
        GoalInterpretation,
        dict,
    ]:
        if not text or not text.strip():
            return (
                GoalInterpretation(),
                {
                    "source": "defaults",
                    "warnings": [
                        "No explicit goal was provided."
                    ],
                },
            )

        fallback = (
            deterministic_goal_parse(text)
        )

        if not use_llm:
            return (
                fallback,
                {
                    "source": "deterministic",
                    "warnings": [],
                },
            )

        schema = (
            GoalInterpretation
            .model_json_schema()
        )
        prompt = """You are the AwareML goal interpreter.

Convert the USER GOAL to exactly one JSON object matching the supplied schema.

Important scientific rules:
1. The empirical ML recommender has exactly four primary objectives:
   accuracy, runtime, energy and CO2.
2. Fairness and explainability are HCAI requirements/constraints, not hidden
   additions to the four-objective utility.
3. Do not select an AutoML framework. Another empirical component does that.
4. Do not invent dataset facts, benchmark results, or performance values.
5. Use streaming_classification as the task.
6. If the goal is ambiguous, preserve moderate/conservative defaults and put
   the ambiguity in uncertainties.

SCHEMA:
{}

USER GOAL:
{}
""".format(
            json.dumps(
                schema,
                ensure_ascii=False,
            ),
            text.strip(),
        )

        try:
            payload, meta = (
                self.client.generate_json(
                    prompt
                )
            )
            interpreted = (
                GoalInterpretation
                .model_validate(payload)
            )

            # Always normalize the four empirical preferences.
            normalized = (
                interpreted
                .primary_weights
                .normalized_dict()
            )
            interpreted.primary_weights = (
                PrimaryObjectiveWeights(
                    **normalized
                )
            )
            meta["warnings"] = []
            return interpreted, meta
        except (
            RuntimeError,
            ValidationError,
            ValueError,
            json.JSONDecodeError,
            Exception,
        ) as exc:
            return (
                fallback,
                {
                    "source": (
                        "deterministic-fallback"
                    ),
                    "warnings": [
                        "LLM goal parse rejected: {}".format(
                            type(exc).__name__
                        )
                    ],
                },
            )
