from pathlib import Path

import pandas as pd

from awareml.llm.config_diff import (
    diff_configs,
)
from awareml.llm.evidence import (
    EvidenceBundle,
    build_during_evidence,
)
from awareml.llm.goal_parser import (
    deterministic_goal_parse,
)
from awareml.llm.grounded_copilot import (
    GroundedCopilotChat,
)
from awareml.llm.review import (
    review_proposal,
)
from awareml.llm.schemas import (
    CopilotConfiguration,
    CopilotProposal,
    PrimaryObjectiveWeights,
)


def test_goal_keeps_four_primary_objectives():
    goal = deterministic_goal_parse(
        "High accuracy, low energy, fair and explainable."
    )
    weights = (
        goal
        .primary_weights
        .normalized_dict()
    )
    assert set(weights) == {
        "accuracy",
        "runtime",
        "energy",
        "co2",
    }
    assert abs(
        sum(weights.values()) - 1.0
    ) < 1e-12
    assert goal.fairness_required
    assert (
        goal.explainability_level
        == "high"
    )


def test_review_gate_and_config_diff():
    config = CopilotConfiguration(
        framework="AutoClass",
        algorithm="adaptive_population_search",
        window_size=1000,
        time_budget_sec=60.0,
        primary_weights=PrimaryObjectiveWeights(),
    )
    proposal = CopilotProposal(
        proposal_id="test",
        goal="test",
        interpretation=(
            deterministic_goal_parse(
                "accurate streaming classifier"
            )
        ),
        proposed_config=config,
        ml_recommender_framework="AutoClass",
        rationale="evidence grounded",
    )

    review = review_proposal(
        proposal,
        "approved_with_edits",
        edits={
            "window_size": 750
        },
    )
    assert (
        review.final_config.window_size
        == 750
    )
    assert review.config_diff


def test_during_chat_is_evidence_keyed():
    evidence = build_during_evidence(
        {
            "framework": "AutoClass",
            "accuracy": 0.8,
            "samples": 1000,
            "drift_events": [
                500,
            ],
            "points": [
                {
                    "sample": 1000,
                    "accuracy": 0.8,
                    "f1_macro": 0.79,
                }
            ],
            "fairness": {
                "status": "ok"
            },
            "explainability": {},
        }
    )
    answer = GroundedCopilotChat().answer(
        "How many drift events?",
        evidence,
        use_llm=False,
    )
    assert (
        "evidence.during.drift.count"
        in answer.evidence_keys
    )


def test_evidence_drops_raw_rows():
    evidence = EvidenceBundle(
        "before",
        {
            "safe": 1,
            "raw_rows": [
                {"x": 1}
            ],
        },
    )
    assert "raw_rows" not in (
        str(evidence.facts)
    )


def test_privacy_metadata_is_not_raw_payload():
    evidence = EvidenceBundle(
        "before",
        {"safe": 1},
    )
    payload = evidence.prompt_payload()

    assert (
        payload["privacy"][
            "raw_dataset_rows_included"
        ]
        is False
    )
    assert (
        payload["privacy"][
            "raw_participant_rows_included"
        ]
        is False
    )

    # Safe metadata names can contain "dataset_rows"; they must not be treated
    # as actual raw-row payload fields.
    assert "raw_rows" not in payload["facts"]
    assert "dataset_rows" not in payload["facts"]
