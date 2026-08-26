import pandas as pd

from awareml.faithfulness.attribution import (
    cited_objectives,
    objective_influence,
)
from awareml.faithfulness.counterfactuals import (
    build_objective_counterfactual,
)
from awareml.faithfulness.metrics import (
    citation_validity,
    evidence_fidelity_score,
    set_distance,
)
from awareml.recommender.v2_ranking import (
    rank_candidates,
)


def candidate_table():
    return pd.DataFrame(
        {
            "framework": [
                "A",
                "B",
                "C",
            ],
            "accuracy": [
                0.92,
                0.88,
                0.80,
            ],
            "runtime": [
                3.0,
                2.0,
                1.0,
            ],
            "energy": [
                0.30,
                0.20,
                0.10,
            ],
            "co2": [
                0.30,
                0.20,
                0.10,
            ],
        }
    )


def test_citation_validity():
    valid = [
        "evidence.before.candidates.A.accuracy",
        "evidence.before.recommendation.top_framework",
    ]
    cited = [
        "evidence.before.candidates.A.accuracy",
        "evidence.before.invalid",
    ]
    assert citation_validity(
        cited,
        valid,
    ) == 0.5


def test_cited_objectives():
    keys = [
        "evidence.before.candidates.A.accuracy",
        "evidence.before.candidates.A.energy",
    ]
    assert cited_objectives(
        keys
    ) == [
        "accuracy",
        "energy",
    ]


def test_objective_influence_sums_to_one():
    weights = {
        "accuracy": 0.55,
        "runtime": 0.15,
        "energy": 0.15,
        "co2": 0.15,
    }
    influence = objective_influence(
        candidate_table(),
        weights,
    )
    assert set(
        influence
    ) == {
        "accuracy",
        "runtime",
        "energy",
        "co2",
    }
    assert abs(
        sum(
            influence.values()
        )
        - 1.0
    ) < 1e-12


def test_counterfactual_changes_evidence():
    weights = {
        "accuracy": 0.55,
        "runtime": 0.15,
        "energy": 0.15,
        "co2": 0.15,
    }
    ranked, _ = rank_candidates(
        candidate_table(),
        weights=weights,
    )
    cf, _, meta = (
        build_objective_counterfactual(
            ranked,
            "accuracy",
            weights,
        )
    )
    assert (
        meta[
            "changed_objectives"
        ]
        == ["accuracy"]
    )
    assert len(cf) == 3


def test_evidence_fidelity_score_bounds():
    score = evidence_fidelity_score(
        grounding_validity=1.0,
        decision_alignment=1.0,
        attribution_alignment=0.5,
        counterfactual_sensitivity=0.8,
        irrelevant_invariance=1.0,
    )
    assert 0.0 <= score <= 1.0


def test_set_distance():
    value = set_distance(
        ["a", "b"],
        ["a", "c"],
    )
    assert abs(
        value - (2.0 / 3.0)
    ) < 1e-12
