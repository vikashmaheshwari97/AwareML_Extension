from __future__ import annotations

from pathlib import Path

from awareml.engine.pareto import (
    common_composite_fairness_keys,
    fairness_score_from_result,
)
from awareml.llm.grounded_chat import GroundedChat
from awareml.ui_v2.research_evidence import (
    load_phase12_objective_selection_reliability,
)


ROOT = Path(__file__).resolve().parents[1]


def _fair(dp, eo, eod, pp, er):
    return {
        "status": "ok",
        "dp_diff": dp,
        "equal_opportunity_diff": eo,
        "equalized_odds_gap": eod,
        "predictive_parity_diff": pp,
        "error_rate_gap": er,
    }


def test_common_composite_uses_same_criteria_for_every_framework():
    a = _fair(0.10, 0.08, 0.12, None, 0.05)
    b = _fair(0.12, 0.06, 0.11, 0.07, 0.04)
    keys = common_composite_fairness_keys([a, b])
    assert "predictive_parity_diff" not in keys
    assert keys == [
        "dp_diff",
        "equal_opportunity_diff",
        "equalized_odds_gap",
        "error_rate_gap",
    ]
    expected_a = 1.0 - (0.10 + 0.08 + 0.12 + 0.05) / 4.0
    expected_b = 1.0 - (0.12 + 0.06 + 0.11 + 0.04) / 4.0
    assert abs(fairness_score_from_result(a, "composite", keys) - expected_a) < 1e-12
    assert abs(fairness_score_from_result(b, "composite", keys) - expected_b) < 1e-12


def test_phase12_frozen_reliability_is_readable():
    info = load_phase12_objective_selection_reliability(ROOT)
    assert info is not None
    assert int(info["benchmark_cases"]) == 55
    assert abs(float(info["micro_f1"]) - 0.6974789915966386) < 1e-12
    assert abs(float(info["paraphrase_consistency_rate"]) - 0.74) < 1e-12
    assert info["primary_failure_tendency"] == "over_selection"
    assert info["release_status"] == "frozen"


def test_information_seeking_answers_differ_by_question_type():
    facts = {
        "frameworks": {
            "AutoClass": {
                "accuracy": 0.80,
                "f1_macro": 0.77,
                "runtime_sec": 10.0,
                "energy_kwh": 0.001,
                "co2_kg": 0.0004,
                "drift_events": [],
                "fairness": _fair(0.10, 0.08, 0.11, 0.05, 0.06),
                "explainability": {
                    "status": "ok",
                    "fidelity": 0.8,
                    "stability": 0.7,
                    "consistency": 0.75,
                },
            },
            "OAML": {
                "accuracy": 0.75,
                "f1_macro": 0.74,
                "runtime_sec": 8.0,
                "energy_kwh": 0.0008,
                "co2_kg": 0.0003,
                "drift_events": [1000],
                "fairness": _fair(0.08, 0.07, 0.09, 0.04, 0.05),
                "explainability": {
                    "status": "ok",
                    "fidelity": 0.7,
                    "stability": 0.8,
                    "consistency": 0.72,
                },
            },
        },
        "ranking": [
            {"framework": "AutoClass", "rank": 1, "utility": 0.80},
            {"framework": "OAML", "rank": 2, "utility": 0.76},
        ],
    }
    chat = GroundedChat()
    why = chat._deterministic_answer(
        "Why was AutoClass recommended?",
        facts,
        category="explanation_probe",
    )
    compare = chat._deterministic_answer(
        "Compare AutoClass versus OAML",
        facts,
        category="counterfactual_or_comparison",
    )
    fairness = chat._deterministic_answer(
        "Show me fairness evidence for AutoClass",
        facts,
        category="evidence_request",
    )
    assert why != compare
    assert compare != fairness
    assert "ranked #1" in why
    assert "Comparison of AutoClass and OAML" in compare
    assert "DP=" in fairness


def test_ui_hotfix_markers_present():
    specialist = (ROOT / "awareml" / "ui_v2" / "pages_specialist.py").read_text(
        encoding="utf-8"
    )
    copilot = (ROOT / "awareml" / "ui_v2" / "pages_copilot.py").read_text(
        encoding="utf-8"
    )
    study_labs = (ROOT / "awareml" / "ui_v2" / "study_labs_v3.py").read_text(
        encoding="utf-8"
    )
    assert '"Equal opportunity": "equal_opportunity_diff"' in specialist
    assert "All fairness metrics" in specialist
    # V3 delegates the research-grade human-study pages out of the specialist
    # wrapper. Verify both the delegation and the participant-facing controls
    # in their canonical implementation file.
    assert "return trust_calibration_research_page()" in specialist
    assert "return information_seeking_research_page()" in specialist
    assert "Participant mode" in study_labs
    assert "category=category" in study_labs
    assert "Objective-selection reliability · frozen Phase 12" in copilot
    assert "will be evaluated systematically in Phase 12" not in copilot
