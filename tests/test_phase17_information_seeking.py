from pathlib import Path

import pandas as pd

from awareml.studies.information_seeking import (
    classify_follow_up,
    derive_representative_pattern,
)
from awareml.studies.information_seeking_analysis import (
    analyze_information_seeking,
    session_summaries,
)
from awareml.studies.information_seeking_grounding import (
    detect_requested_topics,
    grounded_information_answer,
)


ROOT = Path(__file__).resolve().parents[1]


def test_follow_up_classifier_categories():
    assert classify_follow_up("Show me the evidence") == "evidence_request"
    assert classify_follow_up("Why is it ranked first?") == "explanation_probe"
    assert classify_follow_up("Are you sure this is correct?") == "challenge"
    assert classify_follow_up("Compare this with OAML") == "counterfactual_or_comparison"
    assert classify_follow_up("What does equalized odds mean?") == "clarification"


def test_classifier_is_not_declared_authoritative():
    protocol = (
        ROOT / "data" / "journal" / "information_seeking_v1" / "design" / "protocol.json"
    ).read_text(encoding="utf-8")
    assert '"role": "analysis_assist_only"' in protocol
    assert '"qualitative_coding_replacement": false' in protocol


def test_representative_patterns():
    assert derive_representative_pattern(0, None, [], "accept") == "Accept without follow-up"
    assert derive_representative_pattern(1, "challenge", [], "reject") == "Challenge then decline"
    assert derive_representative_pattern(2, "counterfactual_or_comparison", [], "accept") == "Compare before deciding"


def test_session_summary_captures_required_fields():
    events = pd.DataFrame(
        [
            {
                "id": 1,
                "session_hash": "abc",
                "event_type": "session_start",
                "payload_json": '{"experience_group":"Novice / student"}',
                "created_at": "2026-01-01 00:00:00",
            },
            {
                "id": 2,
                "session_hash": "abc",
                "event_type": "evidence_view",
                "payload_json": '{"evidence_area":"fairness"}',
                "created_at": "2026-01-01 00:00:02",
            },
            {
                "id": 3,
                "session_hash": "abc",
                "event_type": "chat_turn",
                "payload_json": '{"category":"evidence_request","time_since_session_start_sec":4.5}',
                "created_at": "2026-01-01 00:00:04",
            },
            {
                "id": 4,
                "session_hash": "abc",
                "event_type": "session_end",
                "payload_json": '{"follow_up_occurred":true,"follow_up_count":1,"first_follow_up_category":"evidence_request","time_before_first_follow_up_sec":4.5,"evidence_viewed":["fairness"],"recommendation_decision":"accept","final_confidence":6}',
                "created_at": "2026-01-01 00:00:10",
            },
        ]
    )
    summary = session_summaries(events)
    row = summary.iloc[0]
    assert bool(row["completed"])
    assert bool(row["follow_up_occurred"])
    assert int(row["follow_up_count"]) == 1
    assert row["first_follow_up_category"] == "evidence_request"
    assert float(row["time_before_first_follow_up_sec"]) == 4.5
    assert row["evidence_viewed"] == ["fairness"]
    assert row["recommendation_decision"] == "accept"


def test_analysis_produces_completion_gate_outputs():
    events = pd.DataFrame(
        [
            {
                "id": 1,
                "session_hash": "abc",
                "event_type": "session_start",
                "payload_json": "{}",
                "created_at": "x",
            },
            {
                "id": 2,
                "session_hash": "abc",
                "event_type": "chat_turn",
                "payload_json": '{"category":"clarification","time_since_session_start_sec":2.0}',
                "created_at": "x",
            },
            {
                "id": 3,
                "session_hash": "abc",
                "event_type": "session_end",
                "payload_json": '{"follow_up_occurred":true,"follow_up_count":1,"first_follow_up_category":"clarification","time_before_first_follow_up_sec":2.0,"recommendation_decision":"override"}',
                "created_at": "x",
            },
        ]
    )
    analysis = analyze_information_seeking(events)
    assert analysis["completed_sessions"] == 1
    assert analysis["follow_up_rate"] == 1.0
    assert analysis["first_follow_up_category_counts"]["clarification"] == 1
    assert analysis["representative_pattern_counts"]


def test_enhanced_information_seeking_ui_has_required_logging():
    ui = (
        ROOT / "awareml" / "ui_v2" / "phase17_information_seeking.py"
    ).read_text(encoding="utf-8")
    for marker in [
        '"follow_up_occurred"',
        '"follow_up_count"',
        '"first_follow_up_category"',
        '"time_before_first_follow_up_sec"',
        '"evidence_viewed"',
        '"recommendation_decision"',
        "Manual qualitative themes",
        "Representative pattern",
    ]:
        assert marker in ui


def test_information_seeking_integration_override_present():
    source = (
        ROOT / "awareml" / "ui_v2" / "study_labs_v3.py"
    ).read_text(encoding="utf-8")
    assert "PHASE17_INFORMATION_SEEKING_OVERRIDE_V1" in source
    assert "phase17_information_seeking" in source


def test_multi_topic_grounding_answers_all_requested_topics():
    results = [
        {
            "framework": "OAML",
            "accuracy": 0.6944,
            "f1_macro": 0.6877,
            "runtime_sec": 11.3389,
            "energy_kwh": 0.0001969,
            "co2_kg": 0.00008204,
            "fairness": {
                "dp_diff": 0.07792,
                "equal_opportunity_diff": 0.01812,
                "equalized_odds_gap": 0.03007,
                "predictive_parity_diff": 0.08785,
                "error_rate_gap": 0.03123,
            },
        }
    ]
    ranking = [{"framework": "OAML", "rank": 1, "utility": 0.9042}]
    answer, topics, targets = grounded_information_answer(
        "As OAML is recommended, explain accuracy, fairness, and sustainability.",
        "explanation_probe",
        results,
        ranking,
    )
    assert topics == ["performance", "fairness", "sustainability"]
    assert targets == ["OAML"]
    assert "Accuracy & performance" in answer
    assert "Fairness" in answer
    assert "Sustainability" in answer
    assert "Demographic parity gap" in answer
    assert "[frameworks." not in answer


def test_participant_ui_hides_classifier_label_and_supports_saved_context():
    ui = (
        ROOT / "awareml" / "ui_v2" / "phase17_information_seeking.py"
    ).read_text(encoding="utf-8")
    assert "behavior category:" not in ui
    assert "Optional example follow-up questions" in ui
    assert "save_study_context" in ui
    assert "resolve_study_context" in ui
    assert "Participants do not need to run the dashboard first" in ui
