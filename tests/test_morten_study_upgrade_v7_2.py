from pathlib import Path
import json

from awareml.studies.trust_morten_v2 import Phase16V2Store, StimulusBankV2
from awareml.ui_v2.phase16_morten_v2 import _evidence_answer


def test_start_new_main_study_controls_are_present_for_both_studies():
    trust = Path("awareml/ui_v2/phase16_morten_v2.py").read_text(encoding="utf-8")
    info = Path("awareml/ui_v2/phase17_morten_v2.py").read_text(encoding="utf-8")
    assert '"Start new Main Study"' in trust
    assert "_reset_trust_participant_session" in trust
    assert '"Start new Main Study"' in info
    assert "_reset_information_participant_session" in info


def test_participant_ui_removes_frozen_track1_wording():
    trust = Path("awareml/ui_v2/phase16_morten_v2.py").read_text(encoding="utf-8")
    info = Path("awareml/ui_v2/phase17_morten_v2.py").read_text(encoding="utf-8")
    for source in [trust, info]:
        assert "frozen Track-1" not in source
        assert "Frozen scenario priorities" not in source
    assert "Scenario priorities" in trust
    assert "Scenario priorities" in info


def test_generic_trust_evidence_answers_are_rich_and_metric_grounded():
    bank = StimulusBankV2()
    stimulus = bank.pairs[bank.pair_order[0]]["correct"]
    trial = stimulus.participant_dict("TEST", 1)

    ranking = _evidence_answer("What evidence supports the ranking?", trial)
    assert "rank #1" in ranking
    assert "five-framework evidence" in ranking
    assert any(metric in ranking.lower() for metric in ["accuracy", "runtime", "energy", "co₂"])

    metrics = _evidence_answer("Explain the metric evidence.", trial)
    assert "Top reference evidence" in metrics
    assert "Scenario priorities" in metrics

    comparison = _evidence_answer("Compare the frameworks using the reference evidence.", trial)
    assert "Controlled comparison" in comparison
    assert comparison.count("rank #") >= 2


def test_trust_researcher_workspace_v72_has_high_resolution_views_and_human_agency():
    source = Path("awareml/ui_v2/phase16_morten_v2.py").read_text(encoding="utf-8")
    markers = [
        "Calibration map",
        "S-TIAS item profile by correctness",
        "Reliance & corrections",
        "Scenario evaluation counts",
        "Framework × explanation-type observation matrix",
        "Trial-position trajectories",
        "Post-study human-agency and study-experience responses",
        "Expertise rating × participant calibration gap",
    ]
    for marker in markers:
        assert marker in source


def test_information_researcher_workspace_v72_has_navigation_question_and_journey_views():
    source = Path("awareml/ui_v2/phase17_morten_v2.py").read_text(encoding="utf-8")
    markers = [
        "Information-seeking funnel",
        "Participant behavior map",
        "Participant × evidence-area access matrix",
        "Evidence-navigation transition matrix",
        "Follow-up timing by category",
        "Decision time × information-seeking depth",
        "Session × interaction-event composition",
        "Expertise-group behavior profile",
        "Qualitative-theme co-occurrence",
    ]
    for marker in markers:
        assert marker in source


def test_secondary_questionnaire_rows_can_be_retrieved(tmp_path):
    store = Phase16V2Store(tmp_path / "trust.sqlite")
    assert store.secondary_rows("final") == []


def test_information_seeking_protocol_explicitly_documents_repeated_access():
    protocol = json.loads(
        Path("data/journal/information_seeking_v2/frozen/protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["schema_version"] == "information_seeking_v2_1"
    assert "repeat" in protocol["analysis"]["researcher_visualization"].lower()
