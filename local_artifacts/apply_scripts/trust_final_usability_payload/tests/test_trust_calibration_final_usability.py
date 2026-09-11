from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "awareml" / "ui_v2" / "phase16_trust_calibration.py"


def _source():
    return UI.read_text(encoding="utf-8")


def test_participant_study_is_self_contained_and_beginner_readable():
    source = _source()
    assert "No prior AwareML run is required." in source
    assert "Reference evidence." in source
    assert "_EVIDENCE_TAG_RE" in source
    assert "How to evaluate an explanation" in source
    assert "Accept — I would rely on it as shown" in source


def test_reference_evidence_does_not_render_condition_labels():
    source = _source()
    assert "_reference_evidence_for_stimulus" in source
    assert 'evidence = row.get("evidence_summary")' in source
    assert "The panel does not tell you whether the explanation is correct." in source


def test_pilot_and_main_analysis_are_session_isolated():
    source = _source()
    assert 'analysis_key = "p16_analysis_result_{}".format(mode)' in source
    assert 'st.session_state.pop("p16_analysis_result", None)' in source
    assert 'key="p16_run_analysis_{}".format(mode)' in source
    assert "No {} analysis is available yet" in source


def test_main_study_zero_state_and_readiness_are_explained():
    source = _source()
    assert "Main Study currently has 0 participants because final collection is still locked." in source
    assert "Pending instrument" in source
    assert "Pending calculation" in source
    assert "Pending determination" in source
    assert "How to start Main Study" in source


def test_windows_commands_use_module_invocation():
    source = _source()
    assert "python -m scripts.freeze_phase16_design" in source
    assert "python -m scripts.analyze_phase16_trust --mode final" in source
    assert "python -m scripts.freeze_phase16_results" in source
