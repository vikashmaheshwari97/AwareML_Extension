from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_trust_calibration_has_simple_access_choice():
    source = (
        ROOT / "awareml" / "ui_v2" / "phase16_trust_calibration.py"
    ).read_text(encoding="utf-8")

    assert "Choose your study access" in source
    assert "Enter participant study" in source
    assert "Open researcher workspace" in source
    assert "Participant Study" in source
    assert "Researcher Workspace" in source


def test_embedded_participant_view_hides_dashboard_navigation():
    source = (
        ROOT / "awareml" / "ui_v2" / "phase16_trust_calibration.py"
    ).read_text(encoding="utf-8")

    assert "_hide_dashboard_navigation" in source
    assert 'section[data-testid="stSidebar"]' in source
    assert "Exit study view" in source
    assert "Return to AwareML dashboard" in source


def test_participant_mode_is_not_selected_by_participant():
    source = (
        ROOT / "awareml" / "ui_v2" / "phase16_trust_calibration.py"
    ).read_text(encoding="utf-8")

    assert "_resolve_participant_collection_mode" in source
    assert "Pilot Study" in source
    assert "Main Study" in source
    assert 'st.radio("Collection mode"' not in source
    assert 'st.selectbox("Collection mode"' not in source


def test_participant_only_entrypoint_remains_isolated():
    source = (ROOT / "phase16_participant_app.py").read_text(encoding="utf-8")

    assert "render_phase16_participant_study(isolated=True)" in source
    assert "PAGE_REGISTRY" not in source
    assert "Advanced Labs" not in source
