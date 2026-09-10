from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pilot_researcher_key_is_simple_but_final_requires_private_configuration():
    source = (
        ROOT / "awareml" / "ui_v2" / "phase16_trust_calibration.py"
    ).read_text(encoding="utf-8")

    assert 'LOCAL_PILOT_RESEARCHER_KEY = "phase16-local-test-key"' in source
    assert "not FINAL_DESIGN_MANIFEST.exists()" in source
    assert "_resolve_participant_collection_mode() == \"pilot\"" in source
    assert "st.secrets.get" not in source
    assert "Local Pilot Study access: use `phase16-local-test-key`" in source


def test_stimulus_diversity_is_reviewed_but_not_changed_by_ui_patch():
    source = (
        ROOT / "awareml" / "ui_v2" / "phase16_trust_calibration.py"
    ).read_text(encoding="utf-8")

    assert "Stimulus diversity review before Main Study." in source
    assert "Framework identity and explanation method" in source
    assert "before the design is frozen" in source


def test_global_layout_polish_is_presentation_only():
    source = (
        ROOT / "awareml" / "ui_v2" / "layout_polish.py"
    ).read_text(encoding="utf-8")

    assert "def inject_layout_polish()" in source
    assert 'div[data-testid="stHorizontalBlock"]' in source
    assert ".r9-card" in source
    assert "st.session_state" not in source
    assert "TrustCalibrationStudy" not in source


def test_app_loads_global_layout_polish():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "from awareml.ui_v2.layout_polish import inject_layout_polish" in app
    assert "inject_layout_polish()" in app
