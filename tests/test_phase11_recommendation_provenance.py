from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_3d_decision_space_has_explicit_pre_run_provenance():
    text = (ROOT / "awareml" / "ui_v2" / "pages_core.py").read_text(
        encoding="utf-8"
    )
    assert "PRE-RUN PREFERENCE RECOMMENDATION" in text
    assert "Pre-run recommended framework" in text
    assert "Pre-run preference utility" in text
    assert '{} predicted accuracy".format(selected)' in text
    assert "Recommended framework and inspected framework are different concepts" in text


def test_decision_lab_has_explicit_post_run_provenance():
    text = (ROOT / "awareml" / "ui_v2" / "pages_specialist.py").read_text(
        encoding="utf-8"
    )
    assert "OBSERVED POST-RUN RECOMMENDATION" in text
    assert "Post-run recommended framework" in text
    assert "outcomes actually measured" in text
