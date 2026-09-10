from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase15_lab_is_integrated():
    advanced = (
        ROOT / "awareml" / "ui_v2" / "pages_advanced.py"
    ).read_text(encoding="utf-8")

    assert '"Explanation Integrity": phase15_explanation_integrity_page' in advanced
    assert "Explanation Integrity · Phase 15" not in advanced
    assert "phase15_explanation_integrity_page" in advanced


def test_live_probe_is_explicitly_exploratory():
    source = (
        ROOT / "awareml" / "ui_v2" / "phase15_explanation_integrity.py"
    ).read_text(encoding="utf-8")

    assert "Exploratory Live Dataset Probe" in source
    assert "NOT frozen journal evidence" in source
    assert "Correctness · factual agreement" in source
    assert "Faithfulness · response to evidence interventions" in source
