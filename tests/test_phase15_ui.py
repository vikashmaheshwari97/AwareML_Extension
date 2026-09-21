from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_explanation_integrity_lab_is_standalone_and_merged():
    pages = (
        ROOT / "awareml" / "ui_v2" / "pages.py"
    ).read_text(encoding="utf-8")
    wrapper = (
        ROOT / "awareml" / "ui_v2" / "pages_explanation_integrity.py"
    ).read_text(encoding="utf-8")
    advanced = (
        ROOT / "awareml" / "ui_v2" / "pages_advanced.py"
    ).read_text(encoding="utf-8")

    assert '"Explanation Integrity Lab": explanation_integrity_lab_page' in pages
    assert '"Correctness & Faithfulness"' in wrapper
    assert '"Faithfulness Benchmark"' in wrapper
    assert '"Explainability Diagnostics"' in wrapper
    assert "phase15_explanation_integrity_page(show_header=False)" in wrapper
    assert '"Explanation Integrity": phase15_explanation_integrity_page' not in advanced
    assert "Explanation Integrity · Phase 15" not in wrapper

def test_live_probe_is_explicitly_exploratory():
    source = (
        ROOT / "awareml" / "ui_v2" / "phase15_explanation_integrity.py"
    ).read_text(encoding="utf-8")

    assert "Exploratory Live Dataset Probe" in source
    assert "NOT frozen journal evidence" in source
    assert "Correctness · factual agreement" in source
    assert "Faithfulness · response to evidence interventions" in source
