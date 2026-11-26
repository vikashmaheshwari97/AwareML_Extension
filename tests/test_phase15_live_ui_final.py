from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_three_phase15_tabs_are_preserved():
    source = (
        ROOT / "awareml" / "ui_v2" / "phase15_explanation_integrity.py"
    ).read_text(encoding="utf-8")

    assert '"Controlled benchmark"' in source
    assert '"Live Dataset Probe · exploratory"' in source
    assert '"Track 2 stimulus bank"' in source


def test_live_timeout_policy_remains_300_seconds():
    source = (
        ROOT / "awareml" / "ui_v2" / "phase15_explanation_integrity.py"
    ).read_text(encoding="utf-8")
    assert "LIVE_OLLAMA_TIMEOUT_SEC = 300" in source
