from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_live_probe_final_ui_has_no_nested_manual_expander():
    ui = (
        ROOT / "awareml" / "ui_v2" / "phase15_explanation_integrity.py"
    ).read_text(encoding="utf-8")

    assert "Show advanced manual verification" in ui
    assert 'Advanced manual verification · optional' not in ui
    assert "p15_live_v9_" in ui
