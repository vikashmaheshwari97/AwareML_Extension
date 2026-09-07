from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_stage_runner_remains_batch_first():
    helper = (
        ROOT / "awareml" / "ui_v2" / "phase15_live_batch_ui.py"
    ).read_text(encoding="utf-8")

    assert "Run complete Phase-15 live check" in helper
    assert "Results by source" in helper
    assert "You do not need to run the stages again" in helper
