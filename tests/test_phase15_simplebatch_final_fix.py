from pathlib import Path

from awareml.explanation_integrity.faithfulness import (
    OllamaEvidenceExplanationGenerator,
)
from awareml.explanation_integrity.live_guided import (
    LiveGuidedOllamaGenerator,
)

ROOT = Path(__file__).resolve().parents[1]


def test_final_prompt_separation_is_preserved():
    assert (
        OllamaEvidenceExplanationGenerator.prompt_version
        == "phase15_explanation_prompt_v4"
    )
    assert (
        LiveGuidedOllamaGenerator.prompt_version
        == "phase15_live_guided_prompt_v3"
    )


def test_final_live_ui_is_batch_first_v9():
    ui = (
        ROOT / "awareml" / "ui_v2" / "phase15_explanation_integrity.py"
    ).read_text(encoding="utf-8")
    helper = (
        ROOT / "awareml" / "ui_v2" / "phase15_live_batch_ui.py"
    ).read_text(encoding="utf-8")

    assert "Run complete Phase-15 live check" in helper
    assert "Show advanced manual verification" in ui
    assert "p15_live_v9_" in ui
    assert "p15_live_v9_" in helper
