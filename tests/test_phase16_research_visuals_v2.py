from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_visual_module_contains_redesign_features():
    source = (ROOT / "awareml" / "ui_v2" / "phase16_research_visuals.py").read_text(encoding="utf-8")
    assert "CONSTRUCT_LABELS" in source
    assert "ITEM_LABELS" in source
    assert "Likert response distribution" in source
    assert "Researcher-only views designed for fast interpretation of calibration, reliance, subgroup patterns and timing." in source
    assert "Trust discrimination by explanation source" in source
    assert "plotly.graph_objects" in source
    assert "go.Heatmap" in source

def test_apply_script_is_present():
    source = (ROOT / 'scripts' / 'apply_phase16_researcher_visuals_v2.py').read_text(encoding='utf-8')
    assert 'Phase-16 Researcher Workspace visual redesign v2 applied.' in source
    assert 'It does not change study logic, frozen stimuli, analysis formulas, or collected responses.' in source
