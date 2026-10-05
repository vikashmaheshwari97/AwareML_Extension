from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_visual_module_uses_safe_research_views():
    path = ROOT / "awareml/ui_v2/phase16_research_visuals.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert "Likert response distribution" in source
    assert "Each expertise group forms a line" in source
    assert "Trust by explanation source" in source
    assert ".mark_line(point=True" in source
    assert ".mark_circle" in source


def test_ui_no_longer_uses_problematic_analysis_json_renderer():
    path = ROOT / "awareml/ui_v2/phase16_trust_calibration.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'st.json(_analysis_ui_result(result), expanded=False)' not in source
    assert "research_team_self_assessed" in source


def test_frozen_design_manifest_exists_after_apply():
    assert (
        ROOT
        / "data/journal/trust_calibration_phase16_v1/frozen_design/manifest.json"
    ).exists()
