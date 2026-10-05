from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_hotfix_script_parses():
    ast.parse(
        (ROOT / "scripts/apply_main_stack_ci_hotfix_v9_1.py").read_text(encoding="utf-8")
    )


def test_goal_copilot_contract_is_current():
    source = (ROOT / "tests/test_goal_copilot_unified_ux.py").read_text(encoding="utf-8")
    assert 'assert "Run Studio" in UNIFIED' in source
    assert 'assert "Streaming Observatory" in UNIFIED' in source
    assert 'assert "Decision Lab" in UNIFIED' in source


def test_visual_v2_contract_is_current_v8():
    source = (ROOT / "tests/test_phase16_research_visuals_v2.py").read_text(encoding="utf-8")
    assert "Researcher-only views designed for fast interpretation of calibration, reliance, subgroup patterns and timing." in source
    assert 'assert "Trust discrimination by explanation source" in source' in source
    assert 'assert "plotly.graph_objects" in source' in source
