from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIFIED = (ROOT / "awareml" / "ui_v2" / "copilot_unified.py").read_text(encoding="utf-8")
PAGE = (ROOT / "awareml" / "ui_v2" / "pages_copilot.py").read_text(encoding="utf-8")


def test_goal_tab_uses_unified_renderer():
    assert "from .copilot_unified import render_goal_copilot_unified_page" in PAGE
    assert "with goal_tab:\n        render_goal_copilot_unified_page()" in PAGE


def test_active_goal_view_has_no_simple_research_switch():
    assert '["Simple View", "Research View"]' not in UNIFIED
    assert 'segmented_control("Copilot view"' not in UNIFIED


def test_active_goal_view_has_no_phase12_validation_panel():
    assert "Validation & technical transparency" not in UNIFIED
    assert "Frozen Phase-12 validation" not in UNIFIED
    assert "V3.1 development checkpoint" not in UNIFIED


def test_ui_contains_no_emoji_icons():
    for icon in ("ðŸ§ ", "âš–ï¸", "ðŸŽ¯", "âš¡", "ðŸ”‹", "ðŸŒ±", "âœ…", "ðŸ“Š"):
        assert icon not in UNIFIED


def test_flow_cards_are_aligned_and_numbered():
    assert "awareml-flow-card" in UNIFIED
    assert '"01"' in UNIFIED
    assert '"02"' in UNIFIED
    assert '"03"' in UNIFIED
    assert '"04"' in UNIFIED


def test_hcai_is_rendered_as_equal_cards():
    assert "Human-centred AI requirements" in UNIFIED
    assert "awareml-hcai-card" in UNIFIED
    assert "Drift sensitivity" in UNIFIED
    assert "Fairness requirement" in UNIFIED
    assert "Explainability" in UNIFIED


def test_recommendation_sections_are_combined():
    assert "Evidence-backed recommendation brief" in UNIFIED
    assert "Recommendation evidence and execution plan" in UNIFIED
    assert "Why this framework is ranked first" in UNIFIED
    assert "Predicted outcomes under the active priorities" in UNIFIED
    assert "Approved-plan configuration" in UNIFIED


def test_existing_benchmark_is_explained_without_erasing_prediction_provenance():
    assert "Observed benchmark available" in UNIFIED
    assert "Measured post-run evidence is available separately in Decision Lab" in UNIFIED


def test_technical_evidence_is_richer_and_no_helper_nested_expander():
    assert "Technical recommendation evidence" in UNIFIED
    assert "Evidence provenance" in UNIFIED
    assert "Evidence identifiers" in UNIFIED
    assert "Configuration provenance" in UNIFIED
    assert "Raw supported configuration" in UNIFIED
    assert "evidence_chips(" not in UNIFIED


def test_human_correction_is_prominent():
    assert "Human-reviewed priorities are active" in UNIFIED
    assert "Human-corrected priorities are active." in UNIFIED


def test_post_approval_next_steps_are_explicit():
    assert "What happens next" in UNIFIED
    assert "Approval does not automatically start the benchmark" in UNIFIED
    assert "Execute in Run Studio" in UNIFIED
    assert "Streaming Observatory" in UNIFIED
    assert "Decision Lab" in UNIFIED


def test_context_free_layout_is_compact():
    assert "Objective-only mode" in UNIFIED
    assert "Framework prediction pending" in UNIFIED
    assert "Framework prediction pending" in UNIFIED

