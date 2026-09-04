from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = (ROOT / "awareml/ui_v2/pre14_usability.py").read_text(encoding="utf-8")
PAGE = (ROOT / "awareml/ui_v2/pages_copilot.py").read_text(encoding="utf-8")


def test_research_view_does_not_nest_expanders():
    assert 'with st.expander("Research evidence & diagnostics"' not in HELPER
    assert "research_renderer(interpretation, parse_meta, state)" in HELPER


def test_beginner_plan_has_clear_provenance_and_rationale():
    assert "Why this framework fits your priorities" in HELPER
    assert "The LLM did not choose" in HELPER
    assert "ML Recommender V2 compared the five candidate frameworks" in HELPER


def test_fairness_is_not_mislabeled_disabled():
    assert '"Setting": "Fairness constraint"' in HELPER
    assert "Not requested by this goal" in HELPER
    assert "does not disable post-run fairness analysis" in HELPER


def test_simple_view_has_persistent_final_plan_review():
    assert "Save final plan decision" in HELPER
    assert 'ReviewStore(ROOT / "artifacts" / "copilot" / "reviews.jsonl")' in HELPER


def test_simple_view_hides_advanced_research_detail():
    assert 'if state.get("copilot_view_mode") == "Simple View":\n        return' in PAGE
