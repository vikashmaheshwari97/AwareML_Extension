from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = (ROOT / "awareml" / "ui_v2" / "copilot_workspace_v5.py").read_text(encoding="utf-8")
PAGES = (ROOT / "awareml" / "ui_v2" / "pages_copilot.py").read_text(encoding="utf-8")
UNIFIED = (ROOT / "awareml" / "ui_v2" / "copilot_unified.py").read_text(encoding="utf-8")
THREE = (ROOT / "awareml" / "ui_v2" / "copilot_three_path.py").read_text(encoding="utf-8")


def test_header_has_three_evidence_levels():
    assert "AwareML Copilot Workspace" in WORKSPACE
    assert "Goal interpretation" in WORKSPACE
    assert "Historical preference prior" in WORKSPACE
    assert "Dataset-aware ML Recommender V2" in WORKSPACE
    assert "Current evidence mode" in WORKSPACE


def test_no_dataset_goal_produces_framework_guidance():
    assert "Framework guidance from the evidence available now" in WORKSPACE
    assert "Historical starting point · no dataset required." in WORKSPACE
    assert "HistoricalPreferenceRecommender" in WORKSPACE
    assert "47-dataset / 705-run" in WORKSPACE


def test_dataset_mode_compares_evidence_paths():
    assert "Compare the available evidence paths" in WORKSPACE
    assert "GLOBAL HISTORICAL PRIOR" in WORKSPACE
    assert "DATASET-AWARE ML RECOMMENDER V2" in WORKSPACE
    assert "Different · expected" in WORKSPACE


def test_pages_use_new_workspace_header():
    assert "render_copilot_workspace_header" in PAGES


def test_goal_copilot_uses_framework_guidance_both_modes():
    assert "render_goal_framework_guidance(state,context_free,None)" in UNIFIED or \
           "render_goal_framework_guidance(state, context_free, None)" in UNIFIED
    assert "render_goal_framework_guidance(state,interpretation,proposal_dict)" in UNIFIED or \
           "render_goal_framework_guidance(state, interpretation, proposal_dict)" in UNIFIED


def test_streamlit_duplicate_default_slider_pattern_removed():
    assert 'int(st.session_state.get("three_hist_accuracy"' not in THREE
    assert 'int(st.session_state.get("three_v2_accuracy"' not in THREE
