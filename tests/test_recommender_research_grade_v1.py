from pathlib import Path

import pandas as pd

from awareml.ui_v2.phase18_validation_evidence import phase18_validation_snapshot
from awareml.ui_v2.recommender_research_grade import _handoff_mapping, _preference_stability

ROOT = Path(__file__).resolve().parents[1]


def test_phase18_packaged_evidence_exact_aggregate():
    snap = phase18_validation_snapshot()
    assert snap["datasets"] == 31
    assert snap["preference_cases"] == 3100
    assert snap["top1_matches"] == 2821
    assert snap["top3_successes"] == 2923
    assert abs(snap["top1"] - 0.91) < 1e-12
    assert abs(snap["top3"] - (2923.0 / 3100.0)) < 1e-12
    assert abs(snap["normalized_regret_mean"] - 0.01726558064516129) < 1e-12
    assert abs(snap["spearman_mean"] - 0.9152484193548388) < 1e-12


def test_phase18_objective_winner_rates():
    snap = phase18_validation_snapshot()
    rows = {row["objective"]: row for row in snap["objective_winners"]}
    assert int(rows["accuracy"]["matches"]) == 29
    assert int(rows["runtime"]["matches"]) == 28
    assert int(rows["energy"]["matches"]) == 28
    assert int(rows["co2"]["matches"]) == 28
    for key in ("accuracy", "runtime", "energy", "co2"):
        assert int(rows[key]["cases"]) == 31


def test_handoff_only_maps_supported_run_studio_fields():
    supported, retained = _handoff_mapping({
        "framework": "AutoClass",
        "window_size": 1000,
        "time_budget_sec": 60.0,
        "framework_parameters": {"some_parameter": 1},
        "drift": {"monitoring_enabled": True, "detector": "ADWIN", "sensitivity": "high"},
        "fairness": {"requested": True, "metric": "composite"},
        "explainability": {"level": "high", "method": "SHAP"},
        "sustainability": {"track_energy": True, "track_co2": True},
    })
    assert supported["run_frameworks"] == ["AutoClass"]
    assert supported["run_window"] == 1000
    assert supported["run_budget"] == 60
    assert supported["run_xai_method"] == "shap"
    assert supported["run_codecarbon"] is True
    text = " ".join(retained).lower()
    assert "drift" in text
    assert "fairness" in text
    assert "hyperparameters" in text


def test_preference_sensitivity_reranks_predictions_without_fitting():
    candidates = pd.DataFrame([
        {"framework":"A","accuracy":0.90,"runtime":30.0,"energy":0.003,"co2":0.0012},
        {"framework":"B","accuracy":0.85,"runtime":8.0,"energy":0.001,"co2":0.0004},
        {"framework":"C","accuracy":0.87,"runtime":12.0,"energy":0.0014,"co2":0.0005},
        {"framework":"D","accuracy":0.82,"runtime":10.0,"energy":0.0012,"co2":0.00045},
        {"framework":"E","accuracy":0.88,"runtime":20.0,"energy":0.002,"co2":0.0008},
    ])
    long, winners = _preference_stability(candidates)
    assert len(long) == 30
    assert int((long["Rank"] == 1).sum()) == 6
    assert int(winners["Profiles won"].sum()) == 6


def test_patch_targets_new_research_grade_language():
    three = (ROOT / "awareml" / "ui_v2" / "copilot_three_path.py").read_text(encoding="utf-8")
    workspace = (ROOT / "awareml" / "ui_v2" / "copilot_workspace_v5.py").read_text(encoding="utf-8")
    unified = (ROOT / "awareml" / "ui_v2" / "copilot_unified.py").read_text(encoding="utf-8")
    core = (ROOT / "awareml" / "ui_v2" / "pages_core.py").read_text(encoding="utf-8")
    assert 'metric("Final held-out datasets", "31")' in three
    assert "Dataset-specific pre-execution recommendation" in three
    assert "render_phase18_heldout_validation" in three
    assert "Cross-dataset preference score" in workspace
    assert "render_priority_review_preview" in unified
    assert "Stage approved settings for Run Studio" in (ROOT / "awareml" / "ui_v2" / "recommender_research_grade.py").read_text(encoding="utf-8")
    assert "render_decision_space_research_layer" in core
    assert "render_run_studio_handoff" in core


def test_ui_upgrade_does_not_add_recommender_fit_or_auto_run():
    source = (ROOT / "awareml" / "ui_v2" / "recommender_research_grade.py").read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert 'st.button("Run benchmark"' not in source
    assert "legacy_pages.run_studio_page(" not in source
    assert "no benchmark has been started" in source.lower()
