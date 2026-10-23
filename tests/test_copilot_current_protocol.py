"""Synthetic bundles below are test fixtures, never experimental evidence."""
import copy
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from awareml.protocol_v3.common import NUMERIC_FEATURES, OBJECTIVES, PRIMARY, TARGETS, digest, sha256
from awareml.protocol_v3.models import FrameworkMean, feature_columns
from awareml.ui_v2 import recommender_bridge as bridge


def make_bundle(path, frameworks=PRIMARY, cohort="dev"):
    path.mkdir()
    rows = []
    for dataset in range(3):
        for index, fw in enumerate(frameworks):
            row = {"dataset_id": "d%d" % dataset, "framework": fw, "cohort": cohort,
                   "seed_count": 5, **{key: 1 for key in NUMERIC_FEATURES}}
            row.update(accuracy_mean=.6 + .05 * index, runtime_sec_mean=2 + index,
                       energy_kwh_mean=.001 * (2 + index), co2_kg_mean=.0004 * (2 + index))
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(path / "development_aggregates.csv", index=False)
    frozen = {"mode": "production", "hardware_issues": []}
    collection = {"collection_sha256": digest(frozen), "frozen": frozen}
    (path / "training_collection.json").write_text(json.dumps(collection), encoding="utf-8")
    models = {}
    for target in OBJECTIVES:
        model = FrameworkMean().fit(frame, frame[TARGETS[target] + "_mean"])
        name = target + ".joblib"
        joblib.dump(model, path / name)
        models[target] = {"file": name, "sha256": sha256(path / name), "family": "FrameworkMean"}
    manifest = {"schema": "awareml_meta_recommender_v3", "portfolio": "primary" if len(frameworks) == 4 else "extended",
                "feature_set": "prefix", "features": feature_columns(False), "profile_size": 1000,
                "frameworks": list(frameworks), "training_dataset_ids": ["d0", "d1", "d2"],
                "created_utc": "test_fixture_only", "collection_sha256": collection["collection_sha256"],
                "protocol": {"profile_size": 1000, "seeds": [42, 43, 44, 45, 46],
                             "scientific_time_limit_sec": None, "max_samples": 100000,
                             "safety_timeout_sec": 7200, "protocol_id": "test_fixture_only"},
                "models": models, "uncertainty": {k: {"absolute_radius": .01} for k in OBJECTIVES},
                "environment": {"packages": {"scikit-learn": importlib.metadata.version("scikit-learn")}},
                "file_hashes": {p.name: sha256(p) for p in path.iterdir()}}
    save_manifest(path, manifest)
    return path


def save_manifest(path, manifest):
    (path / "manifest.json").write_text(json.dumps({"bundle_sha256": digest(manifest), "frozen": manifest}), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    path = make_bundle(tmp_path / "primary_prefix")
    monkeypatch.setenv("AWAREML_RECOMMENDER_V3_BUNDLE", str(path))
    return path


def frame():
    return pd.DataFrame({"x": np.arange(1200), "target": np.arange(1200) % 2})


def test_pending_has_no_invented_training_or_external_counts(tmp_path):
    status = bridge.bundle_status(tmp_path / "missing")
    assert not status["ready"] and status["state"] == "training_pending"
    assert "underlying_runs" not in status and "development_datasets" not in status


def test_counts_come_from_active_portfolio_and_real_seed_aggregates(bundle):
    status = bridge.bundle_status()
    assert status["ready"]
    assert (status["development_datasets"], status["training_rows"], status["underlying_runs"]) == (3, 12, 60)
    assert status["framework_count"] == 4
    assert bridge.evaluation_evidence("external") is None


def test_extended_counts_and_chacha_label_are_explicit(tmp_path):
    path = make_bundle(tmp_path / "extended", tuple(PRIMARY) + ("ChaCha",))
    ranked, status = bridge.historical_prior({"accuracy": 1}, path)
    assert status["training_rows"] == 15 and status["underlying_runs"] == 75
    assert "ChaCha-OVR" in ranked.iloc[0].framework_label


def test_historical_prior_uses_development_aggregates_and_active_weights(bundle):
    accuracy, _ = bridge.historical_prior({"accuracy": 1})
    runtime, _ = bridge.historical_prior({"runtime": 1})
    assert accuracy.iloc[0].framework == PRIMARY[-1]
    assert runtime.iloc[0].framework == PRIMARY[0]
    assert set(accuracy.support_datasets) == {3}


def test_external_outcomes_cannot_become_historical_prior(tmp_path):
    status = bridge.bundle_status(make_bundle(tmp_path / "bad", cohort="eval"))
    assert not status["ready"] and "development" in status["message"]


def test_bundle_tampering_blocks_all_inference(bundle):
    with (bundle / "development_aggregates.csv").open("a", encoding="utf-8") as handle:
        handle.write("tamper")
    assert bridge.bundle_status()["state"] == "invalid_bundle"
    with pytest.raises(RuntimeError, match="Artifact changed"):
        bridge.load_current_recommender()


@pytest.mark.parametrize("key,value", [
    ("feature_set", "prefix_plus_metadata"),
    ("features", ["framework", "n_samples_dataset"]),
    ("profile_size", 10000),
])
def test_non_deployable_schema_is_rejected(bundle, key, value):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))["frozen"]
    manifest[key] = value
    save_manifest(bundle, manifest)
    assert not bridge.bundle_status()["ready"]


def test_missing_model_has_no_legacy_fallback(bundle):
    (bundle / "accuracy.joblib").unlink()
    assert not bridge.bundle_status()["ready"]
    with pytest.raises(RuntimeError):
        bridge.load_current_recommender()


def test_prefix_inference_ignores_unseen_tail(bundle):
    original = frame()
    changed = original.copy()
    changed.loc[1000:, "x"] = 999999
    changed.loc[1000:, "target"] = "unseen_tail_class"
    service = bridge.load_current_recommender()
    first, meta = service.recommend_dataframe(original, "target", {"accuracy": 1})
    second, other = service.recommend_dataframe(changed, "target", {"accuracy": 1})
    pd.testing.assert_frame_equal(first, second)
    assert meta["profile"] == other["profile"]
    assert meta["profile"]["profile_sample_size"] == 1000
    assert not {"n_samples_dataset", "drift_type", "source_type", "dataset_family"}.intersection(meta["profile"])


def test_stale_v2_proposal_is_cleared_but_goal_interpretation_is_retained(tmp_path):
    state = {"copilot_proposal": {"interpretation": {"selected_objectives": ["Accuracy"]}},
             "copilot_ranked": "old result", "copilot_review": "old approval"}
    bridge.sync_copilot_evidence(state, bridge.bundle_status(tmp_path / "absent"))
    assert "copilot_proposal" not in state and "copilot_review" not in state
    assert "copilot_ranked" not in state
    assert state["copilot_context_free_interpretation"]["selected_objectives"] == ["Accuracy"]


def test_dataset_prefix_changes_invalidate_proposal_but_tail_does_not(bundle):
    state = {"dataset": frame(), "target": "target"}
    bridge.sync_copilot_evidence(state)
    state["copilot_proposal"] = {"interpretation": {"selected_objectives": ["Accuracy"]}}
    state["dataset"].loc[1100, "x"] = -999
    bridge.sync_copilot_evidence(state)
    assert "copilot_proposal" in state
    state["dataset"].loc[0, "x"] = -999
    bridge.sync_copilot_evidence(state)
    assert "copilot_proposal" not in state


def test_copilot_proposal_and_human_correction_use_prefix_models(bundle, monkeypatch):
    from awareml.llm.copilot import CopilotService
    from awareml.llm.schemas import GoalInterpretation
    from awareml.recommender.v2_service import V2Recommender
    from awareml.recommender import v2_profile
    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy V2 inference must not be called")
    monkeypatch.setattr(V2Recommender, "recommend_profile", forbidden)
    monkeypatch.setattr(v2_profile, "profile_from_dataframe_v2", forbidden)
    class Parser:
        def parse(self, *args, **kwargs):
            return GoalInterpretation(), {}
    service = CopilotService(recommender=bridge.load_current_recommender(), goal_parser=Parser())
    state = {"dataset": frame(), "target": "target"}
    proposal, ranked, evidence, meta = bridge.propose_current(service, state, "accurate", False)
    state.update(copilot_proposal=proposal, copilot_ranked=ranked, copilot_meta=meta)
    assert proposal.ml_recommender_framework == PRIMARY[-1]
    winner = bridge.apply_current_objective_override(state, ["Runtime"])
    assert winner == PRIMARY[0]
    assert state["copilot_proposal"].interpretation.primary_weights.runtime == 1
    assert state["copilot_review"] is None
    assert state["copilot_evidence"].facts["dataset_profile"]["profile_sample_size"] == 1000


def test_mismatched_external_report_is_not_presented_as_completed(bundle):
    path = bundle.parent / (bundle.name + "_external_evaluation")
    path.mkdir()
    (path / "evaluation.json").write_text(json.dumps({
        "bundle_manifest_sha256": "another model", "portfolio": "primary"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        bridge.evaluation_evidence("external")


def test_current_sidebar_route_has_no_separate_version_page():
    root = Path(__file__).resolve().parents[1]
    pages = (root / "awareml/ui_v2/pages.py").read_text(encoding="utf-8")
    assert '"Meta-Recommender V3":' not in pages
    assert '"Copilot Workspace"' in pages
    active = (root / "awareml/ui_v2/copilot_unified.py").read_text(encoding="utf-8")
    assert "propose_current(service,state,goal,use_llm)" in active
    assert "load_v2_recommender" not in active


def test_streamlit_pending_copilot_and_dataset_tabs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("AWAREML_RECOMMENDER_V3_BUNDLE", str(tmp_path / "missing"))
    app = AppTest.from_string("from awareml.ui_v2.pages_copilot import copilot_workspace_page\ncopilot_workspace_page()")
    app.run(timeout=30)
    assert not app.exception
    app.button(key="copilot_workspace_open_dataset").click().run()
    assert not app.exception
    assert any("pending" in item.value.lower() for item in app.info)
    assert not any(item.value == "705" for item in app.metric)
    app.button(key="copilot_workspace_open_historical").click().run()
    assert not app.exception
    assert not any(item.value == "705" for item in app.metric)


def test_streamlit_trained_dataset_panel_reports_real_counts(bundle):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string("from awareml.ui_v2.copilot_recommender import render_dataset_aware_tab\nrender_dataset_aware_tab()")
    app.session_state["awareml_state"] = {"dataset": frame(), "target": "target", "dataset_name": "test fixture"}
    app.run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == "3" and app.metric[1].value == "12"
    assert app.metric[2].value == "60" and app.metric[3].value == "5"
    assert app.dataframe[0].value.shape[0] == 4


def test_goal_interpretation_remains_usable_while_training_is_pending(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("AWAREML_RECOMMENDER_V3_BUNDLE", str(tmp_path / "absent"))
    app = AppTest.from_string("from awareml.ui_v2.pages_copilot import copilot_workspace_page\ncopilot_workspace_page()")
    app.session_state["awareml_state"] = {"dataset": frame(), "target": "target"}
    app.run(timeout=30)
    app.text_area(key="goal_v3_text").set_value("High prediction accuracy and low energy consumption.")
    app.toggle(key="goal_v3_use_llm").set_value(False)
    app.button(key="goal_v3_generate").click().run(timeout=30)
    assert not app.exception
    state = app.session_state["awareml_state"]
    assert state.get("copilot_context_free_interpretation") is not None
    assert state.get("copilot_proposal") is None


def test_trained_copilot_uses_current_prefix_bundle_end_to_end(bundle):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string("from awareml.ui_v2.pages_copilot import copilot_workspace_page\ncopilot_workspace_page()")
    app.session_state["awareml_state"] = {"dataset": frame(), "target": "target", "dataset_name": "test fixture"}
    app.run(timeout=30)
    app.text_area(key="goal_v3_text").set_value("High prediction accuracy and low energy consumption.")
    app.toggle(key="goal_v3_use_llm").set_value(False)
    app.button(key="goal_v3_generate").click().run(timeout=30)
    assert not app.exception
    state = app.session_state["awareml_state"]
    assert state.get("copilot_proposal") is not None
    assert state["copilot_meta"]["ranking"]["recommender_version"] == "v3"
    assert len(state["copilot_ranked"]) == 4
    assert state["copilot_prefix_profile"]["profile_sample_size"] == 1000
