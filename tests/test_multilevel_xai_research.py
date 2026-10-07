import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from awareml.analysis.explainability import _safe_probability_matrix, explain_framework
from awareml.explanation_integrity.attribution_alignment import alignment_scores, analyze_records, preference_pairs
from awareml.explanation_integrity.concept_audit import collect_probe, analyze_probe, validate_plan
from awareml.explanation_integrity.sequence_attribution import paired_shapley
from awareml.studies.behavioral_calibration import summarize_behavior, paired_participant_interval, pilot_assignment
from awareml.studies.behavioral_pilot_store import BehavioralPilotStore


def test_probability_columns_align_to_requested_labels():
    class Model:
        classes_ = np.array([0, 1])
        def predict_proba(self, X):
            return np.tile([.2, .8], (len(X), 1))
    values, _ = _safe_probability_matrix(Model(), pd.DataFrame({"x": [1, 2]}), [1, 0])
    np.testing.assert_allclose(values, [[.8, .2], [.8, .2]])


@pytest.mark.parametrize("values", [[[float("nan"), 1]], [[-1, 2]], [[0, 0]]])
def test_invalid_probabilities_unavailable(values):
    class Model:
        def predict_proba(self, X):
            return values
    assert _safe_probability_matrix(Model(), pd.DataFrame({"x": [1]}), [0, 1])[0] is None


def test_model_diagnostics_measured_without_changing_legacy_fields():
    class Model:
        def predict_one(self, x):
            return int(x["signal"] > .5)
    X = pd.DataFrame({"signal": [0, 1] * 30, "noise": [2] * 60})
    result = explain_framework(Model(), X, pd.Series([0, 1] * 30), method_preference="permutation")
    assert result["status"] == "ok"
    new = result["multilevel_diagnostics"]
    assert new["deletion_drop_signed"] == .5
    assert new["sufficiency_gap_signed"] == 0
    assert new["features_for_90pct_mass"] == 1
    assert result["fidelity"] == .5


def test_alignment_keeps_negative_direction():
    scores = alignment_scores([1, 2, 3], [-1, -2, -3], top_k=1)
    assert scores["cosine_signed"] == pytest.approx(-1)
    assert scores["spearman_signed"] == pytest.approx(-1)
    assert scores["spearman_absolute"] == pytest.approx(1)
    assert scores["absolute_mass_overlap"] == 1


def test_zero_and_constant_vectors_are_not_perfect_rank_alignment():
    assert alignment_scores([0, 0], [0, 0])["status"] == "unavailable"
    assert alignment_scores([1, 1], [1, 1])["spearman_signed"] is None


def test_topk_ties_are_not_arbitrary():
    score = alignment_scores([2, 2, 1], [2, 1, 2], top_k=1)
    assert score["top_k_jaccard"] == pytest.approx(1/3)


def record(explanation, values, split="train"):
    return {"case_id": "case", "model": "test-model", "attribution_method": "measured_test",
            "feature_ids": ["0", "1", "2"], "decision_attributions": [1, 2, 3],
            "explanation_attributions": values, "decision": "A", "explanation": explanation,
            "input_sha256": hashlib.sha256(b"input").hexdigest(), "target_definition": "test score", "split": split}


def test_preferences_train_only_and_matched_targets():
    records = [record("better", [1, 2, 3]), record("worse", [3, 2, 1])]
    assert preference_pairs(records)[0]["chosen"] == "better"
    records[1]["split"] = "test"
    assert preference_pairs(records) == []
    records[1]["split"] = "train"
    records[1]["input_sha256"] = hashlib.sha256(b"different").hexdigest()
    assert preference_pairs(records) == []


@pytest.mark.parametrize("mutation", ["duplicate_ids", "length", "nan", "missing_source"])
def test_attribution_invalid_inputs(mutation):
    r = record("x", [1, 2, 3])
    if mutation == "duplicate_ids": r["feature_ids"] = ["0", "0", "2"]
    if mutation == "length": r["decision_attributions"] = [1, 2]
    if mutation == "nan": r["decision_attributions"] = [1, 2, float("nan")]
    if mutation == "missing_source": del r["input_sha256"]
    with pytest.raises(ValueError): analyze_records([r])


def test_paired_shapley_additive_ground_truth():
    def score(subset):
        return (2 + sum([1, 2, -3][i] for i in subset), 4 + sum([3, -1, 2][i] for i in subset))
    out = paired_shapley(score, 3, 12)
    assert out["decision_attributions"] == [1, 2, -3]
    assert out["explanation_attributions"] == [3, -1, 2]
    assert out["additivity_residual"] == [0, 0]


def plan():
    return {"case_id": "C1", "prompt": "Choose by evidence", "choices": ["A", "B"],
            "interventions": [{"id": "hint", "concept": "authority", "kind": "hint", "hint_target": "B",
                               "prompt": "Choose by evidence. Expert prefers B.", "reviewed_single_change": True}]}


def test_probe_no_annotation_is_not_zero_hidden_hint_rate():
    def generate(prompt, seed):
        return {"answer": "B" if "Expert" in prompt else "A", "explanation": "Evidence supports this.", "model": "test"}
    bundle = collect_probe(plan(), generate, repeats=3)
    row = analyze_probe(bundle)["interventions"][0]
    assert row["hint_flip_count"] == 3
    assert row["hidden_hint_rate"] is None
    assert row["kl_cf_to_baseline"] > 0
    for r in bundle["records"]:
        if r["variant_id"] == "hint": r["hint_acknowledged"] = False
    assert analyze_probe(bundle)["interventions"][0]["hidden_hint_rate"] == 1


def test_probe_failures_not_scored_as_wrong_answers():
    def fail(prompt, seed): raise RuntimeError("offline")
    report = analyze_probe(collect_probe(plan(), fail, repeats=2))
    assert report["failed_calls"] == 4
    assert report["interventions"][0]["kl_cf_to_baseline"] is None


def test_probe_requires_review_and_rejects_duplicate_ids():
    p = plan(); p["interventions"][0]["reviewed_single_change"] = False
    with pytest.raises(ValueError): validate_plan(p)
    p = plan(); p["interventions"].append(copy.deepcopy(p["interventions"][0]))
    with pytest.raises(ValueError): validate_plan(p)


def response_frame():
    rows = []
    for i, (initial, ai, final, p) in enumerate([("B", "A", "A", 1.), ("A", "B", "B", 1.), ("A", "B", "A", 0.)]):
        rows.append(dict(participant_id="p", trial_id=str(i), dataset_id="d", condition="rationale",
                         ground_truth="A", ai_answer=ai, initial_answer=initial, final_answer=final, ai_correct_probability=p))
    return pd.DataFrame(rows)


def test_behavioral_denominators_and_brier():
    summary = summarize_behavior(response_frame())
    assert summary["beneficial_reliance"] == {"value": 1., "denominator": 1}
    assert summary["overreliance"] == {"value": .5, "denominator": 2}
    assert summary["harmful_switch"]["value"] == .5
    assert summary["brier_ai_correctness"] == pytest.approx(1/3)
    assert summary["accuracy_change"] == 0


def test_behavioral_missing_denominator_and_bad_probability():
    frame = response_frame().iloc[:1].copy()
    assert summarize_behavior(frame)["harmful_switch"]["value"] is None
    frame["ai_correct_probability"] = 7
    with pytest.raises(ValueError): summarize_behavior(frame)


def test_behavioral_duplicates_rejected():
    frame = response_frame()
    with pytest.raises(ValueError): summarize_behavior(pd.concat([frame, frame]))


def test_bootstrap_pairs_participants_not_trials():
    frame = response_frame()
    rows = []
    for participant in ["p1", "p2", "p3"]:
        for condition in ["answer_only", "rationale"]:
            f = frame.copy(); f["participant_id"] = participant; f["condition"] = condition
            f["trial_id"] += condition
            rows.append(f)
    result = paired_participant_interval(pd.concat(rows), "answer_only", "rationale", draws=100)
    assert result["paired_participants"] == 3
    assert result["ci95"] == [0, 0]


def bank():
    return {"schema_version": "xai_behavioral_pilot_v1", "cases": [
        {"case_id": str(i), "dataset_id": "d" + str(i), "question": "Which recorded accuracy is higher?",
         "choices": ["A", "B"], "ground_truth": "A", "ai_answer": "A", "rationale": "A is higher.",
         "diagnostics": "Recorded measurements", "task_evidence": "A .9; B .8", "ground_truth_policy": "max measured accuracy",
         "reviewed": True} for i in range(3)]}


def test_pilot_rotation_each_case_sees_all_conditions():
    seen = {}
    for index in range(3):
        for item in pilot_assignment(bank(), index):
            seen.setdefault(item["case"]["case_id"], set()).add(item["condition"])
    assert all(len(values) == 3 for values in seen.values())


def test_pilot_lock_resume_and_bank_integrity(tmp_path):
    store = BehavioralPilotStore(tmp_path / "pilot.sqlite")
    first = store.register("p", bank())
    assert store.register("p", bank()) == first
    store.save_initial("p", "0", {"answer": "B"})
    store.save_final("p", "0", {"answer": "A"})
    with pytest.raises(ValueError): store.save_final("p", "0", {"answer": "B"})
    changed = bank(); changed["cases"][0]["rationale"] = "Changed"
    with pytest.raises(ValueError): store.register("p", changed)
    assert store.get_trial("p", "0") == ({"answer": "B"}, {"answer": "A"})


def test_upgrade_installer_conflict_backup_and_idempotence(tmp_path):
    from scripts.apply_multilevel_xai_upgrade import apply
    project, package = tmp_path / "project", tmp_path / "package"
    (project / "awareml").mkdir(parents=True)
    (project / "app.py").write_text("original")
    (package / "payload").mkdir(parents=True)
    (package / "payload/app.py").write_text("upgraded")
    manifest = {"files": [{"path": "app.py", "base_sha256": hashlib.sha256(b"original").hexdigest(),
                           "new_sha256": hashlib.sha256(b"upgraded").hexdigest()}]}
    (package / "UPGRADE_MANIFEST.json").write_text(json.dumps(manifest))
    (project / "app.py").write_text("local edit")
    with pytest.raises(ValueError): apply(package, project)
    assert (project / "app.py").read_text() == "local edit"
    result = apply(package, project, allow_local_changes=True)
    from pathlib import Path
    assert (Path(result["backup"]) / "app.py").read_text() == "local edit"
    assert (project / "app.py").read_text() == "upgraded"
    assert apply(package, project)["status"] == "already_applied"


def test_upgrade_corrupt_package_does_not_write(tmp_path):
    from scripts.apply_multilevel_xai_upgrade import apply
    project, package = tmp_path / "project", tmp_path / "package"
    (project / "awareml").mkdir(parents=True)
    (project / "app.py").write_text("original")
    (package / "payload").mkdir(parents=True)
    (package / "payload/app.py").write_text("corrupt")
    (package / "UPGRADE_MANIFEST.json").write_text(json.dumps({"files": [{"path": "app.py", "new_sha256": "bad"}]}))
    with pytest.raises(ValueError): apply(package, project)
    assert (project / "app.py").read_text() == "original"
