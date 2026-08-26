import json

import pandas as pd

from awareml.analysis import explainability as xai_mod
from awareml.analysis.explainability import explain_framework
from awareml.engine.metrics import DriftRecoveryTracker, PredictionDiagnosticsTracker
from awareml.experiments.provenance import build_dataset_provenance, file_sha256


class ProbToy:
    def predict_one(self, x):
        return 1 if float(x["signal"]) >= 0.5 else 0

    def predict_proba_one(self, x):
        p = 0.9 if float(x["signal"]) >= 0.5 else 0.1
        return {0: 1.0 - p, 1: p}


def test_phase31_drift_no_degradation_episode_freezes_after_assessment():
    tracker = DriftRecoveryTracker(tolerance=0.02, min_assessment_samples=10)
    tracker.on_drift(100, baseline_accuracy=0.80, immediate_accuracy=0.80)
    tracker.update(110, 0.795)  # assessment ends: drop only 0.005
    tracker.update(150, 0.60)   # unrelated later deterioration must not mutate episode
    ep = tracker.summary()["episodes"][0]
    assert ep["degradation_observed"] is False
    assert abs(ep["accuracy_drop"] - 0.005) < 1e-12
    assert ep["recovery_samples"] is None


def test_phase31_drift_material_degradation_is_logically_consistent():
    tracker = DriftRecoveryTracker(tolerance=0.02, min_assessment_samples=10)
    tracker.on_drift(100, baseline_accuracy=0.80, immediate_accuracy=0.80)
    tracker.update(105, 0.75)
    tracker.update(110, 0.76)
    tracker.update(130, 0.79)
    ep = tracker.summary()["episodes"][0]
    assert ep["degradation_observed"] is True
    assert ep["accuracy_drop"] > 0.02
    assert ep["recovery_samples"] == 30


def test_phase31_prediction_diagnostics_flags_near_constant_stream():
    d = PredictionDiagnosticsTracker(positive_label=1, near_constant_threshold=0.95)
    for _ in range(96):
        d.update(0)
    for _ in range(4):
        d.update(1)
    out = d.summary()
    assert out["near_constant_prediction"] is True
    assert out["majority_prediction_fraction"] == 0.96
    assert out["positive_prediction_rate"] == 0.04
    assert out["warning"]


def test_phase31_replay_warning_threshold_is_parameterized():
    X = pd.DataFrame({"signal": [0.0, 1.0] * 40, "noise": [0.1, 0.2, 0.3, 0.4] * 20})
    y = pd.Series([0, 1] * 40)
    out = explain_framework(
        ProbToy(), X, y,
        method_preference="permutation",
        reference_accuracy=0.94,
        replay_warning_threshold=0.05,
        seed=7,
    )
    assert out["status"] == "ok"
    assert out["replay_accuracy_gap"] >= 0.05
    assert out["replay_warning_threshold"] == 0.05
    assert out["replay_warning"] is not None


def test_phase31_lime_categorical_names_follow_streaming_encoder_codes(monkeypatch):
    captured = {}

    class FakeExplanation:
        def as_map(self):
            return {1: [(0, 1.0), (1, 0.5)]}

    class FakeLime:
        def __init__(self, **kwargs):
            captured["categorical_names"] = kwargs.get("categorical_names")

        def explain_instance(self, row, predict_fn, **kwargs):
            # Exercise the prediction callback as real LIME would.
            predict_fn(pd.DataFrame([row]).to_numpy())
            return FakeExplanation()

    monkeypatch.setattr(xai_mod, "LimeTabularExplainer", FakeLime)
    X = pd.DataFrame({"signal": [0.0, 1.0] * 20, "workclass": [1.0, 2.0] * 20})
    y = pd.Series([0, 1] * 20)
    vectors, meta = xai_mod._lime_vectors(
        ProbToy(),
        X,
        y,
        seed=3,
        categorical_features=["workclass"],
        categorical_value_names={"workclass": {"Private": 1, "Self-emp": 2}},
    )
    names = captured["categorical_names"][1]
    assert names[0] == "<missing/unknown>"
    assert names[1] == "Private"
    assert names[2] == "Self-emp"
    assert len(vectors) == 3
    assert meta["categorical_feature_count"] == 1


def test_phase31_dataset_provenance_records_exact_file_hash_and_distributions(tmp_path):
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "sex": ["Female", "Male", "Female", "Male"],
        "income": [0, 0, 1, 1],
    })
    path = tmp_path / "adult_toy.csv"
    df.to_csv(path, index=False)
    out = build_dataset_provenance(df, target="income", sensitive_attribute="sex", source_path=path)
    assert out["source_sha256"] == file_sha256(path)
    assert out["rows"] == 4
    assert out["target_distribution"]["n_unique"] == 2
    assert out["sensitive_distribution"]["n_unique"] == 2
    assert [c["name"] for c in out["schema"]] == ["age", "sex", "income"]
