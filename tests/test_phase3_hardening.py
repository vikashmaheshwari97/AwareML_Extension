import pandas as pd

from awareml.analysis.explainability import explain_framework
from awareml.engine.metrics import DriftRecoveryTracker
from awareml.engine.runner import _temporal_fairness_summary
from awareml.types import MetricPoint


class ProbToy:
    def predict_one(self, x):
        return 1 if x["signal"] >= 0.5 else 0

    def predict_proba_one(self, x):
        p = 0.9 if x["signal"] >= 0.5 else 0.1
        return {0: 1.0 - p, 1: p}


def test_phase3_permutation_explanation_is_non_degenerate():
    X = pd.DataFrame({"signal": [0.0, 1.0] * 40, "noise": [0.1, 0.2, 0.3, 0.4] * 20})
    y = pd.Series([0, 1] * 40)
    out = explain_framework(ProbToy(), X, y, method_preference="permutation", seed=7)
    assert out["status"] == "ok"
    assert out["method"] == "permutation/repeated"
    assert out["feature_importance"][0]["feature"] == "signal"
    assert out["feature_importance"][0]["importance"] > 0


class ConstantToy:
    def predict_one(self, x):
        return 0


def test_phase3_rejects_all_zero_explanation_as_unsupported():
    X = pd.DataFrame({"a": list(range(40)), "b": list(range(40, 80))})
    y = pd.Series([0] * 40)
    out = explain_framework(ConstantToy(), X, y, method_preference="permutation", seed=1)
    assert out["status"] == "unsupported"
    assert out["feature_importance"] == []


def test_drift_recovery_does_not_report_one_sample_recovery_without_degradation():
    tracker = DriftRecoveryTracker(tolerance=0.02, min_assessment_samples=20)
    tracker.on_drift(100, baseline_accuracy=0.80, immediate_accuracy=0.80)
    tracker.update(101, 0.80)
    tracker.update(120, 0.80)
    summary = tracker.summary()
    assert summary["n_recovered"] == 0
    assert summary["n_no_observed_degradation"] == 1
    assert summary["recovery_rate"] is None


def test_temporal_fairness_summary_uses_observed_windows_only():
    pts = [
        MetricPoint(sample=100, accuracy=.8, f1_macro=.7, dp_diff=.1, eo_diff=.2, worst_group_accuracy=.7, worst_group_macro_f1=.6),
        MetricPoint(sample=200, accuracy=.8, f1_macro=.7, dp_diff=.3, eo_diff=None, worst_group_accuracy=.65, worst_group_macro_f1=.55),
    ]
    out = _temporal_fairness_summary(pts)
    assert out["metrics"]["dp_diff"]["max"] == .3
    assert out["metrics"]["eo_diff"]["n"] == 1
    assert out["worst_over_time_accuracy"] == .65


class FakeAutoVW:
    def __init__(self):
        self.ready = False
        self.learn_calls = 0

    def predict(self, example):
        self.ready = True
        return 0.5

    def learn(self, example):
        assert self.ready
        self.learn_calls += 1


def test_chacha_native_warmup_calls_predict_before_learn():
    import pytest

    pytest.importorskip("river")
    from awareml.frameworks.chacha import ChaChaAdapter

    # Build only the minimal native OVR state required for this unit test.
    # The old Phase-3 private API (_native_predict/_native_learn) was replaced
    # by the public predict_one/learn_one path when the AwareML OVR extension
    # was introduced.
    obj = ChaChaAdapter.__new__(ChaChaAdapter)
    obj._native_available = True
    obj._fallback_active = False
    obj._labels = []
    obj._ovr_models = {}
    obj.autovw = FakeAutoVW()
    obj.native_error = None
    obj._AutoVW = None
    obj._loguniform = None

    assert obj.predict_one({"x": 1.0}) is None

    # The first class reuses obj.autovw. learn_one() must prime that AutoVW
    # learner with predict() immediately before learn().
    obj.learn_one({"x": 1.0}, 0)

    assert obj.autovw.learn_calls == 1
    assert obj._labels == [0]
    assert obj._ovr_models[0] is obj.autovw


def test_phase3_default_sensitive_policy_is_audit_only():
    from awareml.types import RunConfig

    cfg = RunConfig(target="target", sensitive_attribute="sex")
    assert cfg.sensitive_feature_policy == "audit_only"
