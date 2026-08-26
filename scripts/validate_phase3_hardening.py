from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.analysis.explainability import explain_framework
from awareml.engine.metrics import DriftRecoveryTracker
from awareml.types import RunConfig


class _ToyProbModel:
    def predict_one(self, x):
        return 1 if float(x["signal"]) >= 0.5 else 0

    def predict_proba_one(self, x):
        p = 0.9 if float(x["signal"]) >= 0.5 else 0.1
        return {0: 1.0 - p, 1: p}


def main() -> None:
    cfg = RunConfig(target="target", sensitive_attribute="group")
    assert cfg.xai_method == "auto"
    assert cfg.sensitive_feature_policy == "audit_only"

    X = pd.DataFrame({"signal": [0.0, 1.0] * 40, "noise": [0.1, 0.2, 0.3, 0.4] * 20})
    y = pd.Series([0, 1] * 40)
    exp = explain_framework(_ToyProbModel(), X, y, method_preference="permutation", seed=7)
    assert exp["status"] == "ok"
    assert exp["feature_importance"][0]["feature"] == "signal"
    assert exp["feature_importance"][0]["importance"] > 0

    tracker = DriftRecoveryTracker(tolerance=0.02, min_assessment_samples=20)
    tracker.on_drift(100, baseline_accuracy=0.8, immediate_accuracy=0.8)
    tracker.update(101, 0.8)
    tracker.update(120, 0.8)
    drift = tracker.summary()
    assert drift["n_recovered"] == 0
    assert drift["n_no_observed_degradation"] == 1

    print("AwareML Phase 3 hardening validation: OK")
    print("XAI cascade configured : SHAP -> LIME -> repeated permutation")
    print("Degenerate XAI policy  : unsupported, never decorative all-zero evidence")
    print("Fairness feature policy: audit_only by default")
    print("Drift recovery guard   : post-drift assessment required")

    # Native ChaCha is environment-specific because it depends on FLAML + VW.
    # Probe it without making the whole validator fail; the subsequent recorded
    # run is the authoritative local check.
    try:
        from awareml.frameworks.chacha import ChaChaAdapter

        chacha = ChaChaAdapter(seed=42)
        for x, yv in [
            ({"x": 0.0, "z": 1.0}, 0),
            ({"x": 1.0, "z": 0.0}, 1),
            ({"x": 0.2, "z": 0.8}, 0),
            ({"x": 0.9, "z": 0.1}, 1),
        ]:
            chacha.predict_one(x)
            chacha.learn_one(x, yv)
        print("ChaCha local probe      :", chacha.backend)
        if "fallback" in str(chacha.backend).lower():
            print("ChaCha probe warning    :", chacha.native_error)
        try:
            chacha.close()
        except Exception:
            pass
    except Exception as exc:
        print("ChaCha local probe      : unavailable (%s: %s)" % (type(exc).__name__, exc))


if __name__ == "__main__":
    main()
