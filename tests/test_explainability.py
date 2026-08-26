import pandas as pd

from awareml.analysis.explainability import _safe_accuracy, explain_framework


class BoolStreamModel:
    def predict_one(self, x):
        # Deliberately returns Python bool while y uses integer labels.
        return bool(x["signal"] > 0.5)


def test_streaming_explainability_handles_bool_vs_binary_labels():
    X = pd.DataFrame({
        "signal": ([0.0, 1.0] * 30),
        "noise": ([0.2, 0.8, 0.4] * 20),
    })
    y = pd.Series([0, 1] * 30)
    out = explain_framework(BoolStreamModel(), X, y, seed=42)
    assert out["status"] == "ok"
    assert out["base_accuracy_on_explanation_window"] == 1.0
    assert out["feature_importance"][0]["feature"] == "signal"


def test_safe_accuracy_counts_missing_prediction_as_incorrect():
    assert _safe_accuracy([0, 1, 1], [False, True, None]) == 2 / 3
