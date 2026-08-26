from awareml.engine.metrics import (
    DriftRecoveryTracker,
    LatencyTracker,
    OnlineClassificationMetrics,
    RollingClassificationMetrics,
)


def test_missing_prediction_counts_as_incorrect_by_default():
    m = OnlineClassificationMetrics()
    m.update(1, None)
    m.update(1, 1)
    assert m.total == 2
    assert m.accuracy == 0.5


def test_rolling_accuracy_and_macro_f1():
    m = RollingClassificationMetrics(window_size=3)
    m.update(0, 0)
    m.update(1, 1)
    m.update(1, 0)
    assert abs(m.accuracy - (2 / 3)) < 1e-12
    assert 0.0 <= m.f1_macro <= 1.0
    m.update(1, 1)  # oldest pair is evicted
    assert m.n == 3
    assert abs(m.accuracy - (2 / 3)) < 1e-12


def test_latency_tracker_mean_and_p95():
    t = LatencyTracker()
    for value in [1, 2, 3, 4, 5]:
        t.update(value)
    assert t.mean_ms == 3.0
    assert t.p95_ms is not None
    assert t.p95_ms >= 4.0


def test_drift_recovery_summary():
    tr = DriftRecoveryTracker(tolerance=0.02)
    tr.on_drift(100, baseline_accuracy=0.90, immediate_accuracy=0.70)
    tr.update(110, 0.75)
    tr.update(130, 0.89)
    s = tr.summary()
    assert s["n_drift_events"] == 1
    assert s["n_recovered"] == 1
    assert s["mean_recovery_samples"] == 30.0
    assert s["max_accuracy_drop"] >= 0.15
