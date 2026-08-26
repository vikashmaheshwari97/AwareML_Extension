import pytest

from awareml.experiments.records import (
    FairnessSnapshotRecord,
    RunSummaryRecord,
    WindowMetricRecord,
    make_experiment_id,
)


def test_experiment_id_is_stable_with_nonce():
    a = make_experiment_id("adult", "AutoStreamML", 42, nonce="job-17")
    b = make_experiment_id("adult", "AutoStreamML", 42, nonce="job-17")
    c = make_experiment_id("adult", "AutoStreamML", 43, nonce="job-17")
    assert a == b
    assert a != c
    assert "autostreamml" in a


def test_not_measured_sustainability_cannot_hide_zeros():
    record = RunSummaryRecord(
        experiment_id="x", protocol_version="meta-v2", dataset_id="adult",
        framework="AutoStreamML", seed=42, status="ok",
        sustainability_status="not_measured", energy_kwh=None, co2_kg=None,
    )
    assert record.to_dict()["co2_kg"] is None
    with pytest.raises(ValueError):
        RunSummaryRecord(
            experiment_id="x2", protocol_version="meta-v2", dataset_id="adult",
            framework="AutoStreamML", seed=42, status="ok",
            sustainability_status="not_measured", energy_kwh=0.0, co2_kg=0.0,
        ).to_dict()


def test_insufficient_fairness_is_null_not_perfect():
    snap = FairnessSnapshotRecord(
        experiment_id="x", window_id=0, sample_index=10,
        status="insufficient_support", dp_diff=None, eo_diff=None,
    )
    assert snap.to_dict()["dp_diff"] is None
    with pytest.raises(ValueError):
        FairnessSnapshotRecord(
            experiment_id="x", window_id=0, sample_index=10,
            status="insufficient_support", dp_diff=0.0,
        ).to_dict()


def test_window_metrics_validate_probabilities():
    good = WindowMetricRecord(
        experiment_id="x", window_id=1, sample_index=500,
        prequential_accuracy=0.91, prequential_macro_f1=0.88,
    )
    assert good.to_dict()["prequential_accuracy"] == 0.91
    with pytest.raises(ValueError):
        WindowMetricRecord(
            experiment_id="x", window_id=1, sample_index=500,
            prequential_accuracy=1.1, prequential_macro_f1=0.88,
        ).to_dict()
