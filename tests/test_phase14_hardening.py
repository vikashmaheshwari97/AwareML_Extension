from __future__ import annotations

import math

from awareml.analysis.fairness import SlidingFairness
from awareml.analysis.repeatability import summarize_repeatability
from awareml.analysis.sustainability import SustainabilitySession


def test_calibration_unavailable_never_becomes_zero():
    fair = SlidingFairness(window_size=100, positive_label=1, min_group_n=2)
    for _ in range(4):
        fair.update(1, 1, "A")
        fair.update(0, 0, "B")
    result = fair.compute()
    assert result["calibration_status"] == "unavailable"
    assert result["group_brier_score_gap"] is None
    assert result["group_ece_gap"] is None


def test_group_brier_and_ece_gap_with_probabilities():
    fair = SlidingFairness(
        window_size=200,
        positive_label=1,
        min_group_n=4,
        calibration_bins=5,
    )
    for _ in range(5):
        fair.update(1, 1, "A", {1: 0.95, 0: 0.05})
        fair.update(0, 0, "A", {1: 0.05, 0: 0.95})
        fair.update(1, 1, "B", {1: 0.60, 0: 0.40})
        fair.update(0, 0, "B", {1: 0.40, 0: 0.60})

    result = fair.compute()
    assert result["calibration_status"] == "ok"
    assert result["group_brier_score_gap"] is not None
    assert result["group_brier_score_gap"] > 0
    assert result["group_ece_gap"] is not None
    assert result["group_ece_gap"] > 0
    assert 0 <= result["probability_coverage"] <= 1


def test_repeatability_reports_sample_sd():
    rows = [
        {
            "framework": "AutoClass",
            "runtime_sec": 10.0,
            "energy_kwh": 0.0010,
            "co2_kg": 0.0004,
        },
        {
            "framework": "AutoClass",
            "runtime_sec": 12.0,
            "energy_kwh": 0.0012,
            "co2_kg": 0.0005,
        },
        {
            "framework": "AutoClass",
            "runtime_sec": 11.0,
            "energy_kwh": 0.0011,
            "co2_kg": 0.00045,
        },
    ]
    table = summarize_repeatability(rows, min_repetitions=3)
    assert table.iloc[0]["Repeatability gate"] == "PASS"
    assert math.isclose(float(table.iloc[0]["Runtime (s) mean"]), 11.0)
    assert math.isclose(float(table.iloc[0]["Runtime (s) SD"]), 1.0)


def test_sustainability_not_measured_preserves_missing_values():
    record = SustainabilitySession(enabled=False).start().stop().to_dict()
    assert record["status"] == "not_measured"
    assert record["energy_kwh"] is None
    assert record["co2_kg"] is None
    assert "physical_cpus" in record
    assert "repetition_id" in record
    assert "measurement_failure_reason" in record
