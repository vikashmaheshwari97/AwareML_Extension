from __future__ import annotations

from pathlib import Path

import pandas as pd

from awareml.ui_v2.advanced_visuals import (
    fairness_cross_criterion_interval,
    fairness_ranked_small_multiples,
    temporal_fairness_envelope_figure,
    temporal_fairness_robustness_figure,
)


def _fairness_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Framework": ["AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha"],
        "Demographic parity": [0.116, 0.028, 0.131, 0.029, 0.063],
        "Equal opportunity": [0.003, 0.048, 0.018, 0.010, 0.018],
        "Equalized odds": [0.019, 0.048, 0.079, 0.074, 0.087],
        "Error-rate parity": [0.032, 0.021, 0.010, 0.069, 0.139],
        "Comparable mean gap": [0.0425, 0.03625, 0.0595, 0.0455, 0.07675],
        "Worst available gap": [0.116, 0.048, 0.131, 0.074, 0.139],
    })


def test_aggregate_fairness_uses_position_based_research_visuals():
    frame = _fairness_frame()
    metrics = ["Demographic parity", "Equal opportunity", "Equalized odds", "Error-rate parity"]
    ranked = fairness_ranked_small_multiples(frame, metrics)
    robustness = fairness_cross_criterion_interval(frame, metrics)
    assert ranked.data
    assert robustness.data
    assert "Criterion-wise fairness ranking" in ranked.layout.title.text
    assert "Cross-criterion robustness" in robustness.layout.title.text


def test_temporal_fairness_uses_distribution_envelope_and_robustness_forest():
    results = []
    values = {
        "AutoStreamML": [0.06, 0.04, 0.08, 0.08, 0.05, 0.11, 0.13, 0.07, 0.145, 0.09],
        "AutoClass": [0.065, 0.05, 0.05, 0.07, 0.045, 0.075, 0.10, 0.035, 0.12, 0.06],
        "EvoAutoML": [0.01, 0.04, 0.08, 0.05, 0.04, 0.09, 0.13, 0.04],
        "OAML": [0.06, 0.025, 0.048, 0.075, 0.05, 0.06, 0.108, 0.098, 0.047, 0.093],
        "ChaCha": [0.018, 0.013, 0.006, 0.016, 0.047, 0.007, 0.058, 0.084, 0.045, 0.057],
    }
    for framework, seq in values.items():
        results.append({
            "framework": framework,
            "points": [{"sample": 1000 * (i + 1), "dp_diff": value} for i, value in enumerate(seq)],
        })
    trajectory = temporal_fairness_envelope_figure(results, "dp_diff", "Demographic parity")
    assert trajectory.data
    assert "median/IQR envelope" in trajectory.layout.title.text

    rows = []
    for result in results:
        seq = pd.Series([point["dp_diff"] for point in result["points"]], dtype=float)
        rows.append({
            "Framework": result["framework"],
            "Mean gap": float(seq.mean()),
            "P95 gap": float(seq.quantile(0.95)),
            "Worst-window gap": float(seq.max()),
            "Temporal volatility": float(seq.std(ddof=0)),
            "Windows": int(len(seq)),
        })
    robustness = temporal_fairness_robustness_figure(pd.DataFrame(rows))
    assert robustness.data
    assert "Temporal robustness profile" in robustness.layout.title.text


def test_fairness_page_no_longer_uses_old_bubble_views():
    source = (Path(__file__).resolve().parents[1] / "awareml" / "ui_v2" / "pages_specialist.py").read_text(encoding="utf-8")
    assert "r101_fair_ranked_small_multiples" in source
    assert "r101_fair_cross_criterion_interval" in source
    assert "r101_fair_temporal_envelope" in source
    assert "r101_fair_temporal_robustness" in source
    assert "r100_fair_bubble_matrix" not in source
    assert "r100_fair_risk_map" not in source
