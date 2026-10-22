from pathlib import Path

import numpy as np
import pandas as pd

from awareml.ui_v2.stream_shift_audit import (
    distribution_shift_figure,
    jsd_density_figure,
    jsd_feature_ranking_figure,
    window_distribution_shift,
    window_feature_distribution_shift,
)
from awareml.ui_v2.temporal_visuals_v6 import (
    temporal_comparison_frame,
    temporal_deviation_figure,
    temporal_rank_flow_figure,
    temporal_stability_interval_figure,
)


def _synthetic_stream():
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "x": np.r_[rng.normal(0.0, 1.0, n), rng.normal(1.0, 1.0, n)],
        "cat": (["a"] * 250 + ["b"] * 250 + ["a"] * 100 + ["b"] * 400),
        "target": [0, 1] * n,
    })


def _results():
    return [
        {"framework": "AutoStreamML", "points": [
            {"sample": 500, "rolling_accuracy": 0.60},
            {"sample": 1000, "rolling_accuracy": 0.70},
            {"sample": 1500, "rolling_accuracy": 0.65},
        ]},
        {"framework": "AutoClass", "points": [
            {"sample": 500, "rolling_accuracy": 0.62},
            {"sample": 1000, "rolling_accuracy": 0.61},
            {"sample": 1500, "rolling_accuracy": 0.66},
        ]},
    ]


def test_jsd_evidence_and_figures_are_observed_not_synthetic():
    df = _synthetic_stream()
    audit = window_distribution_shift(df, "target", 500)
    feature = window_feature_distribution_shift(df, "target", 500)
    assert len(audit) == 2
    assert float(audit.iloc[0]["Max feature JSD"]) == 0.0
    assert float(audit.iloc[1]["Max feature JSD"]) > 0.0
    assert len(distribution_shift_figure(audit).data) >= 2
    assert len(jsd_density_figure(feature).data) >= 1
    assert len(jsd_feature_ranking_figure(feature).data) >= 2


def test_window_level_views_replace_heatmap_with_position_based_charts():
    results = _results()
    frame = temporal_comparison_frame(results)
    assert {"Deviation pp", "Window rank", "Framework median"}.issubset(frame.columns)
    assert len(temporal_deviation_figure(results).data) == 2
    assert len(temporal_rank_flow_figure(results).data) == 2
    assert len(temporal_stability_interval_figure(results).data) >= 6


def test_advanced_drift_lab_uses_canonical_streaming_observatory():
    text = Path("awareml/ui_v2/pages_advanced.py").read_text(encoding="utf-8")
    assert '"Drift & Temporal Lab": streaming_observatory_page' in text
    assert "drift_temporal_v2_page" not in text


def test_observatory_restores_primary_jsd_timeline_and_no_heatmap_call():
    text = Path("awareml/ui_v2/pages_observatory.py").read_text(encoding="utf-8")
    assert "Why can a stream show zero confirmed predictive drift?" in text
    assert "r122_distribution_shift_trend" in text
    assert "r122_jsd_density" in text
    assert "r122_jsd_feature_rank" in text
    assert "render_window_research_comparison" in text

    temporal = Path("awareml/ui_v2/temporal_visuals_v6.py").read_text(encoding="utf-8")
    assert "px.imshow" not in temporal
    assert "go.Heatmap" not in temporal
