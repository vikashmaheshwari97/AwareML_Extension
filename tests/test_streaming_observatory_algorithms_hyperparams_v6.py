from __future__ import annotations

import numpy as np
import pandas as pd


def test_jsd_audit_is_zero_for_reference_and_detects_deliberate_shift():
    from awareml.ui_v2.stream_shift_audit import window_feature_distribution_shift

    n = 400
    df = pd.DataFrame({
        "x": np.r_[np.zeros(n // 2), np.ones(n // 2) * 10],
        "category": ["a"] * (n // 2) + ["b"] * (n // 2),
        "target": [0, 1] * (n // 2),
    })
    out = window_feature_distribution_shift(df, "target", 200, 400)
    ref = out[(out["Window"] == 1) & (out["Kind"] == "Feature")]
    shifted = out[(out["Window"] == 2) & (out["Kind"] == "Feature")]
    assert np.allclose(pd.to_numeric(ref["JSD"], errors="coerce").fillna(0), 0.0)
    assert float(pd.to_numeric(shifted["JSD"], errors="coerce").max()) > 0.5


def test_jsd_density_and_feature_rank_are_real_observed_feature_views():
    from awareml.ui_v2.stream_shift_audit import (
        window_feature_distribution_shift,
        jsd_density_figure,
        jsd_feature_ranking_figure,
    )

    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "a": np.r_[rng.normal(0, 1, 300), rng.normal(2, 1, 300)],
        "b": rng.normal(0, 1, 600),
        "target": np.r_[np.zeros(300), np.ones(300)],
    })
    audit = window_feature_distribution_shift(df, "target", 100, 600)
    density = jsd_density_figure(audit)
    ranking = jsd_feature_ranking_figure(audit)
    assert len(density.data) >= 1
    assert len(ranking.data) >= 2
    assert all(getattr(trace, "type", None) == "scatter" for trace in density.data)


def test_framework_catalog_restores_legacy_choices_without_faking_evo_selection():
    from awareml.frameworks.catalog import framework_catalog

    cat = framework_catalog()
    auto = cat["AutoStreamML"]["algorithms"]
    assert "AdaptiveRandomForestClassifier" in auto
    assert "HoeffdingAdaptiveTreeClassifier" in auto
    assert "LogisticRegression" in auto
    assert "GaussianNB" in auto

    autoclass = cat["AutoClass"]["algorithms"]
    assert "HoeffdingAdaptiveTreeClassifier" in autoclass
    assert "KNNADWINClassifier" in autoclass

    assert list(cat["EvoAutoML"]["algorithms"]) == ["EvolutionaryBaggingClassifier"]
    assert list(cat["ChaCha"]["algorithms"]) == ["AutoVW"]


def test_temporal_comparison_uses_quantitative_lines_not_heatmap():
    from awareml.ui_v2.temporal_visuals_v6 import temporal_comparison_figure

    results = [
        {"framework": "A", "points": [
            {"sample": 100, "rolling_accuracy": .70},
            {"sample": 200, "rolling_accuracy": .66},
            {"sample": 300, "rolling_accuracy": .72},
        ]},
        {"framework": "B", "points": [
            {"sample": 100, "rolling_accuracy": .65},
            {"sample": 200, "rolling_accuracy": .69},
            {"sample": 300, "rolling_accuracy": .68},
        ]},
    ]
    fig = temporal_comparison_figure(results)
    assert len(fig.data) == 4
    assert not any(getattr(trace, "type", None) == "heatmap" for trace in fig.data)


def test_hyperparameter_association_uses_recorded_history_only():
    from awareml.ui_v2.hyperparameter_analysis import parameter_association_frame, parameter_change_frame

    result = {
        "parameter_history": [
            {"sample": 100, "rolling_accuracy": .60, "rolling_f1_macro": .58, "active_algorithm": "X", "parameters": {"tunable": {"p": 1}}},
            {"sample": 200, "rolling_accuracy": .62, "rolling_f1_macro": .60, "active_algorithm": "X", "parameters": {"tunable": {"p": 2}}},
            {"sample": 300, "rolling_accuracy": .65, "rolling_f1_macro": .63, "active_algorithm": "X", "parameters": {"tunable": {"p": 3}}},
            {"sample": 400, "rolling_accuracy": .68, "rolling_f1_macro": .66, "active_algorithm": "X", "parameters": {"tunable": {"p": 4}}},
        ]
    }
    assoc = parameter_association_frame(result)
    rho = float(assoc.loc[assoc["Parameter"].eq("p"), "Spearman rho"].iloc[0])
    assert rho > 0.99
    changes = parameter_change_frame(result)
    assert len(changes) == 3


def test_new_source_keeps_scientific_boundaries():
    import inspect
    import awareml.ui_v2.pages_observatory as observatory
    import awareml.ui_v2.hyperparameter_analysis as hp

    o = inspect.getsource(observatory)
    h = inspect.getsource(hp)
    assert "No synthetic drift markers are generated" in o
    assert "not a hypothesis test" in o
    assert "not causal" in h.lower()
    assert "sensitivity = float(max(0.0, min(1.0, 1.0 - st)))" not in o + h
