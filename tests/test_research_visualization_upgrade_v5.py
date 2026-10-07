from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from awareml.ui_v2.pareto_frontier import build_sustainability_pareto_frame
from awareml.ui_v2.research_visuals_v5 import (
    carbon_intensity_consistency_figure,
    correlation_arc_figure,
    correlation_pair_ranking,
    desirability_profile_figure,
    fairness_rank_bump_figure,
    fairness_robustness_frontier,
    fairness_window_evidence_figure,
    sustainability_efficiency_frontier,
    sustainability_rank_bump,
    sustainability_resource_scoreboard,
    system_xai_rank_bump,
    system_xai_ranked_panels,
    utility_decomposition_figure,
)


FRAMEWORKS = ["AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha"]


def _decision_frames():
    ranking = pd.DataFrame({
        "framework": FRAMEWORKS,
        "accuracy": [0.94, 0.97, 0.96, 0.98, 0.91],
        "energy_kwh": [0.0004, 0.0003, 0.0008, 0.0002, 0.0005],
        "co2_kg": [0.0002, 0.00015, 0.0004, 0.0001, 0.00025],
        "runtime_sec": [24, 19, 57, 17, 26],
        "utility": [0.75, 0.82, 0.67, 0.86, 0.61],
    })
    contrib = pd.DataFrame({
        "framework": FRAMEWORKS,
        "accuracy": [0.42, 0.44, 0.43, 0.45, 0.38],
        "runtime": [0.10, 0.12, 0.04, 0.15, 0.09],
        "energy": [0.07, 0.08, 0.02, 0.10, 0.06],
        "co2": [0.07, 0.08, 0.02, 0.10, 0.06],
        "fairness": [0.12, 0.09, 0.11, 0.14, 0.13],
        "interpretability": [0.05, 0.05, 0.05, 0.05, 0.04],
    })
    norm = contrib.copy()
    for col in norm.columns[1:]:
        s = norm[col]
        norm[col] = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.5
    return ranking, contrib, norm


def _fairness_results():
    results = []
    for i, fw in enumerate(FRAMEWORKS):
        pts = []
        for j in range(1, 7):
            pts.append({
                "sample": j * 1000,
                "dp_diff": 0.02 + 0.01 * i + 0.004 * j,
                "equal_opportunity_diff": 0.01 + 0.006 * i + 0.003 * j,
                "equalized_odds_gap": 0.015 + 0.008 * i + 0.0035 * j,
                "error_rate_gap": 0.006 + 0.004 * i + 0.002 * j,
            })
        results.append({
            "framework": fw,
            "points": pts,
            "fairness": {
                "dp_diff": float(np.mean([p["dp_diff"] for p in pts])),
                "equal_opportunity_diff": float(np.mean([p["equal_opportunity_diff"] for p in pts])),
                "equalized_odds_gap": float(np.mean([p["equalized_odds_gap"] for p in pts])),
                "error_rate_gap": float(np.mean([p["error_rate_gap"] for p in pts])),
            },
        })
    return results


def test_decision_visuals_are_position_based():
    ranking, contrib, norm = _decision_frames()
    f1 = utility_decomposition_figure(contrib, ranking)
    f2 = desirability_profile_figure(norm)
    assert any(trace.type == "bar" for trace in f1.data)
    assert len(f2.data) == len(FRAMEWORKS)
    assert all(trace.mode == "lines+markers" for trace in f2.data)


def test_objective_relationship_views_render():
    names = ["accuracy", "runtime_sec", "energy_kwh", "co2_kg", "fairness_score", "interpretability_score"]
    corr = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    corr.loc["runtime_sec", "energy_kwh"] = corr.loc["energy_kwh", "runtime_sec"] = 1.0
    corr.loc["runtime_sec", "co2_kg"] = corr.loc["co2_kg", "runtime_sec"] = 1.0
    corr.loc["accuracy", "fairness_score"] = corr.loc["fairness_score", "accuracy"] = 0.4
    corr.loc["fairness_score", "runtime_sec"] = corr.loc["runtime_sec", "fairness_score"] = -0.7
    assert len(correlation_arc_figure(corr).data) >= len(names)
    assert len(correlation_pair_ranking(corr).data) == 1


def test_fairness_paper_style_windows_and_robustness():
    results = _fairness_results()
    metric_map = {
        "Demographic parity": "dp_diff",
        "Equal opportunity": "equal_opportunity_diff",
        "Equalized odds": "equalized_odds_gap",
        "Error-rate parity": "error_rate_gap",
    }
    fig = fairness_window_evidence_figure(results, metric_map)
    # 4 metrics x 5 frameworks x (window + diamond traces)
    assert len(fig.data) >= 4 * len(FRAMEWORKS)
    fair = pd.DataFrame([
        {"Framework": r["framework"], **{label: r["fairness"][key] for label, key in metric_map.items()}}
        for r in results
    ])
    assert len(fairness_rank_bump_figure(fair, metric_map).data) == len(FRAMEWORKS)
    assert len(fairness_robustness_frontier(fair, metric_map).data) >= len(FRAMEWORKS)


def test_system_xai_has_no_bubble_area_comparison():
    cdf = pd.DataFrame({
        "Framework": FRAMEWORKS,
        "Fidelity": [0.128, 0.176, 0.180, 0.192, 0.056],
        "Stability": [0.429, 0.508, 0.508, 0.508, 1.000],
        "Consistency": [0.993, 0.932, 0.899, 0.932, 0.984],
        "Sensitivity": [0.127, 0.174, 0.235, 0.232, 0.102],
        "Sparsity": [0.859, 0.721, 0.485, 0.694, 0.615],
    })
    panels = system_xai_ranked_panels(cdf)
    bump = system_xai_rank_bump(cdf)
    assert len(panels.data) > 0
    assert len(bump.data) == len(FRAMEWORKS)
    # All comparison markers use fixed scalar sizes; there is no value-sized bubble array.
    for trace in panels.data:
        if getattr(trace, "marker", None) is not None and getattr(trace.marker, "size", None) is not None:
            assert np.isscalar(trace.marker.size)


def test_sustainability_advanced_views_render():
    sdf = pd.DataFrame({
        "Framework": FRAMEWORKS,
        "Runtime s": [24, 19, 57, 17, 26],
        "Energy kWh": [0.0004, 0.0003, 0.0010, 0.0003, 0.0005],
        "CO₂ kg": [0.0002, 0.0001, 0.0004, 0.0001, 0.0002],
        "Carbon intensity gCO2/kWh": [500, 333.3, 400, 333.3, 400],
    })
    assert len(sustainability_resource_scoreboard(sdf).data) > 0
    assert len(sustainability_rank_bump(sdf).data) == len(FRAMEWORKS)
    assert len(sustainability_efficiency_frontier(sdf).data) >= len(FRAMEWORKS)
    assert len(carbon_intensity_consistency_figure(sdf).data) > 0


def test_pareto_adds_one_transparent_2d_optimum():
    ranking, _, _ = _decision_frames()
    out = build_sustainability_pareto_frame(ranking)
    valid = out.dropna(subset=["sustainability_burden", "accuracy_numeric"])
    assert int(valid["optimal_2d"].sum()) == 1
    assert bool(valid.loc[valid["optimal_2d"], "pareto_2d"].iloc[0])


def test_pages_specialist_uses_v5_visuals():
    source = (Path(__file__).resolve().parents[1] / "awareml" / "ui_v2" / "pages_specialist.py").read_text(encoding="utf-8")
    required = [
        "r110_decision_utility_decomposition",
        "r110_decision_corr_arcs",
        "r110_fair_window_evidence",
        "r110_xai_system_ranked_panels",
        "r110_sustain_resource_scoreboard",
        "r110_sustain_carbon_intensity",
    ]
    for token in required:
        assert token in source
    assert "utility_contribution_map(contrib, frame)" not in source
    assert "system_xai_evidence_map(valid_system)" not in source
