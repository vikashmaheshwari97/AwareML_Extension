from pathlib import Path

import pandas as pd

from awareml.ui_v2.responsible_ai_visuals_v6_3 import (
    accuracy_energy_frontier,
    carbon_contribution_waterfall,
    fairness_gap_dumbbell,
    fairness_group_frame,
    fairness_over_time_figure,
    group_metric_matrix_figure,
    rai_cross_dimension_matrix,
    resource_tradeoff_statement,
    sustainability_decision_matrix,
)


def _sustainability_frame():
    return pd.DataFrame({
        "Framework": ["AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha"],
        "Accuracy": [0.723, 0.708, 0.705, 0.686, 0.586],
        "Runtime s": [17.3, 11.5, 46.1, 8.9, 19.0],
        "Energy kWh": [3.01e-4, 2.00e-4, 8.01e-4, 1.54e-4, 3.30e-4],
        "CO₂ kg": [1.26e-4, 8.34e-5, 3.34e-4, 6.43e-5, 1.37e-4],
        "Carbon intensity gCO2/kWh": [416.7] * 5,
    })


def _fairness_payload():
    return {
        "groups": {"Female": 1024, "Male": 938},
        "group_performance": {
            "Female": {
                "accuracy": 0.91,
                "true_positive_rate": 0.89,
                "false_positive_rate": 0.08,
                "predicted_positive_rate": 0.12,
                "positive_predictive_value": 0.82,
                "error_rate": 0.09,
            },
            "Male": {
                "accuracy": 0.86,
                "true_positive_rate": 0.78,
                "false_positive_rate": 0.15,
                "predicted_positive_rate": 0.20,
                "positive_predictive_value": 0.76,
                "error_rate": 0.14,
            },
        },
    }


def test_sustainability_visuals_are_quantitative_and_serializable():
    frame = _sustainability_frame()
    matrix = sustainability_decision_matrix(frame)
    frontier = accuracy_energy_frontier(frame)
    waterfall = carbon_contribution_waterfall(frame)
    assert len(matrix.data) == 1
    assert len(frontier.data) >= 5
    assert waterfall.data[0].type == "waterfall"
    matrix.to_json(); frontier.to_json(); waterfall.to_json()


def test_tradeoff_statement_uses_observed_values():
    statement = resource_tradeoff_statement(_sustainability_frame())
    assert statement["accuracy_framework"] == "AutoStreamML"
    assert statement["energy_framework"] == "OAML"
    assert statement["energy_ratio"] > 1.0
    assert statement["accuracy_delta_pp"] > 0.0


def test_group_matrix_and_dumbbell_use_recorded_group_metrics():
    fair = _fairness_payload()
    frame = fairness_group_frame(fair)
    assert set(frame["Group"]) == {"Female", "Male"}
    assert float(frame.loc[frame["Group"].eq("Female"), "TPR ↑"].iloc[0]) == 0.89
    matrix = group_metric_matrix_figure(fair, "Female", "TPR ↑")
    dumbbell = fairness_gap_dumbbell(fair, "TPR ↑")
    assert len(matrix.data) == 1
    assert len(dumbbell.data) == 3
    matrix.to_json(); dumbbell.to_json()


def test_temporal_fairness_uses_only_observed_points():
    result = {
        "framework": "AutoStreamML",
        "points": [
            {"sample": 500, "dp_diff": 0.04, "eo_diff": 0.05, "equalized_odds_gap": 0.07},
            {"sample": 1000, "dp_diff": 0.06, "eo_diff": 0.08, "equalized_odds_gap": 0.10},
        ],
        "drift_events": [800],
    }
    fig = fairness_over_time_figure(result)
    observed = set()
    for trace in fig.data:
        values = getattr(trace, "y", None)
        if values is None:
            continue
        for value in values:
            observed.add(round(float(value), 4))
    assert {0.04, 0.05, 0.06, 0.07, 0.08, 0.10}.issubset(observed)


def test_cross_dimension_matrix_keeps_exact_metrics():
    frame = pd.DataFrame({
        "Framework": ["AutoStreamML", "AutoClass"],
        "Accuracy": [0.72, 0.70],
        "DP gap": [0.04, 0.03],
        "XAI fidelity": [0.20, 0.18],
        "Energy (kWh)": [0.0003, 0.0002],
        "CO2 (kg)": [0.00012, 0.00008],
    })
    fig = rai_cross_dimension_matrix(frame)
    assert len(fig.data) == 1
    fig.to_json()


def test_pages_remove_old_sustainability_and_snapshot_bar_layouts():
    root = Path(__file__).resolve().parents[1]
    specialist = (root / "awareml" / "ui_v2" / "pages_specialist.py").read_text(encoding="utf-8")
    observatory = (root / "awareml" / "ui_v2" / "pages_observatory.py").read_text(encoding="utf-8")
    assert "sustainability_resource_scoreboard(sdf)" not in specialist
    assert "sustainability_rank_bump(sdf)" not in specialist
    assert "carbon_intensity_consistency_figure(sdf)" not in specialist
    assert "Framework sustainability matrix" in specialist
    assert "Accuracy–Energy efficient frontier" in specialist
    assert "Energy → CO₂ contribution waterfall" in specialist
    assert 'on_select="rerun"' in observatory
    assert "Group × metric fairness matrix" in observatory
    assert "Fairness–performance frontier" in observatory
    assert "rai_metric_bar(" not in observatory
