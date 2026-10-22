from pathlib import Path

import pandas as pd

from awareml.ui_v2.responsible_ai_visuals_v6_4 import (
    group_metric_profile_figure,
    fairness_disparity_structure_figure,
    fairness_tpr_fpr_plane,
    rai_directional_profile_figure,
    rai_rank_flow_figure,
    faithfulness_evidence_profile_figure,
    faithfulness_aef_comparison_figure,
)


def _fairness():
    return {
        "groups": {"1": 236, "2": 264},
        "group_performance": {
            "1": {
                "accuracy": 0.758,
                "true_positive_rate": 0.889,
                "false_positive_rate": 0.352,
                "predicted_positive_rate": 0.597,
                "positive_predictive_value": 0.681,
                "error_rate": 0.242,
            },
            "2": {
                "accuracy": 0.727,
                "true_positive_rate": 0.838,
                "false_positive_rate": 0.464,
                "predicted_positive_rate": 0.701,
                "positive_predictive_value": 0.757,
                "error_rate": 0.273,
            },
        },
    }


def _rai_frame():
    return pd.DataFrame({
        "Framework": ["AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha"],
        "Accuracy": [0.723, 0.708, 0.690, 0.712, 0.591],
        "DP gap": [0.103, 0.097, 0.015, 0.111, 0.043],
        "XAI fidelity": [0.160, 0.172, 0.244, 0.176, 0.000],
        "Energy (kWh)": [6.42e-4, 4.48e-4, 1.04e-3, 3.60e-4, 6.82e-4],
        "CO2 (kg)": [2.68e-4, 1.87e-4, 4.34e-4, 1.50e-4, 2.84e-4],
    })


def test_group_profile_replaces_box_matrix():
    fig = group_metric_profile_figure(_fairness())
    assert len(fig.data) >= 12
    assert not any(getattr(trace, "type", None) == "heatmap" for trace in fig.data)
    assert not any(getattr(getattr(trace, "marker", None), "symbol", None) == "square" for trace in fig.data)


def test_disparity_and_operating_point_views_are_quantitative():
    disparity = fairness_disparity_structure_figure(_fairness())
    operating = fairness_tpr_fpr_plane(_fairness())
    assert len(disparity.data) >= 2
    assert len(operating.data) == 2
    assert operating.layout.xaxis.title.text.startswith("False-positive")
    assert operating.layout.yaxis.title.text.startswith("True-positive")


def test_cross_dimension_views_do_not_use_heatmaps():
    frame = _rai_frame()
    for fig in (rai_directional_profile_figure(frame), rai_rank_flow_figure(frame)):
        assert len(fig.data) == 5
        assert not any(getattr(trace, "type", None) == "heatmap" for trace in fig.data)


def test_faithfulness_views_replace_plain_component_bars():
    deterministic = {
        "mean_grounding_validity": 1.0,
        "mean_decision_alignment": 1.0,
        "mean_attribution_alignment": 1.0,
        "mean_counterfactual_sensitivity": 0.758,
        "mean_irrelevant_invariance": 1.0,
        "mean_evidence_fidelity_score": 0.939,
    }
    ollama = {"mean_evidence_fidelity_score": 0.947}
    profile = faithfulness_evidence_profile_figure(deterministic)
    compare = faithfulness_aef_comparison_figure(deterministic, ollama)
    assert profile.data
    assert compare.data
    assert not any(getattr(trace, "type", None) == "bar" for trace in profile.data)


def test_observatory_source_removes_linked_focus_controls():
    source = (Path(__file__).resolve().parents[1] / "awareml" / "ui_v2" / "pages_observatory.py").read_text(encoding="utf-8")
    assert "Linked group focus" not in source
    assert "Linked metric focus" not in source
    assert "rai_cross_dimension_matrix" not in source
    assert "faithfulness_components(" not in source
    assert "group_metric_profile_figure" in source
    assert "rai_directional_profile_figure" in source
    assert "faithfulness_evidence_profile_figure" in source
