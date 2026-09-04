from pathlib import Path

import pandas as pd

from awareml.ui_v2.pre14_usability import (
    analyze_target_task,
    cluster_positions,
    fairness_support_table,
    weights_for_objectives,
)


def test_equal_objective_weights():
    w = weights_for_objectives(["Accuracy", "Energy"])
    assert w == {"accuracy": 0.5, "runtime": 0.0, "energy": 0.5, "co2": 0.0}


def test_high_cardinality_numeric_target_guard():
    df = pd.DataFrame({"target": list(range(75)) * 3})
    info = analyze_target_task(df, "target")
    assert info["high_cardinality_numeric"] is True
    assert info["unique"] == 75


def test_fairness_support_marks_small_group_weak():
    df = pd.DataFrame({
        "target": [1] * 10 + [0] * 10 + [1] * 50 + [0] * 50,
        "group": ["small"] * 20 + ["large"] * 100,
    })
    table = fairness_support_table(df, "target", "group", 1)
    small = table[table["Group"].eq("small")].iloc[0]
    assert small["Support status"] == "weak"


def test_drift_clustering_is_display_only_logic():
    assert cluster_positions([100, 120, 180, 900, 920], 500) == [120, 910]


def test_installer_markers_present():
    root = Path(__file__).resolve().parents[1]
    core = (root / "awareml/ui_v2/pages_core.py").read_text(encoding="utf-8")
    copilot = (root / "awareml/ui_v2/pages_copilot.py").read_text(encoding="utf-8")
    specialist = (root / "awareml/ui_v2/pages_specialist.py").read_text(encoding="utf-8")
    observatory = (root / "awareml/ui_v2/pages_observatory.py").read_text(encoding="utf-8")
    assert "preference_context" in core
    assert "render_copilot_plan_summary" in copilot
    assert "render_decision_lab_explanation" in specialist
    assert "prepare_drift_display" in observatory
