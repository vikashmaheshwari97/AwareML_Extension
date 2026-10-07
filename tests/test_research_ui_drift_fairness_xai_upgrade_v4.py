from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from awareml.engine.drift_controller import HybridDriftController
from awareml.ui_v2.advanced_visuals import (
    correlation_network,
    desirability_radar,
    fairness_mean_worst_dumbbell,
    system_xai_evidence_map,
    utility_contribution_map,
)
from awareml.ui_v2.fairness_dynamics import (
    fairness_debt_summary,
    fairness_series_frame,
    replay_fairness_audit,
)
from awareml.ui_v2.stream_shift_audit import window_distribution_shift


def test_sustained_performance_drop_can_confirm_change_without_statistical_vote():
    controller = HybridDriftController(
        mode="hybrid",
        window_size=100,
        warmup_samples=20,
        min_separation=15,
        performance_drop=0.03,
        degradation_confirm_samples=5,
    )
    events = []
    for sample in range(1, 101):
        accuracy = 0.90 if sample <= 50 else 0.55
        signal = controller.update(1.0 - accuracy, accuracy, sample)
        if signal.detected:
            events.append(signal)
    assert events
    assert "sustained_performance_drop" in events[0].sources
    summary = controller.summary()
    assert summary["confirmed_drifts"] >= 1
    assert summary["warning_steps"] >= summary["warning_episodes"]


def test_fairness_debt_replay_preserves_observed_values_and_flags_changes():
    results = [{
        "framework": "AutoClass",
        "points": [
            {"sample": 1000, "dp_diff": 0.11},
            {"sample": 2000, "dp_diff": 0.055},
            {"sample": 3000, "dp_diff": 0.048},
            {"sample": 4000, "dp_diff": 0.248},
            {"sample": 5000, "dp_diff": 0.145},
        ],
    }]
    series = fairness_series_frame(results, "dp_diff")
    audit = replay_fairness_audit(series, initial_upper=0.10, debt_decay=0.90)
    assert np.allclose(audit["Gap"].to_numpy(), series["Gap"].to_numpy())
    assert int((audit["Event"] == "fairness_change").sum()) >= 2
    assert int((audit["Violation"] > 0).sum()) >= 1
    assert float(audit["Debt"].max()) > 0.0
    summary = fairness_debt_summary(audit)
    assert len(summary) == 1
    assert int(summary.iloc[0]["Fairness-change flags"]) >= 2


def test_distribution_shift_audit_detects_a_large_deliberate_change():
    rng = np.random.default_rng(42)
    first = pd.DataFrame({
        "feature": rng.normal(0, 1, 1000),
        "category": ["a"] * 800 + ["b"] * 200,
        "target": rng.integers(0, 2, 1000),
    })
    second = pd.DataFrame({
        "feature": rng.normal(4, 1, 1000),
        "category": ["a"] * 150 + ["b"] * 850,
        "target": rng.integers(0, 2, 1000),
    })
    frame = pd.concat([first, second], ignore_index=True)
    audit = window_distribution_shift(frame, "target", window_size=1000, processed_samples=2000)
    assert len(audit) == 2
    assert float(audit.iloc[1]["Max feature JSD"]) > 0.20


def test_new_decision_and_xai_visuals_return_plotly_traces():
    contrib = pd.DataFrame({
        "framework": ["A", "B"],
        "accuracy": [0.40, 0.25], "runtime": [0.10, 0.18],
        "energy": [0.08, 0.12], "co2": [0.08, 0.12],
        "fairness": [0.11, 0.08], "interpretability": [0.05, 0.06],
    })
    ranking = pd.DataFrame({"framework": ["A", "B"], "utility": [0.82, 0.70], "rank": [1, 2]})
    assert utility_contribution_map(contrib, ranking).data

    norm = pd.DataFrame({
        "framework": ["A", "B"], "accuracy": [1.0, 0.6], "runtime": [0.4, 1.0],
        "energy": [0.7, 0.8], "co2": [0.7, 0.8], "fairness": [0.8, 0.6],
        "interpretability": [0.7, 0.9],
    })
    assert len(desirability_radar(norm).data) == 2

    corr = pd.DataFrame([[1.0, 0.95], [0.95, 1.0]], index=["energy", "co2"], columns=["energy", "co2"])
    assert correlation_network(corr).data

    fair = pd.DataFrame({
        "Framework": ["A", "B"],
        "Comparable mean gap": [0.08, 0.12],
        "Worst available gap": [0.14, 0.24],
    })
    assert fairness_mean_worst_dumbbell(fair).data

    xai = pd.DataFrame({
        "Framework": ["A", "B"], "Method used": ["SHAP/Kernel", "permutation/repeated"],
        "Fidelity": [0.18, 0.13], "Stability": [0.51, 0.43],
        "Consistency": [0.93, 0.99], "Sensitivity": [0.17, 0.13], "Sparsity": [0.72, 0.86],
    })
    assert system_xai_evidence_map(xai).data


def test_concept_probe_zero_result_is_explicitly_explained_in_source():
    source = (Path(__file__).resolve().parents[1] / "awareml" / "ui_v2" / "pages_multilevel_explainability.py").read_text(encoding="utf-8")
    assert "every plotted effect was exactly 0.000" in source
    assert "zero remains visible" in source


def test_explainability_uses_explainer_pattern_not_old_toggle():
    source = (Path(__file__).resolve().parents[1] / "awareml" / "ui_v2" / "pages_specialist.py").read_text(encoding="utf-8")
    assert "LLM-Assisted Explainability Explainer" in source
    assert "Use local Ollama for the grounded answer" not in source
    assert "System XAI evidence map" in source
