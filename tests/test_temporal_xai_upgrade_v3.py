from pathlib import Path

import pandas as pd

from awareml.engine.drift_controller import adapt_framework
from awareml.types import RunConfig
from awareml.ui_v2.pareto_frontier import build_sustainability_pareto_frame

ROOT = Path(__file__).resolve().parents[1]


class _DummyFramework:
    def __init__(self):
        self.reset_calls = 0
        self.learned = []

    def reset(self):
        self.reset_calls += 1
        self.learned = []

    def learn_one(self, x, y):
        self.learned.append((dict(x), y))


def test_generic_adaptive_replay_is_explicit_and_bounded():
    fw = _DummyFramework()
    history = [({"x": float(i)}, i % 2) for i in range(12)]
    event = adapt_framework(
        fw,
        history,
        sample_index=250,
        policy="adaptive_replay",
        replay_size=5,
        trigger_sources=["ADWIN", "performance_drop"],
    )
    assert event["status"] == "ok"
    assert event["action"] == "awareml_reset_and_replay"
    assert event["replayed_samples"] == 5
    assert fw.reset_calls == 1
    assert len(fw.learned) == 5
    assert "not an upstream-native refit claim" in event["note"]


def test_runconfig_defaults_preserve_frozen_historical_behavior():
    cfg = RunConfig(target="y")
    assert cfg.drift_detector_mode == "adwin"
    assert cfg.drift_action_policy == "monitor_only"


def test_pareto_uses_combined_energy_co2_burden_and_framework_rows():
    frame = pd.DataFrame([
        {"framework": "A", "accuracy": 0.90, "energy_kwh": 0.2, "co2_kg": 0.2, "runtime_sec": 10},
        {"framework": "B", "accuracy": 0.88, "energy_kwh": 0.1, "co2_kg": 0.1, "runtime_sec": 20},
        {"framework": "C", "accuracy": 0.80, "energy_kwh": 0.3, "co2_kg": 0.3, "runtime_sec": 30},
    ])
    out = build_sustainability_pareto_frame(frame)
    assert {"energy_norm", "co2_norm", "sustainability_burden", "pareto_2d"}.issubset(out.columns)
    assert out.loc[out.framework.eq("B"), "sustainability_burden"].iloc[0] < out.loc[out.framework.eq("A"), "sustainability_burden"].iloc[0]
    assert set(out.loc[out.pareto_2d, "framework"]) == {"A", "B"}


def test_streaming_observatory_is_consolidated_inside_advanced_labs():
    top = (ROOT / "awareml/ui_v2/pages.py").read_text(encoding="utf-8")
    advanced = (ROOT / "awareml/ui_v2/pages_advanced.py").read_text(encoding="utf-8")
    observatory = (ROOT / "awareml/ui_v2/pages_observatory.py").read_text(encoding="utf-8")
    assert '"Streaming Observatory":' not in top
    assert "Streaming Observatory · temporal analytics" in advanced
    assert "Post-drift accuracy & drift-loss recovery" in observatory
    assert "Detector health & adaptation" in observatory


def test_decision_lab_contains_interactive_sustainability_pareto():
    source = (ROOT / "awareml/ui_v2/pages_specialist.py").read_text(encoding="utf-8")
    assert "Interactive Pareto Frontier" in source
    assert "render_sustainability_accuracy_pareto" in source
    pareto = (ROOT / "awareml/ui_v2/pareto_frontier.py").read_text(encoding="utf-8")
    assert "Normalized CO₂ + Energy burden" in pareto
    assert "2D Pareto frontier" in pareto


def test_explainability_upgrade_keeps_phase16_boundary_and_real_alignment_requirement():
    source = (ROOT / "awareml/ui_v2/pages_multilevel_explainability.py").read_text(encoding="utf-8")
    assert "Decision–explanation attribution alignment readiness" in source
    assert "cannot be reconstructed from the existing Phase-16 responses" in source
    assert "do not require repeating the original Phase-16 participant experiment" in source
    assert "separately versioned supplementary pilot/study" in source


def test_ollama_default_is_llama3_8b_and_grounded_summary_is_multilevel():
    config = (ROOT / "awareml/config.py").read_text(encoding="utf-8")
    summary = (ROOT / "awareml/llm/xai_grounded_summary.py").read_text(encoding="utf-8")
    specialist = (ROOT / "awareml/ui_v2/pages_specialist.py").read_text(encoding="utf-8")
    assert '"llama3:8b"' in config
    assert "llama3.1:8b" not in config
    assert "Model-level evidence" in summary
    assert "Hyperparameter/context evidence" in summary
    assert "System-level comparison" in summary
    assert "llama3:8b" in specialist


def test_runner_and_run_studio_are_wired_for_optional_enhanced_drift_handling():
    runner = (ROOT / "awareml/engine/runner.py").read_text(encoding="utf-8")
    run_ui = (ROOT / "awareml/ui/pages.py").read_text(encoding="utf-8")
    assert "HybridDriftController" in runner
    assert "adapt_framework" in runner
    assert "refit_events=refit_events" in runner
    assert "Hybrid · ADWIN + Page-Hinkley + degradation confirmation" in run_ui
    assert "Adaptive reset + recent-window replay" in run_ui
