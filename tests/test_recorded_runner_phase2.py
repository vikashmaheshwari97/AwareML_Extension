import json
from pathlib import Path

import pandas as pd
from awareml.engine.runner import _run_one
from awareml.experiments import ExperimentStore
from awareml.types import RunConfig


class MajorityToy:
    name = "Toy"

    def __init__(self):
        self.seed = 42
        self.backend = "toy"
        self.n0 = 0
        self.n1 = 0

    def predict_one(self, x):
        if self.n0 + self.n1 == 0:
            return None
        return 1 if self.n1 >= self.n0 else 0

    def learn_one(self, x, y):
        if int(y) == 1:
            self.n1 += 1
        else:
            self.n0 += 1

    def reset(self):
        self.n0 = self.n1 = 0

    def native_feature_importance(self):
        return {"x": 1.0}


def test_recorded_runner_emits_phase2_streams(tmp_path: Path):
    df = pd.DataFrame({
        "x": list(range(40)),
        "group": ["A", "B"] * 20,
        "target": [0, 1] * 20,
    })
    store = ExperimentStore(str(tmp_path / "runs"))
    cfg = RunConfig(
        target="target",
        sensitive_attribute="group",
        window_size=10,
        max_samples=40,
        time_budget_sec=30,
        track_sustainability=False,
        fairness_min_group_n=2,
    )
    result = _run_one(
        MajorityToy(),
        df,
        cfg,
        experiment_store=store,
        experiment_id="toy-run",
        dataset_id="toy-dataset",
        protocol_version="meta-v2",
    )
    assert result.status == "ok"
    assert result.samples == 40
    assert result.mean_prediction_latency_ms is not None
    assert result.p95_prediction_latency_ms is not None
    assert result.throughput_samples_sec is not None
    assert len(result.points) == 4

    run_dir = tmp_path / "runs" / "toy-run"
    assert (run_dir / "run.json").exists()
    assert (run_dir / "windows.jsonl").exists()
    assert (run_dir / "fairness.jsonl").exists()
    assert (run_dir / "explainability.jsonl").exists()
    assert (run_dir / "sustainability.jsonl").exists()

    summary = json.loads((run_dir / "run.json").read_text())
    assert summary["prequential_accuracy"] is not None
    assert summary["prequential_macro_f1"] is not None
    assert summary["mean_prediction_latency_ms"] is not None
    assert summary["p95_prediction_latency_ms"] is not None
    assert summary["sustainability_status"] == "not_measured"
    assert summary["energy_kwh"] is None
    assert summary["co2_kg"] is None
