from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.engine.runner import _run_one
from awareml.experiments import ExperimentStore
from awareml.types import RunConfig


class _ToyFramework:
    name = "Phase2Toy"
    backend = "phase2-validator"

    def __init__(self):
        self.seed = 42
        self.counts = {0: 0, 1: 0}

    def predict_one(self, x):
        if sum(self.counts.values()) == 0:
            return None
        return 1 if self.counts[1] >= self.counts[0] else 0

    def learn_one(self, x, y):
        self.counts[int(y)] += 1

    def get_params(self):
        return {"seed": 42, "backend": self.backend}

    def native_feature_importance(self):
        return {"x": 1.0}

    def close(self):
        return None


def main() -> None:
    df = pd.DataFrame({
        "x": list(range(60)),
        "group": ["A", "B"] * 30,
        "target": [0, 1] * 30,
    })
    cfg = RunConfig(
        target="target",
        sensitive_attribute="group",
        window_size=10,
        max_samples=60,
        time_budget_sec=30,
        fairness_min_group_n=2,
        track_sustainability=False,
    )
    with tempfile.TemporaryDirectory(prefix="awareml_phase2_") as td:
        store = ExperimentStore(td)
        result = _run_one(
            _ToyFramework(),
            df,
            cfg,
            experiment_store=store,
            experiment_id="phase2-validator",
            dataset_id="phase2-toy",
            protocol_version="meta-v2-phase2",
        )
        run_dir = Path(td) / "phase2-validator"
        required = [
            "run.json",
            "windows.jsonl",
            "fairness.jsonl",
            "explainability.jsonl",
            "sustainability.jsonl",
        ]
        missing = [x for x in required if not (run_dir / x).exists()]
        if missing:
            raise RuntimeError(f"Missing Phase 2 artifacts: {missing}")
        summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert result.status == "ok"
        assert summary["prequential_accuracy"] is not None
        assert summary["prequential_macro_f1"] is not None
        assert summary["mean_prediction_latency_ms"] is not None
        assert summary["p95_prediction_latency_ms"] is not None
        assert summary["throughput_samples_sec"] is not None
        assert summary["energy_kwh"] is None
        assert summary["co2_kg"] is None
        assert len(result.points) == 6
    print("AwareML Phase 2 streaming instrumentation validation: OK")


if __name__ == "__main__":
    main()
