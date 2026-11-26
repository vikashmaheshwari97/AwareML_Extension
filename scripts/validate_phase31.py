from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.engine.metrics import DriftRecoveryTracker, PredictionDiagnosticsTracker
from awareml.experiments.provenance import build_dataset_provenance
from awareml.types import RunConfig


def main() -> None:
    # 1) Drift episode must stay internally consistent after assessment closes.
    tracker = DriftRecoveryTracker(tolerance=0.02, min_assessment_samples=10)
    tracker.on_drift(100, 0.80, 0.80)
    tracker.update(110, 0.795)
    tracker.update(150, 0.60)
    ep = tracker.summary()["episodes"][0]
    assert ep["degradation_observed"] is False
    assert ep["accuracy_drop"] <= 0.02 + 1e-12

    # 2) Prediction degeneracy safeguard.
    pdg = PredictionDiagnosticsTracker(near_constant_threshold=0.95)
    for _ in range(96):
        pdg.update(0)
    for _ in range(4):
        pdg.update(1)
    assert pdg.summary()["near_constant_prediction"] is True

    # 3) Dataset provenance has schema + target/sensitive distributions + digest.
    df = pd.DataFrame({"x": [1, 2, 3], "sex": ["F", "M", "F"], "target": [0, 1, 0]})
    prov = build_dataset_provenance(df, target="target", sensitive_attribute="sex")
    assert len(prov["dataframe_sha256"]) == 64
    assert prov["target_distribution"]["n_unique"] == 2
    assert prov["sensitive_distribution"]["n_unique"] == 2

    # 4) New journal defaults.
    cfg = RunConfig(target="target")
    assert cfg.xai_replay_warning_threshold == 0.05
    assert cfg.prediction_near_constant_threshold == 0.95

    required = [
        ROOT / "scripts" / "validate_autoclass_adult_multiseed.py",
        ROOT / "scripts" / "validate_gradual_drift_phase31.py",
    ]
    assert all(p.exists() for p in required)

    print("AwareML Phase 3.1 hotfix validation: OK")
    print("- drift episode consistency: OK")
    print("- prediction degeneracy diagnostics: OK")
    print("- dataset provenance contract: OK")
    print("- XAI replay threshold default: 0.05")
    print("Next: run the full-Adult AutoClass multi-seed audit and one gradual-drift validation before Phase 4 production HPC runs.")


if __name__ == "__main__":
    main()
