from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.data import make_drift_stream, profile_dataset
from awareml.engine.pareto import epsilon_pareto_mask
from awareml.recommender.schema import ObjectiveRequest
from awareml.experiments.registry import load_dataset_manifest, load_dataset_registry


def main():
    df = make_drift_stream(600)
    profile = profile_dataset(df, "target", "synthetic-smoke")
    assert profile.n_features == 5
    assert ObjectiveRequest().normalized_weights()["accuracy"] > 0
    train = load_dataset_manifest(str(ROOT / "data/meta/manifests/train_v1_47.yaml"))
    registry = load_dataset_registry(str(ROOT / "data/meta/registry/datasets_v1_47.yaml"))
    assert len(train.dataset_ids) == 47
    assert len(registry.datasets) == 47
    print("AwareML Extension structure validation: OK")


if __name__ == "__main__":
    main()
