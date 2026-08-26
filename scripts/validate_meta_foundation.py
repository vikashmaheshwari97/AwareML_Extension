from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.experiments.legacy import audit_legacy_meta_logs
from awareml.experiments.registry import load_dataset_manifest, load_dataset_registry, validate_train_test_separation


def main() -> None:
    train = load_dataset_manifest(str(ROOT / "data/meta/manifests/train_v1_47.yaml"), require_complete=True)
    test = load_dataset_manifest(str(ROOT / "data/meta/manifests/test_v1_23.yaml"), require_complete=False)
    registry = load_dataset_registry(str(ROOT / "data/meta/registry/datasets_v1_47.yaml"))
    assert len(train.dataset_ids) == 47
    assert len(registry.datasets) == 47
    validate_train_test_separation(train, test, registry=registry)
    audit = audit_legacy_meta_logs(str(ROOT / "data/meta/legacy/meta_logs_v1_47.json"))
    assert audit["rows"] == 234
    assert audit["datasets"] == 47
    print("AwareML Meta-Experiment Foundation validation: OK")


if __name__ == "__main__":
    main()
