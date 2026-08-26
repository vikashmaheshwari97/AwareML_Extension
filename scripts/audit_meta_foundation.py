from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.experiments.legacy import audit_legacy_meta_logs
from awareml.experiments.registry import (
    load_dataset_manifest,
    load_dataset_registry,
    validate_train_test_separation,
)


TRAIN = ROOT / "data/meta/manifests/train_v1_47.yaml"
TEST = ROOT / "data/meta/manifests/test_v1_23.yaml"
REGISTRY = ROOT / "data/meta/registry/datasets_v1_47.yaml"
LEGACY = ROOT / "data/meta/legacy/meta_logs_v1_47.json"


def main() -> None:
    train = load_dataset_manifest(str(TRAIN), require_complete=True)
    test = load_dataset_manifest(str(TEST), require_complete=False)
    registry = load_dataset_registry(str(REGISTRY))
    separation = validate_train_test_separation(train, test, registry=registry, enforce_family_separation=False)
    audit = audit_legacy_meta_logs(str(LEGACY))

    print("=== AwareML Meta-Experiment Foundation ===")
    print(f"Frozen train manifest : {train.manifest_id} ({len(train.dataset_ids)} datasets)")
    print(f"Held-out test manifest: {test.manifest_id} ({len(test.dataset_ids)}/{test.expected_count} registered)")
    print(f"Dataset registry      : {registry.registry_id} ({len(registry.datasets)} datasets)")
    print(f"Legacy meta rows      : {audit['rows']}")
    print(f"Framework coverage    : {audit['frameworks']}")
    print(f"Legacy zero-CO2 rows  : {audit['zero_co2_placeholders']} (treated as missing in V2)")
    print(f"Train/test overlap    : {len(separation['dataset_overlap'])}")
    print(f"Legacy SHA256         : {audit['sha256']}")
    if len(test.dataset_ids) != test.expected_count:
        print("NOTE: held-out v1 names were not present in the available source snapshot; fill/freeze them before journal evaluation.")


if __name__ == "__main__":
    main()
