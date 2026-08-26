from pathlib import Path

from awareml.experiments.legacy import audit_legacy_meta_logs
from awareml.experiments.registry import (
    load_dataset_manifest,
    load_dataset_registry,
    validate_train_test_separation,
)


def test_frozen_v1_train_split_and_registry():
    train = load_dataset_manifest("data/meta/manifests/train_v1_47.yaml")
    registry = load_dataset_registry("data/meta/registry/datasets_v1_47.yaml")
    assert train.frozen is True
    assert len(train.dataset_ids) == 47
    assert len(registry.datasets) == 47
    assert set(train.dataset_ids) == set(registry.by_id)


def test_heldout_manifest_is_explicitly_incomplete_not_invented():
    test = load_dataset_manifest("data/meta/manifests/test_v1_23.yaml", require_complete=False)
    assert test.expected_count == 23
    assert test.frozen is False
    assert test.dataset_ids == []


def test_train_test_separation_and_legacy_integrity():
    train = load_dataset_manifest("data/meta/manifests/train_v1_47.yaml")
    test = load_dataset_manifest("data/meta/manifests/test_v1_23.yaml", require_complete=False)
    registry = load_dataset_registry("data/meta/registry/datasets_v1_47.yaml")
    report = validate_train_test_separation(train, test, registry=registry)
    assert report["dataset_overlap"] == []

    audit = audit_legacy_meta_logs("data/meta/legacy/meta_logs_v1_47.json")
    assert audit["rows"] == 234
    assert audit["datasets"] == 47
    assert audit["frameworks"]["AutoStreamML"] == 47
    assert audit["frameworks"]["OAML"] == 46
    assert audit["zero_co2_placeholders"] > 0
