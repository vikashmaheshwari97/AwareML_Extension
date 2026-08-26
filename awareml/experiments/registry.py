from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml

from .records import DatasetDescriptor


@dataclass
class DatasetManifest:
    manifest_id: str
    purpose: str
    expected_count: int
    dataset_ids: List[str]
    frozen: bool
    schema_version: str = "1.0"
    notes: str = ""

    def validate(self, require_complete: bool = True) -> "DatasetManifest":
        if self.purpose not in {"meta_train", "heldout_test", "development", "audit"}:
            raise ValueError(f"Unsupported manifest purpose: {self.purpose!r}")
        if len(self.dataset_ids) != len(set(self.dataset_ids)):
            raise ValueError(f"Manifest {self.manifest_id} contains duplicate dataset ids.")
        if require_complete and len(self.dataset_ids) != int(self.expected_count):
            raise ValueError(
                f"Manifest {self.manifest_id} expects {self.expected_count} datasets but contains {len(self.dataset_ids)}."
            )
        if self.frozen and len(self.dataset_ids) != int(self.expected_count):
            raise ValueError("A frozen manifest must be complete.")
        return self


@dataclass
class DatasetRegistry:
    registry_id: str
    datasets: List[DatasetDescriptor]
    schema_version: str = "1.0"
    notes: str = ""

    def validate(self) -> "DatasetRegistry":
        ids = [d.dataset_id for d in self.datasets]
        if len(ids) != len(set(ids)):
            raise ValueError("Dataset registry contains duplicate dataset_id values.")
        for d in self.datasets:
            d.validate()
        return self

    @property
    def by_id(self) -> Dict[str, DatasetDescriptor]:
        return {d.dataset_id: d for d in self.datasets}


def _load_yaml(path: str) -> Dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file {path!r} must contain a mapping.")
    return payload


def load_dataset_manifest(path: str, require_complete: bool = True) -> DatasetManifest:
    p = _load_yaml(path)
    obj = DatasetManifest(
        manifest_id=str(p.get("manifest_id", "")),
        purpose=str(p.get("purpose", "")),
        expected_count=int(p.get("expected_count", 0)),
        dataset_ids=[str(x) for x in (p.get("dataset_ids") or [])],
        frozen=bool(p.get("frozen", False)),
        schema_version=str(p.get("schema_version", "1.0")),
        notes=str(p.get("notes", "") or ""),
    )
    return obj.validate(require_complete=require_complete)


def load_dataset_registry(path: str) -> DatasetRegistry:
    p = _load_yaml(path)
    rows = []
    for item in p.get("datasets") or []:
        rows.append(DatasetDescriptor(
            dataset_id=str(item.get("dataset_id", "")),
            file_name=str(item.get("file_name", "")),
            family=str(item.get("family", "")),
            source_type=str(item.get("source_type", "unknown")),
            is_synthetic=item.get("is_synthetic"),
            generator=item.get("generator"),
            drift_type=item.get("drift_type"),
            target=item.get("target"),
            sensitive_attributes=list(item.get("sensitive_attributes") or []),
            fingerprint_sha256=item.get("fingerprint_sha256"),
            notes=item.get("notes"),
            family_inference=item.get("family_inference"),
        ))
    return DatasetRegistry(
        registry_id=str(p.get("registry_id", "")),
        datasets=rows,
        schema_version=str(p.get("schema_version", "1.0")),
        notes=str(p.get("notes", "") or ""),
    ).validate()


def validate_train_test_separation(
    train: DatasetManifest,
    test: DatasetManifest,
    registry: Optional[DatasetRegistry] = None,
    enforce_family_separation: bool = False,
) -> Dict[str, Any]:
    train_ids, test_ids = set(train.dataset_ids), set(test.dataset_ids)
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise ValueError("Train/test dataset overlap detected: " + ", ".join(overlap))

    result: Dict[str, Any] = {
        "dataset_overlap": [],
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "family_overlap": [],
    }
    if registry is not None:
        known = registry.by_id
        missing = sorted((train_ids | test_ids) - set(known))
        if missing:
            raise ValueError("Manifest references datasets missing from registry: " + ", ".join(missing))
        train_families = {known[x].family for x in train_ids}
        test_families = {known[x].family for x in test_ids}
        family_overlap = sorted(train_families & test_families)
        result["family_overlap"] = family_overlap
        if enforce_family_separation and family_overlap:
            raise ValueError("Dataset-family leakage detected: " + ", ".join(family_overlap))
    return result


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
