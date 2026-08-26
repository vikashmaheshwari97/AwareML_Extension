from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.recommender.v2_data import (
    EXPECTED_FRAMEWORKS,
    load_canonical_runs,
    load_recommender_train,
)


SNAPSHOT_DIR = ROOT / "data" / "meta" / "snapshots"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(
            lambda: fh.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def verify_checksum(path: Path) -> None:
    checksum_path = Path(
        str(path) + ".sha256"
    )
    if not checksum_path.exists():
        raise RuntimeError(
            "Missing checksum: {}".format(
                checksum_path
            )
        )

    expected = (
        checksum_path
        .read_text(encoding="utf-8")
        .strip()
        .split()[0]
    )
    actual = sha256_file(path)

    if expected != actual:
        raise RuntimeError(
            "Checksum mismatch for {}: "
            "expected {}, actual {}".format(
                path.name,
                expected,
                actual,
            )
        )


def main() -> None:
    meta_path = (
        SNAPSHOT_DIR / "meta_logs_v2.parquet"
    )
    train_path = (
        SNAPSHOT_DIR
        / "recommender_train_v2.parquet"
    )
    schema_path = (
        SNAPSHOT_DIR
        / "meta_logs_v2_schema_report.json"
    )
    train_schema_path = (
        SNAPSHOT_DIR
        / "recommender_train_v2_schema.json"
    )

    for path in [
        meta_path,
        train_path,
        schema_path,
        train_schema_path,
    ]:
        if not path.exists():
            raise RuntimeError(
                "Missing Phase-6.1 artifact: {}".format(
                    path
                )
            )
        verify_checksum(path)

    runs = load_canonical_runs(meta_path)
    train = load_recommender_train(train_path)

    framework_counts = (
        train["framework"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    target_columns = [
        "accuracy_mean",
        "runtime_sec_mean",
        "energy_kwh_mean",
        "co2_kg_mean",
    ]
    uncertainty_columns = [
        "accuracy_std",
        "runtime_sec_std",
        "energy_kwh_std",
        "co2_kg_std",
    ]

    for col in (
        target_columns
        + uncertainty_columns
    ):
        if train[col].isna().any():
            raise RuntimeError(
                "{} contains null values.".format(
                    col
                )
            )

    if not (
        (
            train["accuracy_mean"] >= 0
        )
        & (
            train["accuracy_mean"] <= 1
        )
    ).all():
        raise RuntimeError(
            "accuracy_mean must be in [0,1]."
        )

    for col in [
        "runtime_sec_mean",
        "energy_kwh_mean",
        "co2_kg_mean",
        "samples_processed_mean",
    ]:
        values = pd.to_numeric(
            train[col], errors="coerce"
        )
        if (values <= 0).any():
            raise RuntimeError(
                "{} contains non-positive values.".format(
                    col
                )
            )

    schema = json.loads(
        schema_path.read_text(
            encoding="utf-8"
        )
    )
    train_schema = json.loads(
        train_schema_path.read_text(
            encoding="utf-8"
        )
    )

    print("=" * 72)
    print(
        "AwareML Phase 6.1 validation: PASS"
    )
    print("=" * 72)
    print(
        "Canonical runs:",
        len(runs),
    )
    print(
        "Unique run keys:",
        len(
            runs[
                [
                    "dataset_id",
                    "framework",
                    "seed",
                ]
            ].drop_duplicates()
        ),
    )
    print(
        "Recommender rows:",
        len(train),
    )
    print(
        "Datasets:",
        train["dataset_id"].nunique(),
    )
    print(
        "Framework counts:",
        framework_counts,
    )
    print(
        "Seed counts per aggregate:",
        sorted(
            train["seed_count"]
            .astype(int)
            .unique()
            .tolist()
        ),
    )
    print(
        "Canonical columns:",
        schema.get(
            "flattened_column_count"
        ),
    )
    print(
        "Primary objectives:",
        train_schema.get(
            "primary_objectives"
        ),
    )
    print(
        "All checksums: PASS"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
