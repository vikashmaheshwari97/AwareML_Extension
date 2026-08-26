from pathlib import Path

import pandas as pd
import pytest

from awareml.recommender.v2_data import (
    load_canonical_runs,
    load_recommender_train,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "meta" / "snapshots"


@pytest.fixture(scope="module")
def runs():
    path = SNAPSHOT_DIR / "meta_logs_v2.parquet"
    if not path.exists():
        pytest.skip(
            "Phase-6.1 canonical Parquet has not been built."
        )
    return load_canonical_runs(path)


@pytest.fixture(scope="module")
def train():
    path = (
        SNAPSHOT_DIR
        / "recommender_train_v2.parquet"
    )
    if not path.exists():
        pytest.skip(
            "Phase-6.1 recommender Parquet has not been built."
        )
    return load_recommender_train(path)


def test_canonical_run_matrix(runs):
    assert len(runs) == 705
    assert runs["dataset_id"].nunique() == 47
    assert (
        runs[
            ["dataset_id", "framework", "seed"]
        ]
        .drop_duplicates()
        .shape[0]
        == 705
    )


def test_recommender_seed_aggregation(train):
    assert len(train) == 235
    assert train["dataset_id"].nunique() == 47
    assert (
        train[
            ["dataset_id", "framework"]
        ]
        .drop_duplicates()
        .shape[0]
        == 235
    )
    assert set(
        train["seed_count"]
        .astype(int)
        .unique()
        .tolist()
    ) == {3}


def test_primary_targets_are_complete(train):
    for col in [
        "accuracy_mean",
        "runtime_sec_mean",
        "energy_kwh_mean",
        "co2_kg_mean",
        "accuracy_std",
        "runtime_sec_std",
        "energy_kwh_std",
        "co2_kg_std",
    ]:
        assert train[col].notna().all()


def test_runtime_energy_carbon_are_positive(train):
    for col in [
        "runtime_sec_mean",
        "energy_kwh_mean",
        "co2_kg_mean",
        "samples_processed_mean",
    ]:
        values = pd.to_numeric(
            train[col], errors="raise"
        )
        assert (values > 0).all()
