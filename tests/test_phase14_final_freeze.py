from __future__ import annotations

from pathlib import Path

import pandas as pd

from awareml.analysis.repeatability import summarize_repeatability
from awareml.analysis.repeatability_registry import (
    PAPER_READY_MIN_REPETITIONS,
    build_dataset_identity,
    canonical_dataframe_sha256,
    find_latest_matching_run,
    register_run,
)


def test_paper_ready_gate_is_five():
    assert PAPER_READY_MIN_REPETITIONS == 5

    three = [
        {
            "framework": "AutoClass",
            "runtime_sec": float(i + 1),
            "energy_kwh": 0.001 + i * 0.0001,
            "co2_kg": 0.0004 + i * 0.00001,
        }
        for i in range(3)
    ]
    table_three = summarize_repeatability(three)
    assert table_three.iloc[0]["Repeatability gate"] == "NEEDS REPETITIONS"

    five = three + [
        {
            "framework": "AutoClass",
            "runtime_sec": 4.0,
            "energy_kwh": 0.0014,
            "co2_kg": 0.00044,
        },
        {
            "framework": "AutoClass",
            "runtime_sec": 5.0,
            "energy_kwh": 0.0015,
            "co2_kg": 0.00045,
        },
    ]
    table_five = summarize_repeatability(five)
    assert table_five.iloc[0]["Repeatability gate"] == "PASS"


def test_identity_uses_hash_target_sensitive_and_positive_label():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "sex": [0, 1, 1],
        "target": [0, 1, 0],
    })
    content_hash = canonical_dataframe_sha256(df)

    base = build_dataset_identity(
        dataset_name="sample.csv",
        dataset_content_sha256=content_hash,
        target="target",
        sensitive_attribute="sex",
        positive_label=1,
    )
    other_positive = build_dataset_identity(
        dataset_name="sample.csv",
        dataset_content_sha256=content_hash,
        target="target",
        sensitive_attribute="sex",
        positive_label=0,
    )
    other_sensitive = build_dataset_identity(
        dataset_name="sample.csv",
        dataset_content_sha256=content_hash,
        target="target",
        sensitive_attribute=None,
        positive_label=1,
    )
    other_hash = build_dataset_identity(
        dataset_name="sample.csv",
        dataset_content_sha256="f" * 64,
        target="target",
        sensitive_attribute="sex",
        positive_label=1,
    )

    assert base["identity_key"] != other_positive["identity_key"]
    assert base["identity_key"] != other_sensitive["identity_key"]
    assert base["identity_key"] != other_hash["identity_key"]


def test_registry_preserves_multiple_runs_and_returns_latest(tmp_path):
    content_hash = "a" * 64
    identity = build_dataset_identity(
        dataset_name="dataset.csv",
        dataset_content_sha256=content_hash,
        target="y",
        sensitive_attribute="group",
        positive_label=1,
    )

    run1 = tmp_path / identity["directory_name"] / "20260101T000000Z"
    run2 = tmp_path / identity["directory_name"] / "20260102T000000Z"
    run1.mkdir(parents=True)
    run2.mkdir(parents=True)

    for path in (run1, run2):
        (path / "phase14_repeated_results.json").write_text(
            "[]", encoding="utf-8"
        )
        (path / "repeatability_manifest.json").write_text(
            "{}", encoding="utf-8"
        )

    register_run(
        tmp_path,
        identity=identity,
        run_dir=run1,
        manifest={
            "created_utc": "2026-01-01T00:00:00+00:00",
            "repetitions": 5,
            "paper_ready": True,
        },
    )
    register_run(
        tmp_path,
        identity=identity,
        run_dir=run2,
        manifest={
            "created_utc": "2026-01-02T00:00:00+00:00",
            "repetitions": 5,
            "paper_ready": True,
        },
    )

    match = find_latest_matching_run(
        tmp_path,
        dataset_content_sha256=content_hash,
        target="y",
        sensitive_attribute="group",
        positive_label=1,
    )

    assert match is not None
    assert match["run_dir_path"].name == "20260102T000000Z"
    assert run1.exists()
    assert run2.exists()


def test_fairness_winner_excludes_degenerate_predictor_source():
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "awareml" / "ui_v2" / "pages_specialist.py"
    ).read_text(encoding="utf-8")

    assert "Lowest interpretable mean disparity" in source
    assert "EXCLUDED · degenerate predictions" in source
    assert '"constant", "near_constant"' in source
