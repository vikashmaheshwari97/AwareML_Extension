from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase18_common import (
    EXPECTED_AGGREGATED, EXPECTED_DATASETS, EXPECTED_EXTENSION, EXPECTED_HISTORICAL,
    EXPECTED_PREFERENCE_CASES, EXPECTED_RUNS, FRAMEWORKS, MAX_SAMPLES,
    PREFERENCES_PER_DATASET, PROTOCOL_ID, SEEDS, TIME_BUDGET_SEC, WINDOW_SIZE,
    apply_task_policy, build_ordinal_quantile_policy, build_preference_manifest,
    build_task_manifest, json_safe, load_inventory, write_json_atomic,
)


def _synthetic_frozen_rows() -> pd.DataFrame:
    inventory = load_inventory().copy()
    inventory["sha256"] = ["{:064x}".format(i + 1) for i in range(len(inventory))]
    inventory["resolved_target"] = ["target_{}".format(i) for i in range(len(inventory))]
    inventory["evaluation_target"] = inventory.apply(
        lambda r: "__phase18_target__" if r["task_policy"] == "ordinal_quantile_max5" else r["resolved_target"], axis=1
    )
    inventory["task_transform_payload_json"] = inventory["task_policy"].map(
        lambda p: json.dumps({"transform_type":"ordinal_quantile_max5","thresholds":[1,2,3,4],"right":True,"effective_classes":5,"transform_sha256":"abc"}) if p == "ordinal_quantile_max5" else json.dumps({"transform_type":"none","effective_classes":2})
    )
    inventory["task_transform_sha256"] = inventory["task_policy"].map(lambda p: "abc" if p == "ordinal_quantile_max5" else "")
    inventory["positive_label_resolved_json"] = inventory["positive_label_json"]
    return inventory


def test_locked_counts():
    assert PROTOCOL_ID == "phase18_final_heldout_31_v1"
    assert EXPECTED_DATASETS == 31
    assert EXPECTED_HISTORICAL == 23
    assert EXPECTED_EXTENSION == 8
    assert tuple(SEEDS) == (42,43,44)
    assert len(FRAMEWORKS) == 5
    assert EXPECTED_RUNS == 465
    assert EXPECTED_AGGREGATED == 155
    assert EXPECTED_PREFERENCE_CASES == 3100
    assert PREFERENCES_PER_DATASET == 100
    assert MAX_SAMPLES == 30000 and WINDOW_SIZE == 1000 and TIME_BUDGET_SEC == 60.0


def test_inventory_31_excludes_adult_and_has_explicit_policy():
    inv = load_inventory()
    assert len(inv) == 31
    assert "adult_binary_stream" not in set(inv["dataset_id"])
    assert inv["dataset_id"].is_unique and inv["filename"].is_unique
    assert inv["cohort"].value_counts().to_dict() == {"historical_23":23, "extension_8":8}
    assert inv["task_policy"].value_counts().to_dict() == {"ordinal_quantile_max5":16, "native_classification":15}


def test_verified_weird_filenames_preserved():
    names=set(load_inventory()["filename"])
    assert "FriedmanGra_ Synthetic.csv" in names
    assert "FriedmanLea_ Synthetic.csv" in names
    assert "hyperA_ Synthetic.csv" in names
    assert "RBFm_100k__Synthetic.csv" in names


def test_ordinal_policy_is_deterministic_and_drops_source_target():
    df=pd.DataFrame({"x":range(20),"y":np.linspace(0,100,20)})
    policy=build_ordinal_quantile_policy(df["y"])
    assert policy["transform_type"] == "ordinal_quantile_max5"
    assert 2 <= policy["effective_classes"] <= 5
    a,target_a,payload_a=apply_task_policy(df,"y","ordinal_quantile_max5",policy)
    b,target_b,payload_b=apply_task_policy(df,"y","ordinal_quantile_max5",policy)
    pd.testing.assert_frame_equal(a,b)
    assert target_a == target_b == "__phase18_target__"
    assert "y" not in a.columns
    assert payload_a == payload_b


def test_preference_manifest_3100():
    frozen=_synthetic_frozen_rows()
    a=build_preference_manifest(frozen); b=build_preference_manifest(frozen)
    pd.testing.assert_frame_equal(a,b)
    assert len(a)==3100
    assert not a[["dataset_id","preference_id"]].duplicated().any()
    assert np.allclose(a[["w_accuracy","w_runtime","w_energy","w_co2"]].sum(axis=1),1.0)


def test_task_manifest_465_and_contains_transform_evidence():
    tasks=build_task_manifest(_synthetic_frozen_rows())
    assert len(tasks)==465
    assert tasks["task_id"].tolist()==list(range(465))
    assert not tasks[["dataset_id","framework","seed"]].duplicated().any()
    assert {"source_target","task_policy","task_transform_payload_json","task_transform_sha256"} <= set(tasks.columns)


def test_slurm_array_465_and_runner_applies_policy():
    sb=(ROOT/"hpc/production/phase18/phase18_array.sbatch").read_text()
    runner=(ROOT/"hpc/production/phase18/run_phase18_task.py").read_text()
    assert "#SBATCH --array=0-464%8" in sb
    assert "apply_task_policy" in runner
    assert "verify_frozen_checksums()" in runner
    assert "allow-chacha-fallback" in runner


def test_evaluator_outputs_3100_cases_without_fit():
    source=(ROOT/"scripts/phase18_evaluate_recommender.py").read_text()
    assert "phase18_preference_eval_primary_3100.csv" in source
    assert "phase18_preference_eval_cikm_compat_3100.csv" in source
    assert ".fit(" not in source


def test_strict_json_writer(tmp_path):
    payload={"nan":float("nan"),"inf":float("inf"),"na":pd.NA,"nested":[np.float64(np.nan)]}
    clean=json_safe(payload)
    assert clean["nan"] is None and clean["inf"] is None and clean["na"] is None
    out=tmp_path/"x.json"; write_json_atomic(out,payload)
    text=out.read_text(); assert "NaN" not in text and "Infinity" not in text


def test_readme_states_derived_tasks_are_not_original_regression_targets():
    text=(ROOT/"PHASE18_V2_5_31_DATASET_PROTOCOL.md").read_text(encoding="utf-8")
    assert "derived classification tasks" in text
    assert "not the original regression targets" in text
    assert "adult_binary_stream" in text
