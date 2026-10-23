from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from awareml.protocol_v3 import campaign, collection, evaluation, execution, models
from awareml.protocol_v3.audit import audit_existing
from awareml.protocol_v3.common import (ROOT, NUMERIC_FEATURES, FRAMEWORKS, PRIMARY, digest,
                                        load_campaign, read_json, sha256, utilities, write_json)
from awareml.protocol_v3.profile import ordinal_thresholds, prefix_profile, profile_csv


def sources(tmp_path, count=4, cohort="dev", offset=0):
    inventory = []
    for i in range(count):
        rng = np.random.RandomState(100 + offset + i)
        path = tmp_path / ("source_{}_{}.csv".format(cohort, i))
        pd.DataFrame({"x": rng.normal(size=45), "z": rng.normal(size=45), "target": rng.randint(0, 2, 45)}).to_csv(path, index=False)
        inventory.append({"dataset_id": cohort + str(i), "path": str(path), "target": "target", "cohort": cohort,
                          "task_policy": "native_classification", "dataset_family": "test_family",
                          "source_type": "unit_test", "drift_type": "unknown"})
    path = tmp_path / ("inventory_" + cohort + ".csv")
    pd.DataFrame(inventory).to_csv(path, index=False)
    return path


def config(tmp_path, frameworks=PRIMARY):
    cfg = read_json(ROOT / "configs/stream_protocol_v3.json")
    cfg.update(frameworks=list(frameworks), primary_portfolio=list(PRIMARY), extended_portfolio=list(FRAMEWORKS),
               max_samples=40, profile_size=10, window_size=10,
               candidate_models=["FrameworkMean", "Ridge"])
    path = tmp_path / "config.json"
    write_json(path, cfg)
    return path


def prepared(tmp_path, cohort="dev", count=4, frameworks=PRIMARY):
    inv = sources(tmp_path, count, cohort)
    cfg = config(tmp_path, frameworks)
    path = tmp_path / ("campaign_" + cohort)
    campaign.prepare(path, cfg, (cohort,), inventory_path=inv)
    return path


def synthetic_measured_result(task, campaign_hash, hw="unit_test_cpu", missing=False):
    # Test fixtures only. No fixture is published as experimental evidence.
    d = int(str(task["dataset_id"])[-1])
    f = list(FRAMEWORKS).index(task["framework"])
    result = {"task": task, "campaign_sha256": campaign_hash, "status": "completed", "eligible": True,
              "samples_processed": int(task["expected_samples"]) - int(missing),
              "accuracy": .50 + .03 * d + .02 * f, "f1_macro": .49 + .03 * d + .02 * f,
              "runtime_sec": 2 + d + f, "energy_kwh": .0001 * (2 + d + f),
              "co2_kg": .00004 * (2 + d + f), "peak_process_tree_rss_mib": 20 + f,
              "energy_status": "measured", "tracking_mode": "machine", "attempt_number": 1,
              "isolation": {"kind": "user_declared_dedicated_host"},
              "environment": {"cpu_model": hw, "node": "test_only", "thread_limits": {"OMP_NUM_THREADS": "2"}},
              "backend": "test_fixture_only", "initialization_sec": .01, "streaming_sec": 1.0,
              "post_profile_accuracy": .5, "post_profile_f1_macro": .5}
    return result


def populate_test_attempts(path, bad_framework=None, hw_change=False):
    m = read_json(path / "campaign.json")
    tasks = pd.read_csv(path / "tasks.csv").to_dict("records")
    for i, task in enumerate(tasks):
        r = synthetic_measured_result(task, m["campaign_sha256"], hw="other_test_cpu" if hw_change and i == 0 else "unit_test_cpu")
        if task["framework"] == bad_framework:
            r.update(status="failed", eligible=False)
        write_json(path / "runs" / ("task_{:05d}".format(int(task["task_id"]))) / "attempt_001/result.json", r)


def test_profile_prefix_invariance_and_no_registry_features():
    a = pd.DataFrame({"x": [1., np.nan, 3., 4.], "y": [0, 1, 0, 1]})
    b = pd.DataFrame({"x": [1., np.nan] + [99.] * 20, "y": [0, 1] + [7] * 20})
    left = prefix_profile(a, "y", 2)
    assert left == prefix_profile(b, "y", 2)
    assert set(left) == set(NUMERIC_FEATURES)
    assert left["missing_fraction"] == .5


def test_profile_csv_inferrs_types_without_suffix(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text("x,y\n1,0\n2,1\ncategory,1\n")
    assert profile_csv(path, "y", 2)["n_numeric_features"] == 1


@pytest.mark.parametrize("frame,target,size", [(pd.DataFrame(), "y", 10), (pd.DataFrame({"x": [1]}), "y", 1),
                                                 (pd.DataFrame({"x": [1], "y": [None]}), "y", 1),
                                                 (pd.DataFrame({"x": [1], "y": [0]}), "y", 0)])
def test_invalid_prefix_rejected(frame, target, size):
    with pytest.raises(ValueError):
        prefix_profile(frame, target, size)


def test_ordinal_thresholds_use_only_explicit_prefix():
    a = pd.Series(range(10))
    assert ordinal_thresholds(a) == ordinal_thresholds(pd.concat([a, pd.Series([100000])]).iloc[:10])
    with pytest.raises(ValueError):
        ordinal_thresholds(pd.Series([1, 1, 1]))


def test_preparation_freezes_common_cap_seeds_and_profiles(tmp_path):
    path = prepared(tmp_path)
    m = load_campaign(path)
    tasks = pd.read_csv(path / "tasks.csv")
    assert len(tasks) == 4 * 4 * 5
    assert set(tasks.expected_samples) == {40}
    assert set(tasks.seed) == {42, 43, 44, 45, 46}
    assert set(pd.read_csv(path / "profiles.csv").profile_sample_size) == {10}
    assert m["frozen"]["protocol"]["scientific_time_limit_sec"] is None
    with pytest.raises(ValueError):
        campaign.prepare(path, tmp_path / "config.json", ("dev",), inventory_path=tmp_path / "inventory_dev.csv")


def test_modified_manifest_or_stream_rejected(tmp_path):
    path = prepared(tmp_path)
    data = read_json(path / "campaign.json")
    data["frozen"]["protocol"]["max_samples"] = 999
    write_json(path / "campaign.json", data)
    with pytest.raises(ValueError):
        load_campaign(path)


def test_duplicate_dataset_content_rejected(tmp_path):
    inv = sources(tmp_path, 2)
    rows = pd.read_csv(inv)
    rows.loc[1, "path"] = rows.loc[0, "path"]
    rows.to_csv(inv, index=False)
    with pytest.raises(ValueError, match="Duplicate stream"):
        campaign.prepare(tmp_path / "new", config(tmp_path), ("dev",), inventory_path=inv)


def test_derived_target_thresholds_and_source_drop(tmp_path):
    inv = sources(tmp_path, 1)
    rows = pd.read_csv(inv)
    source = Path(rows.iloc[0].path)
    df = pd.read_csv(source)
    df.target = np.arange(len(df))
    df.to_csv(source, index=False)
    rows["task_policy"] = "ordinal_quantile_max5"
    rows.to_csv(inv, index=False)
    cfg = config(tmp_path)
    path = tmp_path / "derived"
    campaign.prepare(path, cfg, ("dev",), include_derived=True, inventory_path=inv)
    d = read_json(path / "datasets.json")[0]
    assert d["target_transform"]["thresholds"] == ordinal_thresholds(pd.Series(np.arange(10)))
    frame = pd.read_csv(path / d["stream_path"])
    assert "target" not in frame and "__awareml_v3_class__" in frame


def test_incomplete_results_never_eligible():
    task = {"dataset_id": "dev0", "framework": "AutoClass", "seed": 42, "task_id": 0, "expected_samples": 40}
    result = synthetic_measured_result(task, "hash", missing=True)
    assert "incomplete_stream" in collection.validate_result(result, task, "hash")


def test_smoke_collection_rejected_for_training(tmp_path):
    path = prepared(tmp_path)
    m = read_json(path / "campaign.json")
    m["frozen"]["mode"] = "smoke"
    m["campaign_sha256"] = digest(m["frozen"])
    write_json(path / "campaign.json", m)
    populate_test_attempts(path)
    collection.collect(path, tmp_path / "collected")
    with pytest.raises(ValueError, match="Smoke"):
        collection.load_collection(tmp_path / "collected")


def test_collection_mean_sd_full_seeds_and_failure_visibility(tmp_path):
    path = prepared(tmp_path)
    populate_test_attempts(path)
    col = tmp_path / "collected"
    collection.collect(path, col)
    frame, m = collection.load_collection(col)
    assert len(frame) == 16 and set(frame.seed_count) == {5}
    assert set(frame.accuracy_std) == {0.0}
    assert m["frozen"]["excluded_incomplete_datasets"] == []


def test_optional_chacha_failure_does_not_block_primary(tmp_path):
    path = prepared(tmp_path, frameworks=FRAMEWORKS)
    populate_test_attempts(path, bad_framework="ChaCha")
    collection.collect(path, tmp_path / "primary", portfolio="primary")
    frame, _ = collection.load_collection(tmp_path / "primary")
    assert len(frame) == 16
    collection.collect(path, tmp_path / "extended", portfolio="extended")
    with pytest.raises(ValueError):
        collection.load_collection(tmp_path / "extended")


def test_mixed_hardware_cannot_enter_primary_training(tmp_path):
    path = prepared(tmp_path)
    populate_test_attempts(path, hw_change=True)
    collection.collect(path, tmp_path / "collected")
    with pytest.raises(ValueError, match="hardware"):
        collection.load_collection(tmp_path / "collected")


def test_first_attempt_not_replaced_by_later_success(tmp_path):
    path = prepared(tmp_path)
    populate_test_attempts(path)
    first = path / "runs/task_00000/attempt_001/result.json"
    r = read_json(first)
    write_json(first.parent.parent / "attempt_002/result.json", {**r, "attempt_number": 2})
    write_json(first, {**r, "status": "failed", "eligible": False})
    collection.collect(path, tmp_path / "collected")
    with pytest.raises(ValueError, match="Incomplete"):
        collection.load_collection(tmp_path / "collected")
    collection.collect(path, tmp_path / "explicit_latest", latest_attempt=True)
    frame, _ = collection.load_collection(tmp_path / "explicit_latest")
    assert frame.dataset_id.nunique() == 4


def test_metadata_model_switch_is_real(tmp_path):
    path = prepared(tmp_path)
    populate_test_attempts(path)
    collection.collect(path, tmp_path / "collected")
    frame, _ = collection.load_collection(tmp_path / "collected")
    model = models.fit_model(frame, "accuracy", "Ridge", False, 42)
    # Absence of all metadata fields must not affect the deployable estimator.
    assert np.isfinite(models.predict(model, frame.drop(columns=["dataset_family", "source_type", "drift_type"]), "accuracy", False)).all()
    model2 = models.fit_model(frame, "accuracy", "Ridge", True, 42)
    assert set(models.feature_columns(True)) - set(models.feature_columns(False)) == {"dataset_family", "source_type", "drift_type"}


def test_nested_selection_never_sees_outer_dataset(tmp_path):
    path = prepared(tmp_path)
    populate_test_attempts(path)
    collection.collect(path, tmp_path / "collected")
    frame, _ = collection.load_collection(tmp_path / "collected")
    predictions, decisions = models.nested_predictions(frame, ["FrameworkMean", "Ridge"])
    assert len(predictions) == len(frame)
    assert all(r["outer_dataset"] not in r["selection_dataset_ids"] for r in decisions)
    changed = frame.copy()
    changed.loc[changed.dataset_id.eq("dev0"), "accuracy_mean"] = [.01, .99, .02, .98]
    _, new_decisions = models.nested_predictions(changed, ["FrameworkMean", "Ridge"])
    assert [r["selected_family"] for r in decisions if r["outer_dataset"] == "dev0"] == [r["selected_family"] for r in new_decisions if r["outer_dataset"] == "dev0"]


def test_closed_gap_zero_denominator_and_dataset_inference():
    rows = []
    for d in range(3):
        for p in range(100):
            for name in ("AwareML_V3", "SBS", "VBS", "Random"):
                rows.append(dict(dataset_id=str(d), preference_id=p, method=name, selected_utility=1., oracle_utility=1., sbs_utility=1.,
                                 top1_exact=1., top1_tie_aware=1., top3=1., normalized_regret=0., spearman=1.))
    per_dataset, summary, pairs = evaluation.recommender_statistics(pd.DataFrame(rows))
    assert summary.closed_gap.isna().all()
    assert set(pairs.n_datasets) == {3}
    assert len(per_dataset) == 12


def test_utility_direction_and_invalid_weights():
    values = [[.9, 1, .001, .0004], [.5, 3, .003, .0012]]
    assert np.allclose(utilities(values, [1, 1, 1, 1]), [1, 0])
    with pytest.raises(ValueError):
        utilities(values, [0, 0, 0, 0])


def test_legacy_audit_handles_zero_files(tmp_path):
    result = audit_existing(tmp_path, tmp_path / "audit")
    assert result["status"] == "no_matching_legacy_results" and result["records"] == 0


def test_watchdog_kills_child_process_tree(tmp_path):
    import psutil
    pid_file = tmp_path / "child.pid"
    program = "import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    proc = subprocess.Popen([sys.executable, "-c", program, str(pid_file)], start_new_session=(sys.platform != "win32"))
    for _ in range(100):
        if pid_file.exists():
            break
        time.sleep(.02)
    assert pid_file.exists()
    child_pid = int(pid_file.read_text())
    execution.kill_tree(proc)
    assert proc.poll() is not None
    try:
        child = psutil.Process(child_pid)
        assert child.status() == psutil.STATUS_ZOMBIE or not child.is_running()
    except psutil.NoSuchProcess:
        pass


def test_framework_missing_predictions_share_denominator():
    confusion = execution.Confusion()
    confusion.update(0, None)
    confusion.update(1, 1)
    assert confusion.total == 2 and confusion.accuracy == .5 and confusion.missing == 1


@pytest.mark.integration
def test_bundle_training_evaluation_and_prediction_roundtrip(tmp_path):
    from awareml.protocol_v3.service import V3Recommender
    path = prepared(tmp_path)
    populate_test_attempts(path)
    collection.collect(path, tmp_path / "collected")
    bundle = tmp_path / "bundle"
    manifest = models.train_bundle(tmp_path / "collected", bundle)
    assert manifest["feature_set"] == "prefix" and len(manifest["training_dataset_ids"]) == 4
    result = evaluation.evaluate(bundle, tmp_path / "evaluation")
    assert result["n_preference_cases_per_method"] == 16
    assert set(pd.read_csv(tmp_path / "evaluation/paired_wilcoxon_holm.csv").n_datasets) == {4}
    service = V3Recommender(bundle)
    frame = pd.DataFrame({"x": np.arange(15), "z": np.arange(15), "target": np.arange(15) % 2})
    ranking, meta = service.recommend_dataframe(frame, "target")
    assert len(ranking) == 4 and meta["profile"]["profile_sample_size"] == 10
    assert ranking.utility.is_monotonic_decreasing
    with pytest.raises(ValueError, match="V2"):
        service.recommend_profile({"n_samples_dataset": 20000})
    # A genuinely separate test cohort exercises frozen prediction, all
    # baselines, dataset-level tests and native/derived report separation.
    external = tmp_path / "external_sources"
    external.mkdir()
    inv = sources(external, count=3, cohort="eval", offset=20)
    cfg = config(external)
    evaluation_campaign = tmp_path / "campaign_eval"
    campaign.prepare(evaluation_campaign, cfg, ("eval",), inventory_path=inv)
    populate_test_attempts(evaluation_campaign)
    collection.collect(evaluation_campaign, tmp_path / "collected_eval")
    heldout = evaluation.evaluate(bundle, tmp_path / "heldout", tmp_path / "collected_eval")
    assert heldout["n_datasets"] == 3 and heldout["n_preference_cases_per_method"] == 300
    assert heldout["task_policy_strata"] == [{"cohort": "native_classification", "n_datasets": 3}]
    assert set(pd.read_csv(tmp_path / "heldout/paired_wilcoxon_holm.csv").n_datasets) == {3}
    metadata_bundle = tmp_path / "metadata_bundle"
    models.train_bundle(tmp_path / "collected", metadata_bundle, metadata=True)
    evaluation.evaluate(metadata_bundle, tmp_path / "heldout_metadata", tmp_path / "collected_eval")
    differences = evaluation.compare_ablation(tmp_path / "heldout", tmp_path / "heldout_metadata", tmp_path / "ablation.csv")
    assert all(r["n_datasets"] == 3 for r in differences)
    bad_collection = read_json(tmp_path / "collected_eval/collection.json")
    bad_collection["frozen"]["measurement_hardware"][0]["cpu_model"] = "different_test_cpu"
    bad_collection["collection_sha256"] = digest(bad_collection["frozen"])
    write_json(tmp_path / "collected_eval/collection.json", bad_collection)
    with pytest.raises(ValueError, match="different CPU"):
        evaluation.evaluate(bundle, tmp_path / "wrong_hardware", tmp_path / "collected_eval")


def test_watchdog_failure_never_exposes_partial_targets(tmp_path, monkeypatch):
    # This unmeasured software test owns a private lock directory, so a real
    # measured workload or another test invocation retains its host-wide lock.
    lock_directory = tmp_path / "locks"
    lock_directory.mkdir()
    monkeypatch.setattr(execution.tempfile, "gettempdir", lambda: str(lock_directory))
    inv = sources(tmp_path, 1)
    cfg_path = config(tmp_path)
    cfg = read_json(cfg_path)
    cfg.update(seeds=[42], track_energy=False, safety_timeout_sec=.001)
    write_json(cfg_path, cfg)
    path = tmp_path / "short_watchdog"
    campaign.prepare(path, cfg_path, ("dev",), inventory_path=inv, smoke=True)
    result = execution.run_task(path, 0)
    assert result["status"] == "timeout" and result["eligible"] is False
    assert result.get("accuracy") is None and result.get("energy_kwh") is None


def test_backend_fallback_and_suppressed_errors_rejected():
    class Backend:
        name = "AutoClass"
        backend = "native"
    with pytest.raises(RuntimeError, match="Fallback"):
        execution.backend_check(Backend(), {"fallback_used": True})
    with pytest.raises(RuntimeError, match="Suppressed"):
        execution.backend_check(Backend(), {"research_error_count": 1})


def test_host_lock_excludes_second_workload_and_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(execution.tempfile, "gettempdir", lambda: str(tmp_path))
    with execution.host_lock():
        with pytest.raises(RuntimeError, match="Another V3 workload"):
            with execution.host_lock():
                pytest.fail("Second workload entered")
    with execution.host_lock():
        assert (tmp_path / "awareml-v3-measured-workload.lock").exists()
    assert not (tmp_path / "awareml-v3-measured-workload.lock").exists()
