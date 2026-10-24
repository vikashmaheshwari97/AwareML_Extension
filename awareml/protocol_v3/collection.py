from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .common import (METADATA, NUMERIC_FEATURES, TARGETS, digest, load_campaign,
                     read_json, sha256, utc_now, write_json)

METRICS = ("accuracy", "f1_macro", "runtime_sec", "energy_kwh", "co2_kg", "peak_process_tree_rss_mib",
           "initialization_sec", "streaming_sec", "post_profile_accuracy", "post_profile_f1_macro")


def validate_result(result, task, campaign_sha):
    reasons = []
    if result.get("campaign_sha256") != campaign_sha:
        reasons.append("campaign_hash_mismatch")
    actual = result.get("task") or {}
    for key in ("dataset_id", "framework", "seed", "task_id", "expected_samples"):
        if str(actual.get(key)) != str(task.get(key)):
            reasons.append("task_mismatch_" + key)
    if result.get("status") != "completed" or not result.get("eligible"):
        reasons.append("not_an_eligible_completed_production_run")
    if int(result.get("samples_processed") or 0) != int(task["expected_samples"]):
        reasons.append("incomplete_stream")
    for key in ("accuracy", "runtime_sec", "energy_kwh", "co2_kg", "peak_process_tree_rss_mib"):
        value = result.get(key)
        if value is None or not np.isfinite(float(value)) or float(value) < 0:
            reasons.append("invalid_" + key)
    if result.get("accuracy") is not None and not 0 <= float(result["accuracy"]) <= 1:
        reasons.append("accuracy_out_of_range")
    if result.get("energy_kwh") is not None and float(result["energy_kwh"]) <= 0:
        reasons.append("nonpositive_energy")
    if result.get("peak_process_tree_rss_mib") is not None and float(result["peak_process_tree_rss_mib"]) <= 0:
        reasons.append("nonpositive_memory")
    if result.get("energy_status") != "measured" or result.get("tracking_mode") != "machine":
        reasons.append("unverified_energy_measurement")
    if (result.get("isolation") or {}).get("kind") not in ("slurm_exclusive", "user_declared_dedicated_host"):
        reasons.append("missing_isolation_provenance")
    return reasons


def collect(campaign_dir, output, latest_attempt=False, portfolio="primary"):
    campaign_dir, output = Path(campaign_dir).resolve(), Path(output).resolve()
    manifest = load_campaign(campaign_dir)
    if output.exists():
        raise ValueError("Collection directory exists; use a new path to preserve evidence.")
    tasks = pd.read_csv(campaign_dir / "tasks.csv")
    portfolio_key = "primary_portfolio" if portfolio == "primary" else "extended_portfolio"
    selected_frameworks = manifest["frozen"]["protocol"][portfolio_key]
    if set(selected_frameworks) - set(tasks.framework):
        raise ValueError("The requested portfolio was not included in this campaign.")
    selected, statuses, attempts = [], [], []
    selected_files = {}
    for task in tasks.to_dict("records"):
        paths = sorted((campaign_dir / "runs" / "task_{:05d}".format(int(task["task_id"]))).glob("attempt_*/result.json"))
        for path in paths:
            r = read_json(path)
            attempts.append({**task, "attempt": r.get("attempt_number"), "status": r.get("status"),
                             "eligible_flag": r.get("eligible"), "result_path": str(path.relative_to(campaign_dir)),
                             "sha256": sha256(path), "error": r.get("error")})
        if not paths:
            statuses.append({**task, "status": "missing", "eligible": False, "issues": "not_run"})
            continue
        path = paths[-1] if latest_attempt else paths[0]
        r = read_json(path)
        issues = validate_result(r, task, manifest["campaign_sha256"])
        statuses.append({**task, "status": r.get("status"), "eligible": not issues, "issues": ";".join(issues)})
        selected_files[str(path.relative_to(campaign_dir)).replace("\\", "/")] = sha256(path)
        if not issues:
            row = {**task, **{key: r.get(key) for key in METRICS},
                   "cpu_model": r["environment"]["cpu_model"],
                   "thread_signature": digest(r["environment"]["thread_limits"]),
                   "node": r["environment"]["node"], "backend": r.get("backend")}
            selected.append(row)
    status_frame = pd.DataFrame(statuses)
    complete_ids = []
    portfolio_status = status_frame.loc[status_frame.framework.isin(selected_frameworks)]
    for dataset_id, part in portfolio_status.groupby("dataset_id", sort=True):
        if bool(part.eligible.all()):
            complete_ids.append(dataset_id)
    frame = pd.DataFrame(selected)
    hardware_issues = []
    measurement_hardware = []
    if not frame.empty:
        frame = frame.loc[frame.framework.isin(selected_frameworks)]
        signatures = frame[["cpu_model", "thread_signature"]].drop_duplicates()
        measurement_hardware = signatures.to_dict("records")
        if len(signatures) != 1:
            hardware_issues.append("mixed_cpu_model_or_thread_allocation_in_campaign")
        frame = frame.loc[frame.dataset_id.isin(complete_ids)].copy()
    aggregates = []
    if not frame.empty:
        for (dataset_id, framework), part in frame.groupby(["dataset_id", "framework"], sort=True):
            row = {"dataset_id": dataset_id, "framework": framework, "cohort": part.iloc[0].cohort,
                   "seed_count": len(part), "expected_samples": int(part.iloc[0].expected_samples)}
            for key in METRICS:
                values = pd.to_numeric(part[key], errors="coerce")
                row[key + "_mean"] = float(values.mean()) if values.notna().any() else None
                row[key + "_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else None
            aggregates.append(row)
    profiles = pd.read_csv(campaign_dir / "profiles.csv", keep_default_na=False)
    aggregated = pd.DataFrame(aggregates)
    if not aggregated.empty:
        aggregated = aggregated.merge(profiles.drop(columns=["cohort"]), on="dataset_id", validate="many_to_one")
    output.mkdir(parents=True)
    status_frame.to_csv(output / "task_status.csv", index=False)
    pd.DataFrame(attempts).to_csv(output / "all_attempts.csv", index=False)
    aggregated.to_csv(output / "aggregate.csv", index=False)
    profiles.to_csv(output / "profiles.csv", index=False)
    task_status_counts = status_frame.status.value_counts().to_dict()
    frozen = {"schema": "awareml_v3_collection", "campaign_directory": str(campaign_dir),
              "portfolio": portfolio, "frameworks": selected_frameworks,
              "campaign_sha256": manifest["campaign_sha256"], "campaign_file_sha256": sha256(campaign_dir / "campaign.json"),
              "protocol": manifest["frozen"]["protocol"], "mode": manifest["frozen"]["mode"],
              "expected_datasets": manifest["frozen"]["dataset_count"], "complete_datasets": complete_ids,
              "excluded_incomplete_datasets": sorted(set(tasks.dataset_id) - set(complete_ids)),
              "primary_attempt_policy": "latest_explicitly_requested" if latest_attempt else "first_attempt",
              "task_status_counts": task_status_counts, "hardware_issues": hardware_issues,
              "measurement_hardware": measurement_hardware,
              "created_utc": utc_now(), "selected_attempt_hashes": selected_files,
              "dataset_provenance": read_json(campaign_dir / "datasets.json"),
              "file_hashes": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    result = {"collection_sha256": digest(frozen), "frozen": frozen}
    write_json(output / "collection.json", result)
    return result


def load_collection(directory, allow_complete_case=False):
    directory = Path(directory)
    payload = read_json(directory / "collection.json")
    frozen = payload["frozen"]
    if payload["collection_sha256"] != digest(frozen):
        raise ValueError("Collection manifest was modified.")
    for name, expected in frozen["file_hashes"].items():
        if sha256(directory / name) != expected:
            raise ValueError("Collected file changed: {}".format(name))
    if frozen["mode"] != "production":
        raise ValueError("Smoke data cannot train or evaluate the research model.")
    if frozen["hardware_issues"]:
        raise ValueError("Resolve hardware inconsistency before model training: {}".format(frozen["hardware_issues"]))
    if frozen["excluded_incomplete_datasets"] and not allow_complete_case:
        raise ValueError("Incomplete datasets remain. Inspect failures, or explicitly select --complete-case and report exclusions.")
    if not frozen["complete_datasets"]:
        raise ValueError("No complete production datasets are available.")
    frame = pd.read_csv(directory / "aggregate.csv", keep_default_na=False)
    expected_seeds = len(frozen["protocol"]["seeds"])
    if set(frame.seed_count.astype(int)) != {expected_seeds}:
        raise ValueError("Every training row must contain all declared seeds.")
    return frame, payload
