from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .common import (ROOT, FRAMEWORKS, METADATA, SEEDS, canonical_csv_sha, code_fingerprint,
                     digest, environment, read_json, sha256, utc_now, write_json)
from .profile import ordinal_thresholds, profile_csv


def source_inventory(root=ROOT):
    root = Path(root)
    registry = yaml.safe_load((root / "data/meta/registry/datasets_v1_47_audited.yaml").read_text())
    rows = []
    for item in registry["datasets"]:
        rows.append({"dataset_id": item["dataset_id"], "path": str(root / "data/stream_datasets" / item["file_name"]),
                     "target": item["target"], "cohort": "dev", "task_policy": "native_classification",
                     "dataset_family": item.get("family") or "unknown", "source_type": item.get("source_type") or "unknown",
                     "drift_type": item.get("drift_type") or "unknown", "sensitive_attribute": ""})
    inventory = pd.read_csv(root / "configs/phase18_dataset_inventory_31.csv", keep_default_na=False)
    for item in inventory.to_dict("records"):
        rows.append({"dataset_id": item["dataset_id"], "path": str(root / "data/Testing Datasets" / item["filename"]),
                     "target": item["target"], "cohort": "eval", "task_policy": item["task_policy"],
                     **{key: item.get(key) or "unknown" for key in METADATA},
                     "sensitive_attribute": item.get("sensitive_attribute") or ""})
    return rows


def preference_bank(count=100, seed=42):
    rng = np.random.RandomState(int(seed))
    rows = []
    for i in range(int(count)):
        k = int(rng.randint(1, 5))
        active = rng.choice(4, size=k, replace=False)
        values = np.zeros(4)
        values[active] = rng.dirichlet(np.ones(k))
        rows.append({"preference_id": i, **dict(zip(("accuracy", "runtime", "energy", "co2"), values.tolist()))})
    return rows


def prepare(directory, config_path, cohorts=("dev", "eval"), include_derived=False, inventory_path=None, smoke=False):
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("Campaign directory already exists; choose a new name to preserve frozen evidence.")
    cfg = read_json(config_path)
    frameworks = list(cfg["frameworks"])
    if not frameworks or len(set(frameworks)) != len(frameworks) or set(frameworks) - set(FRAMEWORKS):
        raise ValueError("Choose unique supported framework IDs.")
    seeds = [int(v) for v in cfg["seeds"]]
    if len(set(seeds)) != len(seeds) or (not smoke and len(seeds) < 5):
        raise ValueError("Production campaigns need at least five distinct seeds.")
    if cfg.get("scientific_time_limit_sec") is not None:
        raise ValueError("V3 scientific_time_limit_sec must be null.")
    cap = cfg.get("max_samples")
    if cap is not None and int(cap) < 1:
        raise ValueError("Use null for full streams, or a positive common max_samples.")
    if int(cfg["profile_size"]) < 1 or float(cfg["safety_timeout_sec"]) <= 0 or int(cfg["window_size"]) < 1:
        raise ValueError("Prefix, window and safety timeout must be positive.")
    if float(cfg["memory_poll_sec"]) <= 0 or int(cfg["cpu_threads"]) < 1 or int(cfg["preferences_per_dataset"]) < 1:
        raise ValueError("Memory sampling, thread count and preference count must be positive.")
    if cfg["tracking_mode"] != "machine":
        raise ValueError("This isolated campaign explicitly implements machine-mode measurement.")
    if not smoke and not cfg.get("track_energy", True):
        raise ValueError("Production campaigns require measured energy; use --smoke for unmeasured checks.")
    rows = pd.read_csv(inventory_path, keep_default_na=False).to_dict("records") if inventory_path else source_inventory()
    if len({r["dataset_id"] for r in rows}) != len(rows):
        raise ValueError("Inventory dataset IDs must be globally unique.")
    selected, excluded = [], []
    for r in rows:
        if r["cohort"] not in cohorts:
            continue
        if r.get("task_policy", "native_classification") != "native_classification" and not include_derived:
            excluded.append({"dataset_id": r["dataset_id"], "reason": "derived_task_not_in_primary_cohort"})
            continue
        selected.append(r)
    if not selected:
        raise ValueError("No datasets selected.")
    missing = [r["path"] for r in selected if not Path(r["path"]).is_file()]
    if missing:
        raise FileNotFoundError("{} source CSVs are missing. First missing: {}. Use --cohort dev to prepare development alone.".format(len(missing), missing[0]))
    source_hashes, seen = {}, {}
    for r in selected:
        value = canonical_csv_sha(r["path"])
        if value in seen:
            raise ValueError("Duplicate stream content: {} and {}".format(seen[value], r["dataset_id"]))
        seen[value] = r["dataset_id"]
        source_hashes[r["dataset_id"]] = value
    # Even a separately prepared evaluation cohort must not duplicate a dev source.
    if "eval" in cohorts:
        for r in rows:
            if r["cohort"] == "dev" and r["dataset_id"] not in source_hashes and Path(r["path"]).is_file():
                value = canonical_csv_sha(r["path"])
                if value in seen:
                    raise ValueError("Evaluation source duplicates development stream {}".format(r["dataset_id"]))
    directory.mkdir(parents=True)
    streams = directory / "streams"
    streams.mkdir()
    datasets, profiles, tasks = [], [], []
    for r in selected:
        identifier = str(r["dataset_id"])
        frame = pd.read_csv(r["path"], nrows=int(cap) if cap is not None else None, low_memory=False)
        if frame.empty or frame.columns.duplicated().any():
            raise ValueError("Empty dataset or duplicate columns: {}".format(identifier))
        target = str(frame.columns[-1]) if r["target"] == "__LAST__" else str(r["target"])
        if target not in frame or frame[target].isna().any():
            raise ValueError("Missing target/labels: {}".format(identifier))
        policy = r.get("task_policy") or "native_classification"
        transform = None
        if policy == "ordinal_quantile_max5":
            # The thresholds are learned on the first labeled prefix only.
            prefix = pd.read_csv(r["path"], nrows=int(cfg["profile_size"]))
            thresholds = ordinal_thresholds(prefix[target])
            original_target = target
            target = "__awareml_v3_class__"
            if target in frame:
                raise ValueError("Reserved derived target column already exists.")
            labels = np.digitize(pd.to_numeric(frame[original_target], errors="raise"), thresholds, right=True)
            frame = frame.drop(columns=[original_target])
            frame[target] = labels
            transform = {"kind": "prefix_quantile_classification", "source_target": original_target,
                         "thresholds": thresholds, "right": True, "fitted_on_first": min(int(cfg["profile_size"]), len(frame))}
        elif policy != "native_classification":
            raise ValueError("Unsupported task policy: {}".format(policy))
        sensitive = str(r.get("sensitive_attribute") or "")
        if sensitive and sensitive != target and sensitive in frame:
            frame = frame.drop(columns=[sensitive])
        rel = "streams/{}.csv".format(digest(identifier)[:16])
        path = directory / rel
        frame.to_csv(path, index=False, lineterminator="\n")
        profile = profile_csv(path, target, int(cfg["profile_size"]))
        meta = {key: str(r.get(key) or "unknown") for key in METADATA}
        profiles.append({"dataset_id": identifier, "cohort": r["cohort"], **profile, **meta})
        info = {"dataset_id": identifier, "cohort": r["cohort"], "stream_path": rel, "stream_sha256": sha256(path),
                "source_sha256": sha256(r["path"]), "source_canonical_sha256": source_hashes[identifier],
                "target": target, "expected_samples": len(frame), "task_policy": policy,
                "target_transform": transform, "sensitive_feature_policy": "audit_only", **meta}
        datasets.append(info)
        for fw, seed in itertools.product(frameworks, seeds):
            tasks.append({"task_id": len(tasks), "dataset_id": identifier, "framework": fw, "seed": seed,
                          "cohort": r["cohort"], "expected_samples": len(frame)})
    pd.DataFrame(profiles).to_csv(directory / "profiles.csv", index=False)
    pd.DataFrame(tasks).to_csv(directory / "tasks.csv", index=False)
    pd.DataFrame(preference_bank(cfg["preferences_per_dataset"], cfg["preference_seed"])).to_csv(directory / "preferences.csv", index=False)
    write_json(directory / "datasets.json", datasets)
    file_hashes = {str(p.relative_to(directory)).replace("\\", "/"): sha256(p)
                   for p in sorted(directory.rglob("*")) if p.is_file()}
    frozen = {"protocol": cfg, "mode": "smoke" if smoke else "production", "created_utc": utc_now(),
              "cohorts": list(cohorts), "included_derived": bool(include_derived), "exclusions": excluded,
              "dataset_count": len(datasets), "task_count": len(tasks), "file_hashes": file_hashes,
              "code": code_fingerprint(), "preparation_environment": environment(),
              "evidence_role": "revised_external_evaluation_reuses_previously_evaluated_cohort"}
    manifest = {"campaign_sha256": digest(frozen), "frozen": frozen}
    write_json(directory / "campaign.json", manifest)
    return manifest
