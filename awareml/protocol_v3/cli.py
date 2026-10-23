from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .common import ROOT, FRAMEWORKS, clean, environment, load_campaign, read_json, write_json


def parser():
    p = argparse.ArgumentParser(description="AwareML V3 reproducible streaming experiments and meta-recommender")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("status", help="Show software/environment and campaign/model readiness")
    q.add_argument("--campaign")
    q.add_argument("--bundle")
    q = sub.add_parser("prepare", help="Freeze stream prefixes, profiles, tasks and preferences without executing frameworks")
    q.add_argument("--campaign", required=True)
    q.add_argument("--config", default=str(ROOT / "configs/stream_protocol_v3.json"))
    q.add_argument("--cohort", choices=("dev", "eval", "both"), default="dev")
    q.add_argument("--include-derived", action="store_true")
    q.add_argument("--inventory", help="Optional CSV with dataset_id,path,target,cohort,task_policy")
    q.add_argument("--smoke", action="store_true")
    q = sub.add_parser("run", help="Run one task, or serial tasks, with an external safety watchdog")
    q.add_argument("--campaign", required=True)
    q.add_argument("--task-id", type=int)
    q.add_argument("--serial", action="store_true")
    q.add_argument("--resume", action="store_true", help="Skip any task with an existing attempt; never rerun failures silently")
    q.add_argument("--retry", action="store_true", help="Preserve a new attempt for the explicitly chosen task")
    q.add_argument("--dedicated-host", action="store_true", help="Declare that this workstation is dedicated to one measured workload")
    q = sub.add_parser("worker", help=argparse.SUPPRESS)
    q.add_argument("--campaign", required=True)
    q.add_argument("--task-id", type=int, required=True)
    q.add_argument("--attempt", required=True)
    q = sub.add_parser("collect", help="Validate and aggregate complete runs, retaining failures and all attempts")
    q.add_argument("--campaign", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--latest-attempt", action="store_true")
    q.add_argument("--portfolio", choices=("primary", "extended"), default="primary")
    q = sub.add_parser("train", help="Run nested LODO, final model selection and frozen V3 bundle creation")
    q.add_argument("--collection", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--portfolio", choices=("primary", "extended"), default="primary")
    q.add_argument("--features", choices=("prefix", "metadata", "both"), default="both")
    q.add_argument("--complete-case", action="store_true")
    q.add_argument("--cache", help="Optional shared fit-cache directory for interruption recovery into a new output bundle")
    q = sub.add_parser("evaluate", help="Evaluate nested development predictions or a separate evaluation cohort")
    q.add_argument("--bundle", required=True)
    q.add_argument("--collection", help="Omit for nested development evaluation")
    q.add_argument("--output", required=True)
    q.add_argument("--complete-case", action="store_true")
    q = sub.add_parser("ablation", help="Paired comparison of prefix-only and metadata evaluations")
    q.add_argument("--prefix", required=True)
    q.add_argument("--metadata", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("recommend", help="Use a trained prefix-only V3 bundle on a labeled CSV prefix")
    q.add_argument("--bundle", required=True)
    q.add_argument("--csv", required=True)
    q.add_argument("--target", required=True)
    q.add_argument("--weights", nargs=4, type=float, default=[1, 1, 1, 1], metavar=("ACC", "TIME", "ENERGY", "CO2"))
    q.add_argument("--output", required=True)
    q = sub.add_parser("audit-existing", help="Audit legacy artifacts; zero matching files produces a clear report")
    q.add_argument("--root", default=str(ROOT))
    q.add_argument("--output", required=True)
    q = sub.add_parser("submit", help="Submit exclusive-node Slurm array tasks; no fixed node names")
    q.add_argument("--campaign", required=True)
    q.add_argument("--concurrency", type=int, default=4)
    q.add_argument("--partition")
    q.add_argument("--constraint", help="Cluster hardware constraint to keep CPU hardware homogeneous")
    q.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("smoke", help="Unmeasured, explicitly ineligible ASML/AutoClass binary and multiclass integration checks")
    q.add_argument("--output", required=True)
    q.add_argument("--frameworks", nargs="+", choices=FRAMEWORKS, default=["AutoStreamML", "AutoClass"])
    return p


def smoke(output, frameworks=("AutoStreamML", "AutoClass")):
    import numpy as np
    import pandas as pd
    from .campaign import prepare
    from .execution import run_task
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.RandomState(42)
    rows = []
    for classes in (2, 3):
        path = output / ("stream_{}.csv".format(classes))
        x = rng.normal(size=160)
        y = (x > 0).astype(int) if classes == 2 else np.digitize(x, [-.4, .4])
        pd.DataFrame({"x": x, "z": rng.normal(size=160), "target": y}).to_csv(path, index=False)
        rows.append({"dataset_id": "smoke_{}classes".format(classes), "path": str(path), "target": "target",
                     "cohort": "dev", "task_policy": "native_classification"})
    pd.DataFrame(rows).to_csv(output / "inventory.csv", index=False)
    cfg = read_json(ROOT / "configs/stream_protocol_v3.json")
    cfg.update(frameworks=list(frameworks), seeds=[42], max_samples=160, profile_size=30,
               window_size=40, track_energy=False, safety_timeout_sec=120)
    write_json(output / "smoke_config.json", cfg)
    manifest = prepare(output / "campaign", output / "smoke_config.json", cohorts=("dev",), inventory_path=output / "inventory.csv", smoke=True)
    results = [run_task(output / "campaign", i) for i in range(manifest["frozen"]["task_count"])]
    report = {"status": "pass" if all(r["status"] == "completed" and r["samples_processed"] == 160 for r in results) else "fail",
              "production_evidence": False, "tested": [{"task": r["task"], "status": r["status"], "error": r.get("error"), "samples": r.get("samples_processed")} for r in results]}
    write_json(output / "smoke_report.json", report)
    return report


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.command == "status":
        result = {"software_version": "3.0.0", "environment": environment(), "old_frozen_results_are_not_V3_results": True}
        if a.campaign:
            m = load_campaign(a.campaign)
            result["campaign"] = {"mode": m["frozen"]["mode"], "tasks": m["frozen"]["task_count"], "sha256": m["campaign_sha256"]}
        if a.bundle:
            from .models import load_bundle
            m, _ = load_bundle(a.bundle)
            result["bundle"] = {"feature_set": m["feature_set"], "training_datasets": len(m["training_dataset_ids"])}
    elif a.command == "prepare":
        from .campaign import prepare
        cohorts = ("dev", "eval") if a.cohort == "both" else (a.cohort,)
        m = prepare(a.campaign, a.config, cohorts, a.include_derived, a.inventory, a.smoke)
        result = {"campaign": str(Path(a.campaign).resolve()), "sha256": m["campaign_sha256"],
                  "datasets": m["frozen"]["dataset_count"], "tasks": m["frozen"]["task_count"], "mode": m["frozen"]["mode"]}
    elif a.command == "worker":
        from .execution import worker
        result = worker(a.campaign, a.task_id, a.attempt)
    elif a.command == "run":
        if bool(a.serial) == (a.task_id is not None):
            p.error("Choose exactly one of --serial or --task-id.")
        if a.serial and a.retry:
            p.error("Retries must identify a single task, preserving an explicit retry decision.")
        from .execution import run_task
        if a.serial:
            m = load_campaign(a.campaign)
            counts = {"completed": 0, "failed": 0, "skipped_existing": 0}
            for i in range(m["frozen"]["task_count"]):
                previous = list((Path(a.campaign) / "runs" / "task_{:05d}".format(i)).glob("attempt_*"))
                if a.resume and previous:
                    counts["skipped_existing"] += 1
                    continue
                r = run_task(a.campaign, i, a.dedicated_host)
                key = "completed" if r["status"] == "completed" else "failed"
                counts[key] += 1
                print("Task {}: {}".format(i, r["status"]), flush=True)
            result = counts
        else:
            r = run_task(a.campaign, a.task_id, a.dedicated_host, a.retry)
            result = {k: r.get(k) for k in ("task", "status", "eligible", "stop_reason", "samples_processed", "error")}
    elif a.command == "collect":
        from .collection import collect
        m = collect(a.campaign, a.output, a.latest_attempt, a.portfolio)
        result = {k: m["frozen"][k] for k in ("expected_datasets", "complete_datasets", "excluded_incomplete_datasets", "task_status_counts", "hardware_issues")}
    elif a.command == "train":
        from .models import train_bundle
        kinds = (False, True) if a.features == "both" else (a.features == "metadata",)
        result = []
        for metadata in kinds:
            directory = Path(a.output) / (a.portfolio + ("_metadata" if metadata else "_prefix"))
            m = train_bundle(a.collection, directory, a.portfolio, metadata, a.complete_case, a.cache)
            result.append({"bundle": str(directory), "features": m["feature_set"], "training_datasets": len(m["training_dataset_ids"])})
    elif a.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(a.bundle, a.output, a.collection, a.complete_case)
    elif a.command == "ablation":
        from .evaluation import compare_ablation
        result = compare_ablation(a.prefix, a.metadata, a.output)
    elif a.command == "recommend":
        import pandas as pd
        from .service import V3Recommender
        service = V3Recommender(a.bundle)
        frame = pd.read_csv(a.csv, nrows=int(service.manifest["profile_size"]))
        ranked, meta = service.recommend_dataframe(frame, a.target, a.weights)
        result = {"ranking": ranked.to_dict("records"), "metadata": meta}
        write_json(a.output, result)
    elif a.command == "audit-existing":
        from .audit import audit_existing
        result = audit_existing(a.root, a.output)
    elif a.command == "submit":
        m = load_campaign(a.campaign)
        if a.concurrency < 1:
            p.error("Concurrency must be positive.")
        cfg = m["frozen"]["protocol"]
        log_dir = Path(a.campaign).resolve() / "slurm"
        log_dir.mkdir(parents=True, exist_ok=True)
        scheduler_seconds = int(cfg["safety_timeout_sec"]) + 600
        duration = "{:02d}:{:02d}:{:02d}".format(scheduler_seconds // 3600, scheduler_seconds // 60 % 60, scheduler_seconds % 60)
        command = ["sbatch", "--exclusive", "--mem=0", "--cpus-per-task={}".format(cfg["cpu_threads"]),
                   "--time=" + duration, "--array=0-{}%{}".format(m["frozen"]["task_count"] - 1, a.concurrency),
                   "--output=" + str(log_dir / "%A_%a.out"), "--error=" + str(log_dir / "%A_%a.err")]
        if a.partition:
            command.append("--partition=" + a.partition)
        if a.constraint:
            command.append("--constraint=" + a.constraint)
        command += [str(ROOT / "hpc/protocol_v3/run_task.sbatch"), str(Path(a.campaign).resolve()), sys.executable]
        result = {"command_argv": command, "submitted": False}
        if not a.dry_run:
            result.update(submitted=True, response=subprocess.check_output(command, cwd=str(ROOT), text=True).strip())
    elif a.command == "smoke":
        result = smoke(a.output, a.frameworks)
    print(json.dumps(clean(result), indent=2, allow_nan=False))
    if a.command == "smoke" and result["status"] != "pass":
        return 1
    if a.command == "run" and ((a.serial and result["failed"]) or (not a.serial and result.get("status") != "completed")):
        return 1
    return 0
