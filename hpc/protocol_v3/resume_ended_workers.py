#!/usr/bin/env python3
"""Resume only ended source shards while other source workers keep running.

Does not modify the frozen campaign or its original, hash-pinned launcher.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location("awareml_frozen_batch", HERE.with_name("batch_launcher.py"))
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def ended_sources(source, array, workers):
    """Require terminal Slurm evidence and a normal launcher summary."""
    if not re.fullmatch(r"[0-9]+", str(array)):
        raise ValueError("Source array must be a numeric Slurm array ID.")
    requested = {str(array) + "_" + str(i) for i in workers}
    output = subprocess.check_output([
        "sacct", "-X", "--noheader", "--parsable2", "--jobs=" + ",".join(sorted(requested)),
        "--format=JobID,State,ExitCode"], universal_newlines=True, timeout=30)
    records = {}
    for line in output.splitlines():
        parts = line.strip().split("|")
        if parts and parts[0] in requested:
            if parts[0] in records:
                raise RuntimeError("Ambiguous accounting records for " + parts[0])
            records[parts[0]] = parts[1:3]
    if set(records) != requested or any(v != ["COMPLETED", "0:0"] for v in records.values()):
        raise RuntimeError("Selected source workers must be COMPLETED with exit 0:0: " + str(records))
    hashes = {}
    for i in workers:
        path = Path(source) / ("shard_%d.summary.json" % i)
        data = batch.read(path)
        if data.get("state") not in ("allocation_boundary", "shard_complete"):
            raise RuntimeError("Review the source worker failure before resuming shard %d." % i)
        if any(item.get("state") not in ("completed", "skipped_completed") for item in data["outcomes"]):
            raise RuntimeError("Unsuccessful source outcome requires review.")
        hashes[path.name] = batch.file_hash(path)
    return hashes


def check_live(array, workers, nodes):
    output = subprocess.check_output([
        "squeue", "--me", "--array", "--noheader", "--format=%i|%j|%T"],
        universal_newlines=True, timeout=30)
    others = []
    for line in output.splitlines():
        job, name, state = (v.strip() for v in line.split("|", 2))
        source = re.fullmatch(re.escape(str(array)) + r"_([0-9]+)", job)
        if source:
            if int(source.group(1)) in workers:
                raise RuntimeError("Selected source worker is still live: " + job)
            others.append(job)
        elif name.startswith("awareml-v3"):
            raise RuntimeError("Unrecognized live V3 submission; cannot prove disjoint work: " + job)
    if len(others) + nodes > 4:
        raise RuntimeError("Keep at most four exclusive workers under the current CPU allowance.")
    return others


def make_plan(source, array, workers, nodes=3, hours=12, min_hours=4):
    old = batch.load_plan(source)
    workers = sorted(set(workers))
    if not workers or any(i < 0 or i >= len(old["shards"]) for i in workers):
        raise ValueError("Select valid ended source worker indices.")
    if not 1 <= nodes <= 4:
        raise ValueError("Use one to four new exclusive workers.")
    seconds, minimum = int(hours * 3600), int(min_hours * 3600)
    if not old["safety_timeout_sec"] + old["cleanup_seconds"] <= minimum <= seconds <= 8 * 86400:
        raise ValueError("Allocation bounds must preserve the full watchdog and cleanup interval.")
    summaries = ended_sources(source, array, workers)
    live = check_live(array, workers, nodes)
    reserved = {int(item["task"]["task_id"]) for i, shard in enumerate(old["shards"])
                if i not in workers for item in shard}
    work, skipped, review = [], [], []
    for i in workers:
        for item in old["shards"][i]:
            task = item["task"]
            task_id = int(task["task_id"])
            history = batch.history(old["campaign"], task_id)
            result = batch.latest(old["campaign"], task_id)
            if history and history[-1]["result_sha256"] is None:
                review.append(task_id)
            elif result and batch.eligible(result, task, old["campaign_sha256"]):
                skipped.append(task_id)
            elif history:
                review.append(task_id)
            else:
                work.append({"task": task, "history": [], "retry_authorized": False})
    if review:
        raise RuntimeError("Review unsuccessful/incomplete ended-worker tasks first: " + str(review))
    if not work:
        raise ValueError("No unstarted tasks remain in the selected ended source workers.")
    ids = [int(item["task"]["task_id"]) for item in work]
    if len(ids) != len(set(ids)) or reserved.intersection(ids):
        raise RuntimeError("Source plan has overlapping task assignments.")
    work.sort(key=lambda item: int(item["task"]["task_id"]))
    nodes = min(nodes, len(work))
    shards = [work[i::nodes] for i in range(nodes)]
    return {**old, "schema": "awareml_v3_resume_ended_workers_v1", "nodes": nodes,
            "allocation_seconds": seconds, "minimum_allocation_seconds": minimum,
            "source_plan": str(Path(source).resolve()), "source_array": str(array),
            "source_workers": workers, "source_plan_sha256": batch.file_hash(Path(source) / "plan.json"),
            "source_summary_hashes": summaries, "reserved_task_ids": sorted(reserved),
            "live_source_workers_at_prepare": live, "source_completed_skipped": sorted(skipped),
            "resume_launcher_sha256": batch.file_hash(HERE), "shards": shards}


def claim_path(plan, worker):
    return Path(plan["source_plan"]) / "resume_claims" / ("shard_%d.json" % worker)


def prepare(source, array, workers, nodes, hours, min_hours, output):
    plan = make_plan(source, array, workers, nodes, hours, min_hours)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    identity = {"resume_plan": str(output), "source_plan_sha256": plan["source_plan_sha256"]}
    created = []
    try:
        for worker in plan["source_workers"]:
            path = claim_path(plan, worker)
            path.parent.mkdir(exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                created.append(path)
                json.dump(identity, handle)
        batch.write(output / "plan.json", {"sha256": batch.digest(plan), "frozen": plan})
    except BaseException:
        for path in created:
            path.unlink()
        raise
    return plan


def load(directory):
    directory = Path(directory).resolve()
    payload = batch.read(directory / "plan.json")
    plan = payload["frozen"]
    if plan["schema"] != "awareml_v3_resume_ended_workers_v1" or batch.digest(plan) != payload["sha256"]:
        raise RuntimeError("Resume plan integrity failure.")
    if batch.file_hash(HERE) != plan["resume_launcher_sha256"]:
        raise RuntimeError("Resume launcher changed; do not run this frozen plan.")
    source = Path(plan["source_plan"])
    if batch.file_hash(source / "plan.json") != plan["source_plan_sha256"]:
        raise RuntimeError("Source plan changed.")
    batch.load_plan(source)  # Original launcher and campaign remain hash-pinned.
    identity = {"resume_plan": str(directory), "source_plan_sha256": plan["source_plan_sha256"]}
    for worker in plan["source_workers"]:
        if batch.read(claim_path(plan, worker)) != identity:
            raise RuntimeError("Ended source shard is reserved by another resume plan.")
    for name, expected in plan["source_summary_hashes"].items():
        if batch.file_hash(source / name) != expected:
            raise RuntimeError("Source worker summary changed.")
    ids = [int(item["task"]["task_id"]) for shard in plan["shards"] for item in shard]
    if len(ids) != len(set(ids)) or set(ids).intersection(plan["reserved_task_ids"]):
        raise RuntimeError("Resume plan overlaps a reserved worker.")
    return plan


def seconds_left():
    """Query the exact array element, including the array placeholder element."""
    array, index = os.environ["SLURM_ARRAY_JOB_ID"], os.environ["SLURM_ARRAY_TASK_ID"]
    job = array + "_" + index
    output = subprocess.check_output(["scontrol", "show", "job", "-o", job],
                                     universal_newlines=True, timeout=30)
    records = [line for line in output.splitlines() if line.strip()]
    if len(records) != 1:
        raise RuntimeError("Expected one exact Slurm array-element record for " + job)
    fields = dict(item.split("=", 1) for item in records[0].split() if "=" in item)
    if fields.get("ArrayJobId") != array or fields.get("ArrayTaskId") != index:
        raise RuntimeError("Scheduler returned a different array element.")
    if fields.get("OverSubscribe") != "NO" and fields.get("Shared") != "0":
        raise RuntimeError("Allocation is not exclusive.")
    return max(0, batch.parse_slurm_duration(fields["TimeLimit"]) - batch.parse_slurm_duration(fields["RunTime"]))


def command(plan, directory):
    directory = Path(directory).resolve()
    args = batch.submit_command(plan, directory)
    args[args.index("--job-name=awareml-v3-batch")] = "--job-name=awareml-v3-resume"
    args.insert(1, "--time-min=" + batch.duration(plan["minimum_allocation_seconds"]))
    args[-1] = "--wrap=exec " + shlex.join([sys.executable, str(HERE), "worker", "--plan", str(directory)])
    return args


def submit(directory, dry_run=False):
    directory = Path(directory).resolve()
    plan = load(directory)
    ended_sources(plan["source_plan"], plan["source_array"], plan["source_workers"])
    check_live(plan["source_array"], plan["source_workers"], plan["nodes"])
    args = command(plan, directory)
    print(shlex.join(args), flush=True)
    if dry_run:
        return
    marker = directory / "submission.json"
    with marker.open("x", encoding="utf-8") as handle:
        json.dump({"state": "submission_started_do_not_repeat", "command": args}, handle)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    job = subprocess.check_output(args, cwd=str(batch.ROOT), env=environment, universal_newlines=True).strip()
    batch.write(marker, {"state": "submitted", "job": job, "command": args})
    print("Submitted:", job, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("preview", "prepare"):
        q = sub.add_parser(name)
        q.add_argument("--source-plan", required=True)
        q.add_argument("--source-array", required=True)
        q.add_argument("--workers", nargs="+", type=int, required=True)
        q.add_argument("--nodes", type=int, default=3)
        q.add_argument("--hours", type=float, default=12)
        q.add_argument("--min-hours", type=float, default=4)
        if name == "prepare":
            q.add_argument("--output", required=True)
    q = sub.add_parser("submit")
    q.add_argument("--plan", required=True)
    q.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("worker")
    q.add_argument("--plan", required=True)
    a = p.parse_args(argv)
    if a.action in ("preview", "prepare"):
        args = (a.source_plan, a.source_array, a.workers, a.nodes, a.hours, a.min_hours)
        plan = make_plan(*args) if a.action == "preview" else prepare(*args, a.output)
        print(json.dumps({"source_workers": plan["source_workers"],
                          "completed_preserved_in_selected_workers": len(plan["source_completed_skipped"]),
                          "tasks_to_resume": sum(map(len, plan["shards"])),
                          "reserved_other_worker_tasks": len(plan["reserved_task_ids"]),
                          "new_workers": plan["nodes"], "jobs_submitted": False}, indent=2))
    elif a.action == "submit":
        submit(a.plan, a.dry_run)
    else:
        plan = load(a.plan)
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if not 0 <= index < len(plan["shards"]):
            raise ValueError("Invalid worker index.")
        with (Path(a.plan) / ("shard_%d.started" % index)).open("x", encoding="utf-8") as handle:
            json.dump({"job": os.environ["SLURM_JOB_ID"]}, handle)
        batch.STOP = False
        signal.signal(signal.SIGUSR1, batch.stop_signal)
        signal.signal(signal.SIGTERM, batch.stop_signal)
        remaining = seconds_left()
        print(json.dumps({"worker": index, "allocation_seconds_left": remaining,
                          "assigned_tasks": len(plan["shards"][index])}), flush=True)
        result = batch.run_shard(plan, plan["shards"][index], a.plan, time.monotonic() + remaining)
        batch.write(Path(a.plan) / ("shard_%d.summary.json" % index), result)
        print(json.dumps(result, indent=2), flush=True)
        return int(result["state"] == "stopped_on_failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
