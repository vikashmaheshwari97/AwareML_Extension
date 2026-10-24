#!/usr/bin/env python3
"""Persistent exclusive-node batches; the frozen V3 executor remains unchanged."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STOP = False


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def frozen_campaign(campaign):
    # Import only here: preview and all unit tests otherwise use the standard library.
    sys.path.insert(0, str(ROOT))
    from awareml.protocol_v3.common import load_campaign
    return load_campaign(campaign)


def task_rows(campaign):
    with (Path(campaign) / "tasks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [int(row["task_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Task inventory is empty or contains duplicate IDs.")
    return rows


def history(campaign, task_id):
    base = Path(campaign) / "runs" / ("task_%05d" % task_id)
    # Include incomplete attempts and their result presence; progress updates are ignored.
    return [{"attempt": p.name, "result_sha256": file_hash(p / "result.json")
             if (p / "result.json").exists() else None}
            for p in sorted(base.glob("attempt_*")) if p.is_dir()]


def eligible(result, task, campaign_sha):
    actual = result.get("task") or {}
    if result.get("campaign_sha256") != campaign_sha:
        return False
    if any(str(actual.get(k)) != str(task[k]) for k in
           ("task_id", "dataset_id", "framework", "seed", "expected_samples")):
        return False
    if result.get("status") != "completed" or result.get("eligible") is not True:
        return False
    if result.get("samples_processed") != int(task["expected_samples"]):
        return False
    for key in ("accuracy", "runtime_sec", "energy_kwh", "co2_kg", "peak_process_tree_rss_mib"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            return False
    return (result["accuracy"] <= 1 and result["energy_kwh"] > 0
            and result["peak_process_tree_rss_mib"] > 0
            and result.get("energy_status") == "measured"
            and result.get("tracking_mode") == "machine"
            and (result.get("isolation") or {}).get("kind") == "slurm_exclusive")


def latest(campaign, task_id):
    paths = sorted((Path(campaign) / "runs" / ("task_%05d" % task_id)).glob("attempt_*/result.json"))
    return read(paths[-1]) if paths else None


def inventory(campaign, manifest):
    complete, fresh, review = [], [], []
    for task in task_rows(campaign):
        task_id = int(task["task_id"])
        result = latest(campaign, task_id)
        if result and eligible(result, task, manifest["campaign_sha256"]):
            complete.append(task_id)
        elif history(campaign, task_id):
            review.append(task_id)
        else:
            fresh.append(task_id)
    return {"completed": complete, "fresh": fresh, "needs_review": review}


def active_jobs(old_arrays):
    output = subprocess.check_output(
        ["squeue", "--me", "--noheader", "--format=%F|%j"], universal_newlines=True, timeout=30)
    active = set()
    for line in output.splitlines():
        array, name = line.strip().split("|", 1)
        if array in old_arrays or name.startswith("awareml-v3"):
            active.add(array)
    return sorted(active)


def assert_drained(old_arrays):
    active = active_jobs(old_arrays)
    if active:
        raise RuntimeError("Existing V3 jobs still queued/running/completing: " + ", ".join(active))


def build_plan(campaign, manifest, nodes, hours, old_arrays, retries):
    cfg = manifest["frozen"]["protocol"]
    if not 1 <= nodes <= 4:
        raise ValueError("Use 1-4 exclusive nodes for the current 600-CPU association limit.")
    allocation = int(hours * 3600)
    if allocation < int(cfg["safety_timeout_sec"]) + 600:
        raise ValueError("Allocation must accommodate the full watchdog plus 600 seconds.")
    if allocation > 8 * 86400:
        raise ValueError("Allocation exceeds the main partition maximum.")
    inv = inventory(campaign, manifest)
    retries = set(retries)
    if retries - set(inv["needs_review"]):
        raise ValueError("Retry IDs must identify reviewed unsuccessful/incomplete tasks.")
    if set(inv["needs_review"]) - retries:
        raise ValueError("Review existing unsuccessful/incomplete attempts first: " + str(inv["needs_review"]))
    rows = {int(t["task_id"]): t for t in task_rows(campaign)}
    ids = sorted(inv["fresh"] + list(retries))
    if not ids:
        raise ValueError("No unfinished tasks to schedule.")
    completed_results = [latest(campaign, i) for i in inv["completed"]]
    cpus = {r["environment"]["cpu_model"] for r in completed_results}
    if len(cpus) != 1 or "EPYC 7763" not in next(iter(cpus)):
        raise ValueError("Existing eligible results must establish a single Artemis EPYC 7763 baseline.")
    budgets = {str(cfg["cpu_threads"])}
    for result in completed_results:
        limits = result["environment"].get("thread_limits", {})
        if any(str(limits.get(k)) not in budgets for k in
               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
            raise ValueError("Preserved results have a different thread budget.")
        if result.get("measurement_provenance", {}).get("gpu_energy_kwh") != 0:
            raise ValueError("Preserved results must have zero GPU energy.")
    ram = [r["measurement_provenance"]["host_ram_gib"] for r in completed_results]
    shards = [[] for _ in range(min(nodes, len(ids)))]
    for index, task_id in enumerate(ids):
        shards[index % len(shards)].append({"task": rows[task_id], "history": history(campaign, task_id),
                                           "retry_authorized": task_id in retries})
    return {"schema": "awareml_v3_exclusive_batch_v1", "campaign": str(Path(campaign).resolve()),
            "campaign_sha256": manifest["campaign_sha256"], "nodes": len(shards),
            "allocation_seconds": allocation, "cleanup_seconds": 600,
            "safety_timeout_sec": int(cfg["safety_timeout_sec"]), "cpu_threads": int(cfg["cpu_threads"]),
            "expected_cpu_model": next(iter(cpus)), "old_arrays": list(old_arrays),
            "expected_host_ram_gib": statistics.median(ram),
            "preserved_completed": inv["completed"], "shards": shards,
            "launcher_sha256": file_hash(__file__),
            "created_utc": datetime.now(timezone.utc).isoformat()}


def load_plan(directory):
    payload = read(Path(directory) / "plan.json")
    plan = payload["frozen"]
    if digest(plan) != payload["sha256"] or file_hash(__file__) != plan["launcher_sha256"]:
        raise ValueError("Batch plan or launcher changed; prepare a new plan.")
    manifest = frozen_campaign(plan["campaign"])
    if manifest["campaign_sha256"] != plan["campaign_sha256"]:
        raise ValueError("Campaign differs from batch plan.")
    return plan


def duration(seconds):
    return "%d:%02d:%02d" % (seconds // 3600, seconds // 60 % 60, seconds % 60)


def submit_command(plan, directory):
    directory = Path(directory).resolve()
    return ["sbatch", "--parsable", "--job-name=awareml-v3-batch", "--partition=main",
            "--nodes=1", "--ntasks=1", "--exclusive", "--mem=0",
            "--cpus-per-task=" + str(plan["cpu_threads"]),
            "--time=" + duration(plan["allocation_seconds"]),
            "--array=0-%d%%%d" % (plan["nodes"] - 1, plan["nodes"]),
            "--exclude=ares[1-20],artemis[10-11,19-20]", "--export=ALL",
            "--chdir=" + str(ROOT), "--signal=B:USR1@600",
            "--output=" + str(directory / "logs/%A_%a.out"),
            "--error=" + str(directory / "logs/%A_%a.err"),
            "--wrap=exec " + shlex.join([sys.executable, str(Path(__file__).resolve()), "worker", "--plan", str(directory)])]


def parse_slurm_duration(value):
    days, _, rest = value.rpartition("-")
    fields = list(map(int, rest.split(":")))
    if len(fields) == 3:
        h, m, s = fields
    elif len(fields) == 2:
        h, m, s = 0, fields[0], fields[1]
    else:
        raise ValueError("Unrecognized Slurm duration: " + value)
    return int(days or 0) * 86400 + h * 3600 + m * 60 + s


def allocation_seconds_left():
    details = subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]],
                                      universal_newlines=True, timeout=30)
    fields = dict(item.split("=", 1) for item in details.split() if "=" in item)
    if fields.get("OverSubscribe") != "NO" and fields.get("Shared") != "0":
        raise RuntimeError("Batch allocation is not exclusive.")
    return max(0, parse_slurm_duration(fields["TimeLimit"]) - parse_slurm_duration(fields["RunTime"]))


def stop_signal(signum, frame):
    global STOP
    STOP = True


def check_node(plan):
    if not re.fullmatch(r"artemis(?:[1-9]|1[2-8])", socket.gethostname().split(".")[0]):
        raise RuntimeError("This plan must execute on an eligible Artemis main-partition node.")
    models = {line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
              if line.startswith("model name")}
    if models != {plan["expected_cpu_model"]}:
        raise RuntimeError("CPU model differs from preserved results: " + str(models))
    mem = next(float(line.split()[1]) / 1048576 for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemTotal:"))
    if not math.isclose(mem, plan["expected_host_ram_gib"], rel_tol=0.01):
        raise RuntimeError("Host RAM capacity differs from preserved results.")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != plan["cpu_threads"]:
        raise RuntimeError("CPU budget differs from the frozen campaign.")
    # Fail closed. Never delete a host/CodeCarbon lock automatically.
    for name in ("awareml-v3-measured-workload.lock", ".codecarbon.lock"):
        path = Path(tempfile.gettempdir()) / name
        if path.exists():
            raise RuntimeError("Inspect existing node-local lock before proceeding: " + str(path))


def run_shard(plan, shard, directory, deadline, runner=subprocess.call, node_check=check_node):
    outcomes = []
    for item in shard:
        if STOP or deadline - time.monotonic() < plan["safety_timeout_sec"] + plan["cleanup_seconds"]:
            return {"state": "allocation_boundary", "outcomes": outcomes}
        task = item["task"]
        task_id = int(task["task_id"])
        previous = latest(plan["campaign"], task_id)
        if previous and eligible(previous, task, plan["campaign_sha256"]):
            outcomes.append({"task_id": task_id, "state": "skipped_completed"})
            continue
        if history(plan["campaign"], task_id) != item["history"]:
            raise RuntimeError("Task history changed after planning; review task %d before retrying." % task_id)
        if item["history"] and not item["retry_authorized"]:
            raise RuntimeError("Retry was not authorized for task %d." % task_id)
        node_check(plan)
        command = [sys.executable, str(ROOT / "scripts/awareml_v3.py"), "run",
                   "--campaign", plan["campaign"], "--task-id", str(task_id)]
        if item["history"]:
            command.append("--retry")
        code = runner(command, cwd=str(ROOT))
        result = latest(plan["campaign"], task_id)
        state = "completed" if code == 0 and result and eligible(result, task, plan["campaign_sha256"]) else "failed"
        outcomes.append({"task_id": task_id, "state": state, "returncode": code})
        write(Path(directory) / "logs" / ("task_%05d.json" % task_id), outcomes[-1])
        if state == "failed":
            return {"state": "stopped_on_failure", "outcomes": outcomes}
    return {"state": "shard_complete", "outcomes": outcomes}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("preview", help="Read-only; safe while old jobs continue")
    q.add_argument("--campaign", required=True)
    q = sub.add_parser("prepare", help="Freeze disjoint shards after all old V3 jobs have drained")
    q.add_argument("--campaign", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--nodes", type=int, default=4)
    q.add_argument("--hours", type=float, default=12)
    q.add_argument("--old-arrays", nargs="+", required=True)
    q.add_argument("--retry-task", type=int, action="append", default=[])
    q = sub.add_parser("submit")
    q.add_argument("--plan", required=True)
    q.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("worker")
    q.add_argument("--plan", required=True)
    a = p.parse_args(argv)
    if a.command == "preview":
        manifest = frozen_campaign(a.campaign)
        inv = inventory(a.campaign, manifest)
        print(json.dumps({"total": sum(map(len, inv.values())),
                          **{k: len(v) for k, v in inv.items()}, "review_task_ids": inv["needs_review"],
                          "read_only": True, "jobs_submitted": False}, indent=2))
    elif a.command == "prepare":
        assert_drained(a.old_arrays)
        manifest = frozen_campaign(a.campaign)
        plan = build_plan(a.campaign, manifest, a.nodes, a.hours, a.old_arrays, a.retry_task)
        directory = Path(a.output).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "logs").mkdir()
        write(directory / "plan.json", {"sha256": digest(plan), "frozen": plan})
        print(json.dumps({"plan": str(directory), "preserved": len(plan["preserved_completed"]),
                          "tasks": sum(map(len, plan["shards"])), "nodes": plan["nodes"]}, indent=2))
    elif a.command == "submit":
        plan = load_plan(a.plan)
        command = submit_command(plan, a.plan)
        print(shlex.join(command), flush=True)
        if not a.dry_run:
            assert_drained(plan["old_arrays"])
            marker = Path(a.plan) / "submission.json"
            with marker.open("x", encoding="utf-8") as handle:
                json.dump({"state": "submission_started_do_not_repeat", "command": command}, handle)
            job = subprocess.check_output(command, universal_newlines=True).strip()
            write(marker, {"state": "submitted", "job": job, "command": command})
            print("Submitted:", job)
    else:
        plan = load_plan(a.plan)
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if not 0 <= index < len(plan["shards"]):
            raise ValueError("Invalid shard index.")
        # A plan can be submitted only once; a shard can execute only once.
        lease = Path(a.plan) / ("shard_%d.started" % index)
        with lease.open("x", encoding="utf-8") as handle:
            json.dump({"job": os.environ["SLURM_JOB_ID"], "node": socket.gethostname()}, handle)
        signal.signal(signal.SIGUSR1, stop_signal)
        signal.signal(signal.SIGTERM, stop_signal)
        deadline = time.monotonic() + allocation_seconds_left()
        result = run_shard(plan, plan["shards"][index], a.plan, deadline)
        write(Path(a.plan) / ("shard_%d.summary.json" % index), result)
        print(json.dumps(result, indent=2))
        return 1 if result["state"] == "stopped_on_failure" else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
