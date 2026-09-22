from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase18_common import EXPECTED_RUNS, RUNS_DIR, load_task_manifest, sha256_file, verify_frozen_checksums  # noqa: E402


def success_valid(task_id: int, row: dict) -> bool:
    task_dir = RUNS_DIR / f"task_{task_id:04d}"
    success = task_dir / "SUCCESS.json"
    result = task_dir / "result.json"
    if not success.exists() or not result.exists():
        return False
    try:
        marker = json.loads(success.read_text(encoding="utf-8"))
        payload = json.loads(result.read_text(encoding="utf-8"))
    except Exception:
        return False
    if marker.get("result_sha256") != sha256_file(result):
        return False
    if int(marker.get("task_id", -1)) != task_id:
        return False
    task = payload.get("task") or {}
    outcome = payload.get("result") or {}
    return (
        str(task.get("dataset_id")) == str(row["dataset_id"])
        and str(task.get("framework")) == str(row["framework"])
        and int(task.get("seed", -1)) == int(row["seed"])
        and str(outcome.get("status")) == "ok"
    )


def compress_ranges(values: list[int]) -> str:
    if not values:
        return ""
    values = sorted(set(values))
    chunks = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        chunks.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    chunks.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect/resume the frozen Phase-18 465-task campaign.")
    parser.add_argument("--submit", action="store_true", help="Submit missing tasks using sbatch.")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--account", default="", help="Optional Slurm account override.")
    parser.add_argument("--partition", default="", help="Optional Slurm partition override.")
    args = parser.parse_args()

    verify_frozen_checksums()
    tasks = load_task_manifest()
    missing = []
    complete = []
    failed = []

    for task_id, row in tasks.iterrows():
        row_dict = row.to_dict()
        if success_valid(int(task_id), row_dict):
            complete.append(int(task_id))
        else:
            missing.append(int(task_id))
            if (RUNS_DIR / f"task_{int(task_id):04d}" / "LAST_FAILURE.json").exists():
                failed.append(int(task_id))

    print("Phase-18 campaign status")
    print("Expected:", EXPECTED_RUNS)
    print("Complete:", len(complete))
    print("Missing / invalid:", len(missing))
    print("With LAST_FAILURE.json:", len(failed))

    if not missing:
        print("Campaign is complete. Run scripts/phase18_collect_results.py next.")
        return

    spec = compress_ranges(missing)
    array_arg = f"{spec}%{max(1, int(args.max_concurrent))}"
    print("Missing array specification:")
    print(array_arg)

    command = ["sbatch", f"--array={array_arg}"]
    if args.account:
        command.append(f"--account={args.account}")
    if args.partition:
        command.append(f"--partition={args.partition}")
    command.append("hpc/production/phase18/phase18_array.sbatch")
    print("Suggested command:")
    print(" ".join(command))

    if args.submit:
        subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
