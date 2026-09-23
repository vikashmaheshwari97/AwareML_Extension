from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase18_common import RUNS_DIR, load_task_manifest, sha256_file, verify_frozen_checksums  # noqa: E402

HOTFIX_ID = "phase18_runtime_hotfix_v1"
RECOVERY_DIR = ROOT / "artifacts" / "phase18_final_heldout_31_v1" / "recovery"


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def current_git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def load_valid_payload(task_id: int, row: dict):
    task_dir = RUNS_DIR / f"task_{task_id:04d}"
    success = task_dir / "SUCCESS.json"
    result = task_dir / "result.json"
    if not success.exists() or not result.exists():
        return None
    try:
        marker = json.loads(success.read_text(encoding="utf-8"))
        payload = json.loads(result.read_text(encoding="utf-8"))
    except Exception:
        return None
    if marker.get("result_sha256") != sha256_file(result):
        return None
    if int(marker.get("task_id", -1)) != task_id:
        return None
    task = payload.get("task") or {}
    outcome = payload.get("result") or {}
    if str(task.get("dataset_id")) != str(row["dataset_id"]) or str(task.get("framework")) != str(row["framework"]) or int(task.get("seed", -1)) != int(row["seed"]) or str(outcome.get("status")) != "ok":
        return None
    return payload


def positive_finite(value):
    try:
        value = float(value)
    except Exception:
        return False
    return math.isfinite(value) and value > 0.0


def sustainability_bad(payload):
    result = (payload or {}).get("result") or {}
    sustain = result.get("sustainability") or {}
    reasons = []
    if not positive_finite(result.get("energy_kwh")):
        reasons.append("energy_kwh_missing_or_nonpositive")
    if not positive_finite(result.get("co2_kg")):
        reasons.append("co2_kg_missing_or_nonpositive")
    if str(sustain.get("status") or "") != "measured":
        reasons.append("sustainability_status_not_measured")
    return reasons


def compress_ranges(values):
    if not values:
        return ""
    values = sorted(set(int(v) for v in values))
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


def archive_success_marker(task_id: int):
    task_dir = RUNS_DIR / f"task_{task_id:04d}"
    success = task_dir / "SUCCESS.json"
    if not success.exists():
        return
    archived = task_dir / "SUCCESS.pre_hotfix_v1.json"
    if not archived.exists():
        success.replace(archived)
    else:
        success.unlink()


def main():
    parser = argparse.ArgumentParser(description="Audit and recover Phase-18 functional failures and sustainability-invalid runs.")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--account", default="uthpc")
    parser.add_argument("--partition", default="main")
    parser.add_argument("--max-concurrent", type=int, default=1)
    args = parser.parse_args()
    if int(args.max_concurrent) != 1:
        raise SystemExit("Recovery must use --max-concurrent 1 to avoid CodeCarbon /tmp/.codecarbon.lock contention.")
    verify_frozen_checksums()
    tasks = load_task_manifest()
    functional_missing = []
    bad_sustainability = []
    reasons_by_task = {}
    for task_id, row in tasks.iterrows():
        payload = load_valid_payload(int(task_id), row.to_dict())
        if payload is None:
            functional_missing.append(int(task_id))
            reasons_by_task.setdefault(int(task_id), []).append("missing_or_invalid_success")
            continue
        reasons = sustainability_bad(payload)
        if reasons:
            bad_sustainability.append(int(task_id))
            reasons_by_task.setdefault(int(task_id), []).extend(reasons)
    recovery_before = sorted(set(functional_missing) | set(bad_sustainability))
    print("=" * 78)
    print("AwareML Phase-18 recovery audit")
    print("=" * 78)
    print("Functional missing/failed:", len(functional_missing))
    print("Successful but sustainability-invalid:", len(bad_sustainability))
    print("Union requiring recovery:", len(recovery_before))
    print("Recovery array:")
    print(compress_ranges(recovery_before) or "<none>")
    if args.prepare or args.submit:
        for task_id in bad_sustainability:
            archive_success_marker(task_id)
    actual_missing = []
    for task_id, row in tasks.iterrows():
        if load_valid_payload(int(task_id), row.to_dict()) is None:
            actual_missing.append(int(task_id))
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / "data" / "journal" / "phase18_final_heldout_31_v1" / "frozen" / "protocol_lock.json"
    frozen_git_head = None
    if lock_path.exists():
        try:
            frozen_git_head = json.loads(lock_path.read_text(encoding="utf-8")).get("git_head")
        except Exception:
            pass
    manifest = {
        "hotfix_id": HOTFIX_ID,
        "created_utc": utc_now(),
        "frozen_protocol_git_head": frozen_git_head,
        "recovery_code_git_head": current_git_head(),
        "frozen_protocol_checksums_verified": True,
        "functional_missing_task_ids": functional_missing,
        "sustainability_invalid_task_ids": bad_sustainability,
        "recovery_task_ids": actual_missing,
        "reasons_by_task": {str(k): v for k, v in sorted(reasons_by_task.items())},
        "submission_policy": {"max_concurrent": 1, "reason": "avoid CodeCarbon node-local lock contention"},
        "scientific_scope": "Execution compatibility and sustainability instrumentation recovery only; frozen datasets, targets, seeds, budgets, preference manifest and recommender remain unchanged."
    }
    path = RECOVERY_DIR / f"{HOTFIX_ID}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Actual missing after preparation:", len(actual_missing))
    spec = compress_ranges(actual_missing)
    print("Actual missing array:")
    print(spec or "<none>")
    print("Recovery manifest:", path)
    if args.submit:
        if not actual_missing:
            print("Nothing to submit.")
            return
        command = ["sbatch", f"--array={spec}%1", f"--account={args.account}", f"--partition={args.partition}", "hpc/production/phase18/phase18_hotfix_array.sbatch"]
        print("Submitting:")
        print(" ".join(command))
        subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
