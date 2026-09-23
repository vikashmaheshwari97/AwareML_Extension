from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
HERE = Path(__file__).resolve().parent
for value in (str(ROOT), str(SCRIPTS), str(HERE)):
    if value not in sys.path:
        sys.path.insert(0, value)

from phase18_runtime_hotfix import HOTFIX_ID, install_phase18_runtime_hotfixes
install_phase18_runtime_hotfixes()
import run_phase18_task as base_runner  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def current_git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def resolve_task_id():
    for i, arg in enumerate(sys.argv):
        if arg == "--task-id" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if arg.startswith("--task-id="):
            return int(arg.split("=", 1)[1])
    raw = os.getenv("SLURM_ARRAY_TASK_ID", "").strip()
    return int(raw) if raw else None


def write_sidecar(task_id, status, error=None):
    if task_id is None:
        return
    out = ROOT / "artifacts" / "phase18_final_heldout_31_v1" / "runs" / f"task_{task_id:04d}"
    out.mkdir(parents=True, exist_ok=True)
    frozen_lock = ROOT / "data" / "journal" / "phase18_final_heldout_31_v1" / "frozen" / "protocol_lock.json"
    frozen_git_head = None
    if frozen_lock.exists():
        try:
            frozen_git_head = json.loads(frozen_lock.read_text(encoding="utf-8")).get("git_head")
        except Exception:
            pass
    payload = {
        "hotfix_id": HOTFIX_ID,
        "task_id": int(task_id),
        "status": status,
        "timestamp_utc": utc_now(),
        "frozen_protocol_git_head": frozen_git_head,
        "execution_hotfix_git_head": current_git_head(),
        "scope": [
            "EvoAutoML external-label integer codec",
            "ExperimentStore nested NumPy/pandas JSON normalization",
            "recovery tasks scheduled serially to avoid CodeCarbon lock contention",
        ],
        "scientific_note": "Execution-compatibility/instrumentation recovery only; frozen datasets, targets, seeds, budgets, preferences and recommender are unchanged.",
        "error": error,
    }
    tmp = out / "HOTFIX_EXECUTION.json.tmp"
    final = out / "HOTFIX_EXECUTION.json"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(final)


def main():
    task_id = resolve_task_id()
    write_sidecar(task_id, "started")
    try:
        base_runner.main()
    except Exception as exc:
        write_sidecar(task_id, "failed", f"{type(exc).__name__}: {exc}")
        raise
    else:
        write_sidecar(task_id, "completed")


if __name__ == "__main__":
    main()
