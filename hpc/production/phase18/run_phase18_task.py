from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from awareml.engine.runner import run_benchmark  # noqa: E402
from awareml.types import RunConfig  # noqa: E402
from phase18_common import (  # noqa: E402
    DATASET_DIR,
    PROTOCOL_ID,
    RUNS_DIR,
    load_task_manifest,
    parse_positive_label,
    apply_task_policy,
    sha256_file,
    utc_now,
    verify_frozen_checksums,
    write_json_atomic,
)


def task_output_dir(task_id: int) -> Path:
    return RUNS_DIR / f"task_{task_id:04d}"


def success_valid(task_dir: Path, task_row: dict) -> bool:
    success = task_dir / "SUCCESS.json"
    result = task_dir / "result.json"
    if not success.exists() or not result.exists():
        return False
    try:
        marker = json.loads(success.read_text(encoding="utf-8"))
        payload = json.loads(result.read_text(encoding="utf-8"))
    except Exception:
        return False
    if int(marker.get("task_id", -1)) != int(task_row["task_id"]):
        return False
    if marker.get("result_sha256") != sha256_file(result):
        return False
    task = payload.get("task") or {}
    if str(task.get("dataset_id")) != str(task_row["dataset_id"]):
        return False
    if str(task.get("framework")) != str(task_row["framework"]):
        return False
    if int(task.get("seed", -1)) != int(task_row["seed"]):
        return False
    return str((payload.get("result") or {}).get("status")) == "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen AwareML Phase-18 HPC task.")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-chacha-fallback",
        action="store_true",
        help="Debug only. Final journal runs should reject transparent ChaCha fallback backends.",
    )
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None:
        raw = os.getenv("SLURM_ARRAY_TASK_ID", "").strip()
        if not raw:
            raise SystemExit("Provide --task-id or SLURM_ARRAY_TASK_ID.")
        task_id = int(raw)

    verify_frozen_checksums()
    tasks = load_task_manifest()
    if task_id < 0 or task_id >= len(tasks):
        raise SystemExit(f"task-id must be 0..{len(tasks)-1}")
    task = tasks.iloc[int(task_id)].to_dict()
    task["task_id"] = int(task_id)
    task["seed"] = int(task["seed"])
    task["max_samples"] = int(task["max_samples"])
    task["window_size"] = int(task["window_size"])
    task["time_budget_sec"] = float(task["time_budget_sec"])
    task["sustainability_repetition_id"] = int(task["sustainability_repetition_id"])
    task["sustainability_repetitions_planned"] = int(task["sustainability_repetitions_planned"])

    out_dir = task_output_dir(task_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    if success_valid(out_dir, task) and not args.force:
        print(f"Task {task_id}: already complete; valid SUCCESS.json found.")
        return

    attempt_root = out_dir / "attempts"
    attempt_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in attempt_root.iterdir() if p.is_dir())
    attempt_no = len(existing) + 1
    attempt_dir = attempt_root / f"attempt_{attempt_no:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)

    dataset_path = DATASET_DIR / str(task["filename"])
    meta = {
        "phase": 18,
        "protocol_id": PROTOCOL_ID,
        "task_id": task_id,
        "attempt": attempt_no,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "started_utc": utc_now(),
        "task": task,
    }
    write_json_atomic(attempt_dir / "attempt_manifest.json", meta)

    try:
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        actual_sha = sha256_file(dataset_path)
        if actual_sha != str(task["dataset_sha256"]):
            raise RuntimeError(
                f"Dataset checksum changed for {dataset_path.name}: actual={actual_sha} "
                f"frozen={task['dataset_sha256']}"
            )

        source_df = pd.read_csv(dataset_path, low_memory=False)
        source_target = str(task.get("source_target") or task.get("target"))
        if source_target not in source_df.columns:
            raise RuntimeError(f"Frozen source target {source_target!r} is missing from {dataset_path.name}.")
        transform_payload = json.loads(str(task.get("task_transform_payload_json") or "{}"))
        df, target, applied_payload = apply_task_policy(
            source_df,
            source_target=source_target,
            task_policy=str(task.get("task_policy") or "native_classification"),
            transform_payload=transform_payload,
            evaluation_target=str(task["target"]),
        )
        frozen_transform_sha = str(task.get("task_transform_sha256") or "")
        if frozen_transform_sha and str(applied_payload.get("transform_sha256") or "") != frozen_transform_sha:
            raise RuntimeError("Frozen task transformation checksum does not match applied transformation.")

        sensitive = str(task.get("sensitive_attribute") or "").strip() or None
        if sensitive and sensitive not in df.columns:
            raise RuntimeError(f"Frozen sensitive attribute {sensitive!r} is missing.")

        positive_label = parse_positive_label(task.get("positive_label_json"))
        cfg = RunConfig(
            target=target,
            sensitive_attribute=sensitive,
            window_size=int(task["window_size"]),
            max_samples=int(task["max_samples"]),
            seed=int(task["seed"]),
            time_budget_sec=float(task["time_budget_sec"]),
            positive_label=positive_label,
            track_sustainability=True,
            sensitive_feature_policy="audit_only",
            missing_prediction_policy="incorrect",
            fairness_min_group_n=10,
            capture_native_xai_snapshots=True,
            xai_method="auto",
            xai_max_rows=250,
            fairness_calibration_bins=10,
            sustainability_repetition_id=int(task["sustainability_repetition_id"]),
            sustainability_repetitions_planned=int(task["sustainability_repetitions_planned"]),
        )

        results = run_benchmark(
            df,
            config=cfg,
            frameworks=[str(task["framework"])],
            record_experiments=True,
            experiment_root=str(ROOT / "artifacts" / "phase18_final_heldout_31_v1" / "experiment_store"),
            dataset_id=str(task["dataset_id"]),
            protocol_version=PROTOCOL_ID,
            experiment_nonce=f"p18-task-{task_id:04d}",
        )
        if len(results) != 1:
            raise RuntimeError(f"Expected exactly one framework result; got {len(results)}")
        result = results[0]
        result_dict = result.to_dict()

        if str(result.status) != "ok":
            raise RuntimeError(f"Framework result status={result.status!r}: {result.error}")
        if str(task["framework"]) == "ChaCha" and "fallback" in str(result.backend).lower():
            if not args.allow_chacha_fallback:
                raise RuntimeError(
                    "ChaCha used a transparent fallback backend. Final Phase-18 production QC rejects "
                    "fallback-backed rows; fix FLAML AutoVW before rerunning this task."
                )
        if int(result.samples) <= 0:
            raise RuntimeError("Framework processed zero samples.")

        payload = {
            "schema_version": "2.0",
            "phase": 18,
            "protocol_id": PROTOCOL_ID,
            "task": task,
            "dataset": {
                "path": str(dataset_path.relative_to(ROOT)),
                "sha256": actual_sha,
                "rows_in_source_file": int(len(source_df)),
                "columns_in_source_file": int(source_df.shape[1]),
                "rows_in_evaluation_stream": int(len(df)),
                "columns_in_evaluation_stream": int(df.shape[1]),
                "source_target": source_target,
                "evaluation_target": target,
                "task_policy": str(task.get("task_policy") or "native_classification"),
                "task_transform_sha256": str(task.get("task_transform_sha256") or ""),
            },
            "run_config": cfg.__dict__,
            "result": result_dict,
            "completed_utc": utc_now(),
            "host": socket.gethostname(),
        }
        attempt_result = attempt_dir / "result.json"
        write_json_atomic(attempt_result, payload)
        write_json_atomic(out_dir / "result.json", payload)
        result_sha = sha256_file(out_dir / "result.json")
        marker = {
            "task_id": task_id,
            "dataset_id": task["dataset_id"],
            "framework": task["framework"],
            "seed": int(task["seed"]),
            "completed_utc": utc_now(),
            "result_sha256": result_sha,
            "attempt": attempt_no,
        }
        write_json_atomic(out_dir / "SUCCESS.json", marker)
        failure = out_dir / "LAST_FAILURE.json"
        if failure.exists():
            failure.unlink()

        print("=" * 72)
        print("Phase-18 task SUCCESS")
        print("Task:", task_id)
        print("Dataset:", task["dataset_id"])
        print("Framework:", task["framework"])
        print("Seed:", task["seed"])
        print("Samples processed:", result.samples)
        print("Accuracy:", result.accuracy)
        print("Runtime sec:", result.runtime_sec)
        print("Energy kWh:", result.energy_kwh)
        print("CO2 kg:", result.co2_kg)
        print("Result SHA256:", result_sha)
        print("=" * 72)

    except Exception as exc:
        failure = {
            "phase": 18,
            "protocol_id": PROTOCOL_ID,
            "task_id": task_id,
            "attempt": attempt_no,
            "failed_utc": utc_now(),
            "task": task,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "host": socket.gethostname(),
        }
        write_json_atomic(attempt_dir / "FAILURE.json", failure)
        write_json_atomic(out_dir / "LAST_FAILURE.json", failure)
        success = out_dir / "SUCCESS.json"
        if success.exists():
            success.unlink()
        print(f"Task {task_id} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
