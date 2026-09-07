from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Mapping

import requests

from awareml.explanation_integrity.benchmark import run_llm_phase15
from awareml.explanation_integrity.faithfulness import (
    OllamaEvidenceExplanationGenerator,
)
from hpc.production.phase15.common import (
    EXPECTED_MODEL,
    EXPECTED_MODEL_DIGEST,
    EXPECTED_OLLAMA_VERSION,
    EXPECTED_PROMPT_VERSION,
    GENERATION_OPTIONS,
    controlled_cases,
    find_project_root,
    git_commit,
    now_utc,
    read_json,
    sha256_file,
    sha256_json,
    stamp_utc,
    write_json,
)


class LockedOllamaClient:
    """Exact-runtime Ollama client for the final Phase-15 empirical campaign."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_sec: float,
        expected_digest: str,
        expected_ollama_version: str,
        strict_runtime: bool = True,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.timeout_sec = float(timeout_sec)
        self.expected_digest = str(expected_digest)
        self.expected_ollama_version = str(expected_ollama_version)
        self.strict_runtime = bool(strict_runtime)

    def _get_json(self, suffix: str, timeout: float = 5.0):
        response = requests.get(
            self.base_url + suffix,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def status(self) -> Dict[str, Any]:
        try:
            tags = self._get_json("/api/tags", timeout=5.0)
            version_payload = self._get_json(
                "/api/version",
                timeout=5.0,
            )
            rows = tags.get("models", []) or []
            exact = None
            for row in rows:
                name = row.get("name") or row.get("model")
                if str(name) == self.model:
                    exact = dict(row)
                    break

            return {
                "reachable": True,
                "configured_model": self.model,
                "resolved_model": (
                    str(exact.get("name") or exact.get("model"))
                    if exact
                    else None
                ),
                "model_digest": (
                    str(exact.get("digest"))
                    if exact and exact.get("digest")
                    else None
                ),
                "model_size_bytes": (
                    exact.get("size")
                    if exact
                    else None
                ),
                "model_modified_at": (
                    exact.get("modified_at")
                    if exact
                    else None
                ),
                "ollama_version": str(
                    version_payload.get("version") or ""
                ),
                "models": [
                    str(row.get("name") or row.get("model"))
                    for row in rows
                    if row.get("name") or row.get("model")
                ],
                "error": None,
            }
        except Exception as exc:
            return {
                "reachable": False,
                "configured_model": self.model,
                "resolved_model": None,
                "model_digest": None,
                "ollama_version": None,
                "models": [],
                "error": "{}: {}".format(
                    type(exc).__name__,
                    exc,
                ),
            }

    def assert_runtime(self) -> Dict[str, Any]:
        status = self.status()
        problems = []

        if not status.get("reachable"):
            problems.append(
                "Ollama unreachable: {}".format(status.get("error"))
            )
        if status.get("resolved_model") != self.model:
            problems.append(
                "Required exact model {} not available; resolved={}"
                .format(self.model, status.get("resolved_model"))
            )
        if status.get("model_digest") != self.expected_digest:
            problems.append(
                "Model digest mismatch: expected {}, got {}"
                .format(
                    self.expected_digest,
                    status.get("model_digest"),
                )
            )
        if status.get("ollama_version") != self.expected_ollama_version:
            problems.append(
                "Ollama version mismatch: expected {}, got {}"
                .format(
                    self.expected_ollama_version,
                    status.get("ollama_version"),
                )
            )

        status["runtime_verified"] = not problems
        status["runtime_problems"] = problems

        if problems and self.strict_runtime:
            raise RuntimeError(
                "Phase-15 runtime lock failed:\n- "
                + "\n- ".join(problems)
            )
        return status

    def generate_text(self, prompt: str):
        payload = {
            "model": self.model,
            "prompt": str(prompt),
            "stream": False,
            "keep_alive": "10m",
            "options": dict(GENERATION_OPTIONS),
        }
        started = time.time()
        response = requests.post(
            self.base_url + "/api/generate",
            json=payload,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        body = response.json()
        text = str(body.get("response") or "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response.")

        return text, {
            "source": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "prompt_version": EXPECTED_PROMPT_VERSION,
            "generation_options": dict(GENERATION_OPTIONS),
            "wall_time_sec": time.time() - started,
            "ollama_total_duration_ns": body.get("total_duration"),
            "ollama_load_duration_ns": body.get("load_duration"),
            "ollama_prompt_eval_count": body.get("prompt_eval_count"),
            "ollama_eval_count": body.get("eval_count"),
            "ollama_eval_duration_ns": body.get("eval_duration"),
        }


def _command_output(args):
    try:
        result = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        return "{}: {}".format(type(exc).__name__, exc)


def _gpu_info():
    text = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    return text


def _attempt_context(
    root: Path,
    campaign_manifest: Mapping[str, Any],
    task_row: Mapping[str, Any],
    runtime_status: Mapping[str, Any],
):
    return {
        "created_utc": now_utc(),
        "campaign_id": campaign_manifest["campaign_id"],
        "task": dict(task_row),
        "git_commit": git_commit(root),
        "campaign_git_commit": campaign_manifest["git_commit"],
        "node": os.getenv("SLURMD_NODENAME") or platform.node(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "slurm_partition": os.getenv("SLURM_JOB_PARTITION"),
        "slurm_cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "gpu_info": _gpu_info(),
        "runtime_status": dict(runtime_status),
    }


def _success_valid(success_path: Path) -> bool:
    if not success_path.exists():
        return False
    try:
        success = read_json(success_path)
        result_path = Path(success["result_path"])
        if not result_path.is_absolute():
            result_path = success_path.parent / result_path
        if not result_path.exists():
            return False
        if sha256_file(result_path) != success.get("result_sha256"):
            return False
        payload = read_json(result_path)
        return bool(payload.get("complete"))
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434",
        ),
    )
    parser.add_argument("--strict-runtime", action="store_true")
    args = parser.parse_args()

    root = find_project_root()
    campaign_root = Path(args.campaign_root).resolve()
    campaign_manifest_path = (
        campaign_root / "campaign_manifest.json"
    )
    campaign = read_json(campaign_manifest_path)

    if git_commit(root) != campaign["git_commit"]:
        raise SystemExit(
            "Git commit changed since campaign preparation. "
            "Expected {}, current {}."
            .format(campaign["git_commit"], git_commit(root))
        )

    tasks = campaign["tasks"]
    if args.task_id < 0 or args.task_id >= len(tasks):
        raise SystemExit("Invalid task id: {}".format(args.task_id))

    task_row = tasks[args.task_id]
    if int(task_row["task_id"]) != args.task_id:
        raise SystemExit("Task manifest is not aligned with array id.")

    cases = controlled_cases()
    case = cases[args.task_id]
    if case.case_id != task_row["case_id"]:
        raise SystemExit(
            "Controlled case mismatch for task {}: {} != {}"
            .format(
                args.task_id,
                case.case_id,
                task_row["case_id"],
            )
        )
    if sha256_json(case.to_dict()) != task_row["case_sha256"]:
        raise SystemExit(
            "Controlled case content changed after campaign preparation."
        )

    case_dir = (
        campaign_root
        / "cases"
        / "task_{:03d}__{}".format(
            args.task_id,
            case.case_id,
        )
    )
    success_path = case_dir / "SUCCESS.json"
    if _success_valid(success_path):
        print(
            "Task {} / {} already has a valid SUCCESS marker; skipping."
            .format(args.task_id, case.case_id)
        )
        return

    attempt_id = "{}__job{}_task{}".format(
        stamp_utc(),
        os.getenv("SLURM_JOB_ID") or "local",
        args.task_id,
    )
    attempt_dir = case_dir / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)

    client = LockedOllamaClient(
        base_url=args.base_url,
        model=EXPECTED_MODEL,
        timeout_sec=args.timeout_sec,
        expected_digest=EXPECTED_MODEL_DIGEST,
        expected_ollama_version=EXPECTED_OLLAMA_VERSION,
        strict_runtime=args.strict_runtime,
    )

    runtime_status = client.assert_runtime()
    context = _attempt_context(
        root,
        campaign,
        task_row,
        runtime_status,
    )
    write_json(attempt_dir / "context.json", context)

    generator = OllamaEvidenceExplanationGenerator(client=client)
    if str(generator.prompt_version) != EXPECTED_PROMPT_VERSION:
        raise SystemExit(
            "Prompt version mismatch: expected {}, got {}."
            .format(
                EXPECTED_PROMPT_VERSION,
                generator.prompt_version,
            )
        )

    started = time.time()
    result = run_llm_phase15(
        cases=[case],
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        fail_fast=False,
        generator=generator,
    )
    result["model_status"] = runtime_status
    result["generation_options"] = dict(GENERATION_OPTIONS)
    result["campaign"] = {
        "campaign_id": campaign["campaign_id"],
        "task_id": args.task_id,
        "case_id": case.case_id,
        "campaign_git_commit": campaign["git_commit"],
        "protocol_sha256": campaign["protocol_sha256"],
    }
    result["wall_time_sec"] = time.time() - started

    result_path = attempt_dir / "phase15_case_result.json"
    write_json(result_path, result)
    write_json(
        attempt_dir / "summary.json",
        result.get("summary") or {},
    )
    write_json(
        attempt_dir / "failures.json",
        result.get("failures") or [],
    )

    if not result.get("complete"):
        failure = {
            "status": "failed",
            "failed_utc": now_utc(),
            "task_id": args.task_id,
            "case_id": case.case_id,
            "attempt_id": attempt_id,
            "result_path": str(result_path),
            "failure_count": result.get("failure_count"),
            "failures": result.get("failures") or [],
        }
        write_json(attempt_dir / "FAILED.json", failure)
        write_json(case_dir / "LAST_FAILURE.json", failure)
        print(json.dumps(failure, indent=2))
        raise SystemExit(2)

    success = {
        "status": "success",
        "completed_utc": now_utc(),
        "task_id": args.task_id,
        "case_id": case.case_id,
        "attempt_id": attempt_id,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "model": EXPECTED_MODEL,
        "model_digest": runtime_status.get("model_digest"),
        "ollama_version": runtime_status.get("ollama_version"),
        "prompt_version": EXPECTED_PROMPT_VERSION,
        "generation_options": dict(GENERATION_OPTIONS),
        "git_commit": campaign["git_commit"],
        "protocol_sha256": campaign["protocol_sha256"],
    }
    write_json(success_path, success)
    try:
        (case_dir / "LAST_FAILURE.json").unlink()
    except FileNotFoundError:
        pass

    print("=" * 88)
    print("Phase-15 HPC case complete")
    print("Task:", args.task_id)
    print("Case:", case.case_id)
    print("Result:", result_path)
    print("SHA256:", success["result_sha256"])
    print("=" * 88)


if __name__ == "__main__":
    main()
