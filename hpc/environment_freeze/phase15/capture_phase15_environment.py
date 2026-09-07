from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict

import requests

from hpc.production.phase15.common import (
    find_project_root,
    git_commit,
    now_utc,
    write_json,
)


def cmd(args):
    try:
        p = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return p.stdout.strip()
    except Exception as exc:
        return "{}: {}".format(type(exc).__name__, exc)


def ollama_status(base_url):
    try:
        tags = requests.get(
            base_url.rstrip("/") + "/api/tags",
            timeout=5,
        )
        tags.raise_for_status()
        version = requests.get(
            base_url.rstrip("/") + "/api/version",
            timeout=5,
        )
        version.raise_for_status()
        return {
            "reachable": True,
            "version": version.json().get("version"),
            "models": tags.json().get("models", []),
        }
    except Exception as exc:
        return {
            "reachable": False,
            "error": "{}: {}".format(type(exc).__name__, exc),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ollama-base-url",
        default=os.getenv(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434",
        ),
    )
    args = parser.parse_args()

    root = find_project_root()
    payload: Dict[str, Any] = {
        "schema_version": "phase15_hpc_environment_v1",
        "created_utc": now_utc(),
        "git_commit": git_commit(root),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "uname": cmd(["uname", "-a"]),
        "lscpu": cmd(["lscpu"]),
        "nvidia_smi": cmd(["nvidia-smi"]),
        "nvidia_smi_query": cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "ollama_cli_version": cmd(["ollama", "--version"]),
        "ollama_api": ollama_status(args.ollama_base_url),
        "pip_freeze": cmd(
            [sys.executable, "-m", "pip", "freeze"]
        ),
        "slurm": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("SLURM_")
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.getenv(
                "CUDA_VISIBLE_DEVICES"
            ),
            "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL"),
            "OLLAMA_MODELS": os.getenv("OLLAMA_MODELS"),
            "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL"),
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
        },
    }
    write_json(Path(args.output), payload)
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
