from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List, Mapping, Optional


EXPECTED_CASE_COUNT = 24
EXPECTED_SOURCE_ORDER = ("B", "E", "F_XAI", "F_CHAT")
EXPECTED_MODEL = "llama3:8b"
EXPECTED_MODEL_DIGEST = (
    "365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1"
)
EXPECTED_OLLAMA_VERSION = "0.32.14"
EXPECTED_PROMPT_VERSION = "phase15_explanation_prompt_v4"

GENERATION_OPTIONS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
    "num_predict": 512,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_text(args: List[str], cwd: Optional[Path] = None) -> str:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return "{}: {}".format(type(exc).__name__, exc)


def git_commit(root: Path) -> str:
    return run_text(["git", "rev-parse", "HEAD"], cwd=root)


def git_is_dirty(root: Path) -> bool:
    text = run_text(["git", "status", "--porcelain"], cwd=root)
    if text.startswith("CalledProcessError"):
        raise RuntimeError("Could not inspect git working tree: " + text)
    return bool(text.strip())


def git_diff_summary(root: Path) -> str:
    return run_text(["git", "status", "--short"], cwd=root)


def controlled_cases():
    from awareml.explanation_integrity.controlled import build_controlled_cases

    cases = list(build_controlled_cases())
    if len(cases) != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            "Expected {} controlled cases, found {}."
            .format(EXPECTED_CASE_COUNT, len(cases))
        )

    expected_ids = []
    for context_idx in range(1, 7):
        for suffix in EXPECTED_SOURCE_ORDER:
            expected_ids.append(
                "P15_{:02d}_{}".format(context_idx, suffix)
            )

    observed = [case.case_id for case in cases]
    if observed != expected_ids:
        raise RuntimeError(
            "Controlled-case order changed.\nExpected: {}\nObserved: {}"
            .format(expected_ids, observed)
        )
    return cases


def prompt_version() -> str:
    from awareml.explanation_integrity.faithfulness import (
        OllamaEvidenceExplanationGenerator,
    )

    value = str(OllamaEvidenceExplanationGenerator.prompt_version)
    if value != EXPECTED_PROMPT_VERSION:
        raise RuntimeError(
            "Empirical prompt version changed: expected {}, got {}."
            .format(EXPECTED_PROMPT_VERSION, value)
        )
    return value


def protocol_path(root: Path) -> Path:
    return (
        Path(root)
        / "data"
        / "journal"
        / "phase15_hpc_empirical_protocol_v1"
        / "design"
        / "protocol.json"
    )


def protocol_sha256(root: Path) -> str:
    path = protocol_path(root)
    if not path.exists():
        raise FileNotFoundError("Missing empirical protocol: {}".format(path))
    return sha256_file(path)


def expected_task_rows(cases) -> List[Dict[str, Any]]:
    rows = []
    for task_id, case in enumerate(cases):
        rows.append(
            {
                "task_id": task_id,
                "case_id": case.case_id,
                "dataset_id": case.dataset_id,
                "source_stage": case.source_stage,
                "source_name": case.source_name,
                "case_sha256": sha256_json(case.to_dict()),
            }
        )
    return rows


def find_project_root(start: Optional[Path] = None) -> Path:
    candidates = []
    if os.getenv("AWAREML_ROOT"):
        candidates.append(Path(os.environ["AWAREML_ROOT"]))
    if start is not None:
        candidates.append(Path(start))
    candidates.extend(
        [
            Path.cwd(),
            Path.home() / "AwareMLExtension",
            Path.home() / "AwareML_Extension",
        ]
    )

    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "awareml").is_dir() and (candidate / "hpc").is_dir():
            return candidate

    raise RuntimeError(
        "Could not locate AwareML project root. Set AWAREML_ROOT."
    )
