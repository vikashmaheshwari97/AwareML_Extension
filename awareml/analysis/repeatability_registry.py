from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Optional

import pandas as pd


PAPER_READY_MIN_REPETITIONS = 5
REGISTRY_FILENAME = "registry.json"


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def _slug(value: Any, fallback: str = "none", max_len: int = 40) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    return (text or fallback)[:max_len]


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_dataframe_sha256(df: pd.DataFrame) -> str:
    """Deterministic content fingerprint used by both CLI and Streamlit.

    It is intentionally independent of the source file path. Column order,
    dtypes, index and values are included so two different uploaded files that
    decode to the same active dataframe receive the same experiment identity.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("canonical_dataframe_sha256 expects a pandas DataFrame.")

    h = sha256()
    h.update(
        json.dumps(
            [str(column) for column in df.columns],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    h.update(
        json.dumps(
            [str(dtype) for dtype in df.dtypes],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    # pandas' stable row hashing avoids building a very large CSV string.
    row_hashes = pd.util.hash_pandas_object(
        df,
        index=True,
        categorize=True,
    ).to_numpy()
    h.update(row_hashes.tobytes())
    h.update(str(tuple(df.shape)).encode("utf-8"))
    return h.hexdigest()


def build_dataset_identity(
    *,
    dataset_name: str,
    dataset_content_sha256: str,
    target: str,
    sensitive_attribute: Optional[str],
    positive_label: Any,
) -> dict[str, Any]:
    if not dataset_content_sha256:
        raise ValueError("dataset_content_sha256 is required.")
    if not target:
        raise ValueError("target is required.")

    sensitive = sensitive_attribute if sensitive_attribute else None
    positive = _json_scalar(positive_label)

    identity_payload = {
        "dataset_content_sha256": str(dataset_content_sha256),
        "target": str(target),
        "sensitive_attribute": sensitive,
        "positive_label": positive,
    }
    identity_key = sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    stem = _slug(Path(str(dataset_name or "dataset")).stem, "dataset", 48)
    directory_name = "{}__{}__t-{}__s-{}__p-{}".format(
        stem,
        str(dataset_content_sha256)[:12],
        _slug(target, "target", 30),
        _slug(sensitive, "none", 30),
        _slug(positive, "none", 20),
    )

    return {
        "identity_key": identity_key,
        "directory_name": directory_name,
        "dataset_name": str(dataset_name or "dataset"),
        **identity_payload,
    }


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def registry_path(root: str | Path) -> Path:
    return Path(root) / REGISTRY_FILENAME


def load_registry(root: str | Path) -> dict[str, Any]:
    path = registry_path(root)
    if not path.exists():
        return {
            "schema_version": "phase14-repeatability-registry-v1",
            "paper_ready_min_repetitions": PAPER_READY_MIN_REPETITIONS,
            "runs": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("registry is not a JSON object")
        payload.setdefault("schema_version", "phase14-repeatability-registry-v1")
        payload.setdefault(
            "paper_ready_min_repetitions",
            PAPER_READY_MIN_REPETITIONS,
        )
        payload.setdefault("runs", [])
        return payload
    except Exception:
        # Never silently destroy a malformed registry.
        raise RuntimeError(
            "Could not read Phase-14 repeatability registry: {}".format(path)
        )


def save_registry(root: str | Path, payload: dict[str, Any]) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = registry_path(root)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temp.replace(path)
    return path


def register_run(
    root: str | Path,
    *,
    identity: dict[str, Any],
    run_dir: str | Path,
    manifest: dict[str, Any],
) -> Path:
    root = Path(root)
    run_dir = Path(run_dir)
    payload = load_registry(root)

    relative = str(run_dir.resolve().relative_to(root.resolve()))
    entry = {
        "identity_key": identity["identity_key"],
        "directory_name": identity["directory_name"],
        "dataset_name": identity["dataset_name"],
        "dataset_content_sha256": identity["dataset_content_sha256"],
        "target": identity["target"],
        "sensitive_attribute": identity["sensitive_attribute"],
        "positive_label": identity["positive_label"],
        "created_utc": manifest.get("created_utc"),
        "repetitions": manifest.get("repetitions"),
        "paper_ready": bool(manifest.get("paper_ready")),
        "run_dir": relative,
        "manifest": str(Path(relative) / "repeatability_manifest.json"),
    }

    # Run directories are timestamped, so a rerun becomes a new registry entry.
    existing = [
        row
        for row in payload.get("runs", [])
        if row.get("run_dir") != relative
    ]
    existing.append(entry)
    existing.sort(key=lambda row: str(row.get("created_utc") or ""))
    payload["runs"] = existing
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    return save_registry(root, payload)


def find_latest_matching_run(
    root: str | Path,
    *,
    dataset_content_sha256: str,
    target: str,
    sensitive_attribute: Optional[str],
    positive_label: Any,
) -> Optional[dict[str, Any]]:
    root = Path(root)
    identity = build_dataset_identity(
        dataset_name="dataset",
        dataset_content_sha256=dataset_content_sha256,
        target=target,
        sensitive_attribute=sensitive_attribute,
        positive_label=positive_label,
    )
    payload = load_registry(root)

    matches = [
        row
        for row in payload.get("runs", [])
        if row.get("identity_key") == identity["identity_key"]
    ]
    if not matches:
        return None

    matches.sort(
        key=lambda row: str(row.get("created_utc") or ""),
        reverse=True,
    )
    row = dict(matches[0])
    run_dir = root / str(row["run_dir"])
    row["run_dir_path"] = run_dir
    row["manifest_path"] = run_dir / "repeatability_manifest.json"
    row["results_path"] = run_dir / "phase14_repeated_results.json"
    row["repeatability_table_path"] = run_dir / "repeatability_table.csv"
    row["hardware_table_path"] = run_dir / "hardware_table.csv"
    return row


def list_dataset_runs(root: str | Path) -> list[dict[str, Any]]:
    return list(load_registry(root).get("runs", []))
