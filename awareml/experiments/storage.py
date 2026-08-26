from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Optional


class ExperimentStore:
    """Append-safe per-experiment storage designed for local and Slurm execution.

    Every HPC task writes into its own experiment directory, so workers never
    contend for a shared ``meta_logs.json`` file. A later reducer can validate
    and consolidate these immutable artifacts into versioned recommender snapshots.
    """

    STREAM_FILES = {
        "windows": "windows.jsonl",
        "drift_events": "drift_events.jsonl",
        "fairness": "fairness.jsonl",
        "explainability": "explainability.jsonl",
        "sustainability": "sustainability.jsonl",
    }

    def __init__(self, root: str = "artifacts/meta_experiments"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, experiment_id: str) -> Path:
        if not experiment_id or any(x in experiment_id for x in ["/", "\\", ".."]):
            raise ValueError("experiment_id must be a safe path token.")
        path = self.root / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _jsonable(record: Any) -> Dict[str, Any]:
        if hasattr(record, "to_dict"):
            data = record.to_dict()
        elif isinstance(record, dict):
            data = record
        else:
            raise TypeError("record must be a dict or expose to_dict().")
        if not isinstance(data, dict):
            raise TypeError("record serialization must produce a dict.")
        return data

    def _atomic_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def write_run_summary(self, record: Any, overwrite: bool = False) -> Path:
        data = self._jsonable(record)
        exp_id = str(data.get("experiment_id") or "")
        path = self.run_dir(exp_id) / "run.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Run summary already exists: {path}")
        self._atomic_json(path, data)
        return path

    def write_execution_manifest(self, experiment_id: str, manifest: Dict[str, Any], overwrite: bool = False) -> Path:
        path = self.run_dir(experiment_id) / "execution_manifest.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Execution manifest already exists: {path}")
        self._atomic_json(path, dict(manifest))
        return path

    def append(self, experiment_id: str, stream: str, record: Any) -> Path:
        if stream not in self.STREAM_FILES:
            raise ValueError(f"Unknown stream {stream!r}; expected one of {sorted(self.STREAM_FILES)}")
        data = self._jsonable(record)
        record_exp = data.get("experiment_id")
        if record_exp and str(record_exp) != str(experiment_id):
            raise ValueError("record experiment_id does not match target experiment directory.")
        path = self.run_dir(experiment_id) / self.STREAM_FILES[stream]
        line = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        # One process owns one run directory by contract; O_APPEND prevents a
        # partial seek/overwrite if the same process writes repeatedly.
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        return path

    def append_window(self, record: Any) -> Path:
        data = self._jsonable(record)
        return self.append(str(data["experiment_id"]), "windows", data)

    def append_drift_event(self, record: Any) -> Path:
        data = self._jsonable(record)
        return self.append(str(data["experiment_id"]), "drift_events", data)

    def append_fairness(self, record: Any) -> Path:
        data = self._jsonable(record)
        return self.append(str(data["experiment_id"]), "fairness", data)

    def append_explainability(self, record: Any) -> Path:
        data = self._jsonable(record)
        return self.append(str(data["experiment_id"]), "explainability", data)

    def append_sustainability(self, record: Any) -> Path:
        data = self._jsonable(record)
        return self.append(str(data["experiment_id"]), "sustainability", data)
