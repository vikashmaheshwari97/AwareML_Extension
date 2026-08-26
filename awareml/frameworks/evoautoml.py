from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any
import numpy as np
import pandas as pd

from .base import BaseStreamingFramework


class EvoAutoMLAdapter(BaseStreamingFramework):
    """Native EvOAutoML 0.0.14 adapter running in the isolated ``.venv-evo``.

    EvOAutoML pins an older pandas/scikit-learn stack than the Streamlit UI, therefore
    it is executed behind a tiny newline-delimited JSON worker boundary. This keeps the
    benchmark faithful to the upstream ``EvolutionaryBaggingClassifier`` without
    contaminating the main AwareML environment.
    """

    name = "EvoAutoML"

    def __init__(self, seed: int = 42, population_size: int = 10, sampling_rate: int = 500):
        super().__init__(seed)
        self.population_size = max(4, int(population_size))
        self.sampling_rate = max(25, int(sampling_rate))
        self.backend = "EvOAutoML isolated native"
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._worker_params: dict[str, Any] = {}
        self._start_worker()

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _python_executable(self) -> Path:
        override = os.getenv("AWAREML_EVO_PYTHON", "").strip()
        if override:
            p = Path(override)
            if not p.is_absolute():
                p = self._project_root() / p
            if p.exists():
                return p

        root = self._project_root()
        candidates = [
            root / ".venv-evo" / "Scripts" / "python.exe",
            root / ".venv-evo" / "bin" / "python",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise RuntimeError(
            "EvoAutoML environment not found. Expected .venv-evo with Python 3.8.10, "
            "river==0.10.1 and EvOAutoML==0.0.14."
        )

    def _start_worker(self) -> None:
        py = self._python_executable()
        worker = self._project_root() / "awareml" / "workers" / "evo_worker.py"
        self._proc = subprocess.Popen(
            [str(py), "-u", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(self._project_root()),
        )
        response = self._request({
            "cmd": "init",
            "seed": self.seed,
            "population_size": self.population_size,
            "sampling_rate": self.sampling_rate,
        })
        self._worker_params = response.get("params", {})
        version = self._worker_params.get("evo_version", "0.0.14")
        self.backend = f"EvOAutoML {version} isolated native"

    def _request(self, payload: dict) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            stderr = ""
            if self._proc is not None and self._proc.stderr is not None:
                try:
                    stderr = self._proc.stderr.read()
                except Exception:
                    pass
            raise RuntimeError(f"EvoAutoML worker is not running. {stderr}".strip())
        assert self._proc.stdin is not None and self._proc.stdout is not None
        with self._lock:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        if not line:
            stderr = ""
            if self._proc.stderr is not None:
                try:
                    stderr = self._proc.stderr.read()
                except Exception:
                    pass
            raise RuntimeError(f"EvoAutoML worker returned no response. {stderr}".strip())
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Unknown EvoAutoML worker error"))
        return response

    def predict_one(self, x: dict[str, float]) -> Any:
        return self._request({"cmd": "predict", "x": x}).get("prediction")

    @staticmethod
    def _decode_proba(items):
        if not isinstance(items, list):
            return None
        out = {}
        for pair in items:
            if isinstance(pair, list) and len(pair) == 2:
                try:
                    out[pair[0]] = float(pair[1])
                except Exception:
                    continue
        return out or None

    def predict_proba_one(self, x: dict[str, float]):
        return self._decode_proba(self._request({"cmd": "predict_proba", "x": x}).get("proba"))

    def predict(self, X):
        rows = X.to_dict(orient="records") if isinstance(X, pd.DataFrame) else list(X)
        return np.asarray(self._request({"cmd": "predict_batch", "rows": rows}).get("predictions", []), dtype=object)

    def predict_proba(self, X):
        rows = X.to_dict(orient="records") if isinstance(X, pd.DataFrame) else list(X)
        raw = self._request({"cmd": "predict_proba_batch", "rows": rows}).get("probas", [])
        out = [self._decode_proba(items) for items in raw]
        if any(not isinstance(v, dict) or not v for v in out):
            raise RuntimeError("EvoAutoML class probabilities are unavailable for one or more rows.")
        return out

    def learn_one(self, x: dict[str, float], y: Any) -> None:
        try:
            if hasattr(y, "item"):
                y = y.item()
        except Exception:
            pass
        self._request({"cmd": "learn", "x": x, "y": y})

    def reset(self) -> None:
        self._request({"cmd": "reset"})

    def get_params(self) -> dict:
        try:
            self._worker_params = self._request({"cmd": "params"}).get("params", self._worker_params)
        except Exception:
            pass
        return {
            **super().get_params(),
            **self._worker_params,
            "isolated_environment": True,
            "required_river": "0.10.1",
        }

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None and proc.stdout is not None:
                proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                proc.stdin.flush()
                proc.stdout.readline()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
