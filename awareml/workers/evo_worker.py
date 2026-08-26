"""Native EvOAutoML worker executed by the isolated ``.venv-evo`` interpreter."""
from __future__ import annotations

import json
import sys
from typing import Any

import river
import sklearn
import pandas
import numpy
import EvOAutoML
from EvOAutoML import classification


def _jsonable(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return str(value)


class EvoWorker:
    def __init__(self, seed: int = 42, population_size: int = 10, sampling_rate: int = 500):
        self.seed = int(seed)
        self.population_size = int(population_size)
        self.sampling_rate = int(sampling_rate)
        self._build()

    def _build(self):
        self.model = classification.EvolutionaryBaggingClassifier(
            seed=self.seed,
            population_size=self.population_size,
            sampling_rate=self.sampling_rate,
        )
        self.seen = 0

    def predict_one(self, x: dict):
        try:
            return self.model.predict_one(x)
        except Exception:
            return None

    def predict_proba_one(self, x: dict):
        try:
            fn = getattr(self.model, "predict_proba_one", None)
            value = fn(x) if fn is not None else None
            if isinstance(value, dict) and value:
                return [[_jsonable(k), float(v)] for k, v in value.items()]
        except Exception:
            pass
        return None

    def learn_one(self, x: dict, y: Any):
        self.model.learn_one(x, y)
        self.seen += 1

    def params(self):
        return {
            "evo_version": getattr(EvOAutoML, "__version__", "0.0.14"),
            "river_version": getattr(river, "__version__", "unknown"),
            "sklearn_version": getattr(sklearn, "__version__", "unknown"),
            "pandas_version": getattr(pandas, "__version__", "unknown"),
            "numpy_version": getattr(numpy, "__version__", "unknown"),
            "model": "EvolutionaryBaggingClassifier",
            "population_size": self.population_size,
            "sampling_rate": self.sampling_rate,
            "seen": self.seen,
        }


def respond(payload: dict):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    worker = None
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            cmd = req.get("cmd")
            if cmd == "init":
                worker = EvoWorker(
                    seed=req.get("seed", 42),
                    population_size=req.get("population_size", 10),
                    sampling_rate=req.get("sampling_rate", 500),
                )
                respond({"ok": True, "params": worker.params()})
                continue
            if worker is None:
                raise RuntimeError("Worker not initialized")
            if cmd == "predict":
                respond({"ok": True, "prediction": _jsonable(worker.predict_one(req.get("x", {})))})
            elif cmd == "predict_batch":
                rows = req.get("rows", []) or []
                respond({"ok": True, "predictions": [_jsonable(worker.predict_one(x)) for x in rows]})
            elif cmd == "predict_proba":
                respond({"ok": True, "proba": worker.predict_proba_one(req.get("x", {}))})
            elif cmd == "predict_proba_batch":
                rows = req.get("rows", []) or []
                respond({"ok": True, "probas": [worker.predict_proba_one(x) for x in rows]})
            elif cmd == "learn":
                worker.learn_one(req.get("x", {}), req.get("y"))
                respond({"ok": True})
            elif cmd == "reset":
                worker._build()
                respond({"ok": True})
            elif cmd == "params":
                respond({"ok": True, "params": worker.params()})
            elif cmd == "close":
                respond({"ok": True})
                break
            else:
                respond({"ok": False, "error": f"Unknown command: {cmd}"})
        except Exception as exc:
            respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
