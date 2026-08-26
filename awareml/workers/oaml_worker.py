"""OAML worker for River-0.8 online mode and an optional GAMA streaming bridge."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from typing import Any

import pandas as pd
import river
from river import tree

try:
    from river.drift import ADWIN
except Exception:
    ADWIN = None

GAMA_AVAILABLE = importlib.util.find_spec("gama") is not None
if GAMA_AVAILABLE:
    try:
        from gama import GamaClassifier
        from gama.search_methods import AsyncEA
    except Exception:
        GAMA_AVAILABLE = False
        GamaClassifier = None
        AsyncEA = None
else:
    GamaClassifier = None
    AsyncEA = None


def _jsonable(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return str(value)


class OAMLWorker:
    def __init__(self, seed: int = 42, grace_period: int = 100, mode: str = "online"):
        self.seed = int(seed)
        self.grace_period = int(grace_period)
        self.mode = mode if mode in {"online", "gama"} else "online"
        self.gama_time_budget = max(2, int(os.getenv("AWAREML_OAML_GAMA_BUDGET", "5")))
        self.warmup_size = max(20, int(os.getenv("AWAREML_OAML_GAMA_WARMUP", "50")))
        self.refit_interval = max(self.warmup_size, int(os.getenv("AWAREML_OAML_GAMA_REFIT", "500")))
        self._build()

    def _new_river_model(self):
        try:
            return tree.HoeffdingAdaptiveTreeClassifier(grace_period=self.grace_period), "HoeffdingAdaptiveTreeClassifier"
        except Exception:
            return tree.HoeffdingTreeClassifier(grace_period=self.grace_period), "HoeffdingTreeClassifier"

    def _build(self):
        self.online_model, self.online_model_name = self._new_river_model()
        self.detector = ADWIN() if ADWIN is not None else None
        self.local_drift_events = 0
        self.gama_model = None
        self.gama_fitted = False
        self.gama_fit_count = 0
        self.gama_error = None
        self.X_buffer = []
        self.y_buffer = []
        self.seen = 0
        self.temp_dir = tempfile.mkdtemp(prefix="awareml_gama_")

    def _fit_gama(self):
        if self.mode != "gama" or not GAMA_AVAILABLE or len(self.X_buffer) < self.warmup_size:
            return
        try:
            model = GamaClassifier(
                scoring="accuracy",
                max_total_time=self.gama_time_budget,
                search=AsyncEA(population_size=5),
                store="nothing",
                output_directory=self.temp_dir,
                n_jobs=1,
            )
            start = max(0, len(self.X_buffer) - self.refit_interval)
            X = pd.DataFrame(self.X_buffer[start:])
            y = pd.Series(self.y_buffer[start:])
            model.fit(X, y)
            self.gama_model = model
            self.gama_fitted = True
            self.gama_fit_count += 1
            self.gama_error = None
        except Exception as exc:
            self.gama_error = f"{type(exc).__name__}: {exc}"
            # Keep the River learner alive; failure is exposed in params/backend.

    def predict_one(self, x: dict):
        if self.mode == "gama" and self.gama_fitted and self.gama_model is not None:
            try:
                return self.gama_model.predict(pd.DataFrame([x]))[0]
            except Exception as exc:
                self.gama_error = f"predict: {type(exc).__name__}: {exc}"
        try:
            return self.online_model.predict_one(x)
        except Exception:
            return None

    def predict_proba_one(self, x: dict):
        if self.mode == "gama" and self.gama_fitted and self.gama_model is not None:
            try:
                if hasattr(self.gama_model, "predict_proba"):
                    probs = self.gama_model.predict_proba(pd.DataFrame([x]))[0]
                    classes = list(getattr(self.gama_model, "classes_", []))
                    if classes and len(classes) == len(probs):
                        return [[_jsonable(k), float(v)] for k, v in zip(classes, probs)]
            except Exception as exc:
                self.gama_error = f"predict_proba: {type(exc).__name__}: {exc}"
        try:
            fn = getattr(self.online_model, "predict_proba_one", None)
            value = fn(x) if fn is not None else None
            if isinstance(value, dict) and value:
                return [[_jsonable(k), float(v)] for k, v in value.items()]
        except Exception:
            pass
        return None

    def learn_one(self, x: dict, y: Any):
        pred = self.predict_one(x)
        if self.detector is not None and pred is not None:
            try:
                self.detector.update(0.0 if pred == y else 1.0)
                drifted = bool(getattr(self.detector, "drift_detected", False) or getattr(self.detector, "change_detected", False))
                if drifted:
                    self.local_drift_events += 1
            except Exception:
                drifted = False
        else:
            drifted = False

        self.online_model.learn_one(x, y)
        self.X_buffer.append(x)
        self.y_buffer.append(y)
        self.seen += 1
        if self.mode == "gama":
            needs_initial = (not self.gama_fitted and self.seen >= self.warmup_size)
            needs_refit = self.gama_fitted and (drifted or self.seen % self.refit_interval == 0)
            if needs_initial or needs_refit:
                self._fit_gama()

    def params(self):
        if self.mode == "gama":
            if self.gama_fitted:
                backend = "OAML GAMA streaming bridge (native GAMA + River 0.8 fallback)"
            elif GAMA_AVAILABLE:
                backend = "OAML GAMA bridge warming up (River 0.8 fallback active)"
            else:
                backend = "OAML online/River 0.8 (GAMA unavailable)"
        else:
            backend = f"OAML online/River {getattr(river, '__version__', 'unknown')}"
        return {
            "backend": backend,
            "mode": self.mode,
            "river_version": getattr(river, "__version__", "unknown"),
            "gama_available": bool(GAMA_AVAILABLE),
            "gama_fitted": bool(self.gama_fitted),
            "gama_fit_count": self.gama_fit_count,
            "gama_error": self.gama_error,
            "gama_time_budget_sec": self.gama_time_budget,
            "gama_warmup": self.warmup_size,
            "gama_refit_interval": self.refit_interval,
            "grace_period": self.grace_period,
            "online_model": self.online_model_name,
            "local_drift_events": self.local_drift_events,
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
            req = json.loads(raw); cmd = req.get("cmd")
            if cmd == "init":
                worker = OAMLWorker(seed=req.get("seed", 42), grace_period=req.get("grace_period", 100), mode=req.get("mode", "online"))
                respond({"ok": True, "params": worker.params()}); continue
            if worker is None:
                raise RuntimeError("Worker not initialized")
            if cmd == "predict": respond({"ok": True, "prediction": _jsonable(worker.predict_one(req.get("x", {})))})
            elif cmd == "predict_batch":
                rows = req.get("rows", []) or []
                respond({"ok": True, "predictions": [_jsonable(worker.predict_one(x)) for x in rows]})
            elif cmd == "predict_proba": respond({"ok": True, "proba": worker.predict_proba_one(req.get("x", {}))})
            elif cmd == "predict_proba_batch":
                rows = req.get("rows", []) or []
                respond({"ok": True, "probas": [worker.predict_proba_one(x) for x in rows]})
            elif cmd == "learn": worker.learn_one(req.get("x", {}), req.get("y")); respond({"ok": True})
            elif cmd == "reset": worker._build(); respond({"ok": True})
            elif cmd == "params": respond({"ok": True, "params": worker.params()})
            elif cmd == "close": respond({"ok": True}); break
            else: respond({"ok": False, "error": f"Unknown command: {cmd}"})
        except Exception as exc:
            respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
