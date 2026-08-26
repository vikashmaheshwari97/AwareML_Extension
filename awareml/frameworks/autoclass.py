from __future__ import annotations

import copy
import random
from typing import Any
import numpy as np
from river import metrics, tree, linear_model, naive_bayes

from .base import BaseStreamingFramework
from .common import normalize_importance, safe_metric_value


class AutoClassAdapter(BaseStreamingFramework):
    """Adaptive population search inspired by the AutoClass implementation in AutoML_Stream."""

    name = "AutoClass"

    def __init__(self, seed: int = 42, exploration_window: int = 500, population_size: int = 8):
        super().__init__(seed)
        self.backend = "AwareML-port/adaptive-population"
        self.exploration_window = max(50, int(exploration_window))
        self.population_size = max(3, int(population_size))
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._spaces = {
            "HoeffdingTreeClassifier": {"grace_period": [50, 100, 200, 300]},
            "LogisticRegression": {"l2": [0.0, 1e-4, 1e-3, 1e-2]},
        }
        self._build()

    def _model_pool(self):
        return [
            tree.HoeffdingTreeClassifier(),
            linear_model.LogisticRegression(),
            naive_bayes.GaussianNB(),
        ]

    def _random_model(self):
        model = copy.deepcopy(self._rng.choice(self._model_pool()))
        name = type(model).__name__
        params = self._spaces.get(name, {})
        if params and hasattr(model, "_set_params"):
            chosen = {k: self._rng.choice(v) for k, v in params.items()}
            try:
                model = model._set_params(chosen)
            except Exception:
                pass
        return model

    def _build(self):
        self.population = [self._random_model() for _ in range(self.population_size)]
        self.scores = [metrics.Accuracy() for _ in self.population]
        self.best_idx = 0
        self.counter = 0

    def predict_one(self, x: dict[str, float]) -> Any:
        try:
            return self.population[self.best_idx].predict_one(x)
        except Exception:
            return None


    def predict_proba_one(self, x: dict[str, float]):
        try:
            model = self.population[self.best_idx]
            fn = getattr(model, "predict_proba_one", None)
            if fn is None:
                return None
            out = fn(x)
            return dict(out) if isinstance(out, dict) and out else None
        except Exception:
            return None

    def learn_one(self, x: dict[str, float], y: Any) -> None:
        for i, model in enumerate(self.population):
            try:
                pred = model.predict_one(x)
                if pred is not None:
                    self.scores[i].update(y, pred)
                repeats = max(1, int(self._np_rng.poisson(2)))
                for _ in range(repeats):
                    model.learn_one(x, y)
            except Exception:
                continue
        self.best_idx = int(np.argmax([safe_metric_value(m) for m in self.scores]))
        self.counter += 1
        if self.counter % self.exploration_window == 0:
            self._mutate_worst()

    def _mutate_worst(self):
        scores = [safe_metric_value(m) for m in self.scores]
        best = self.population[int(np.argmax(scores))]
        worst = int(np.argmin(scores))
        child = copy.deepcopy(best)
        name = type(child).__name__
        space = self._spaces.get(name, {})
        if space and hasattr(child, "_set_params"):
            params = {}
            current = child._get_params() if hasattr(child, "_get_params") else {}
            for key, values in space.items():
                now = current.get(key)
                if now in values:
                    idx = values.index(now)
                    idx = max(0, min(len(values) - 1, idx + self._rng.choice([-1, 0, 1])))
                    params[key] = values[idx]
                else:
                    params[key] = self._rng.choice(values)
            try:
                child = child._set_params(params)
            except Exception:
                pass
        else:
            child = self._random_model()
        self.population[worst] = child
        self.scores[worst] = metrics.Accuracy()

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._build()

    def get_params(self) -> dict:
        return {**super().get_params(), "exploration_window": self.exploration_window, "population_size": self.population_size}

    def native_feature_importance(self):
        model = self.population[self.best_idx]
        value = getattr(model, "feature_importances_", None)
        return normalize_importance(value) if isinstance(value, dict) else None
