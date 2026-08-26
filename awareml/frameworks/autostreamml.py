from __future__ import annotations

import random
from collections import Counter
from typing import Any
import numpy as np
from river import metrics, preprocessing, linear_model, naive_bayes, tree

try:
    from river import neighbors
except Exception:
    neighbors = None

from .base import BaseStreamingFramework
from .common import clone_model, normalize_importance, safe_metric_value


class AutoStreamMLAdapter(BaseStreamingFramework):
    """Fresh adapter preserving AutoStreamML's population + ARDNS-style neighborhood refresh."""

    name = "AutoStreamML"

    def __init__(self, seed: int = 42, exploration_window: int = 500, budget: int = 8, ensemble_size: int = 3):
        super().__init__(seed)
        self.backend = "AwareML-port/ARDNS"
        self.exploration_window = max(50, int(exploration_window))
        self.budget = max(3, int(budget))
        self.ensemble_size = max(1, int(ensemble_size))
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._build()

    def _base_models(self):
        models = [
            linear_model.LogisticRegression(),
            naive_bayes.GaussianNB(),
            tree.HoeffdingTreeClassifier(grace_period=100),
        ]
        if neighbors is not None:
            try:
                models.append(neighbors.KNNClassifier(n_neighbors=5))
            except Exception:
                pass
        return models

    def _make_candidate(self, variant: int = 0):
        model = clone_model(self._rng.choice(self._base_models()))
        scaler = preprocessing.StandardScaler() if variant % 2 == 0 else preprocessing.MinMaxScaler()
        return scaler | model

    def _build(self):
        self.candidates = [self._make_candidate(i) for i in range(self.budget)]
        self.scores = [metrics.Accuracy() for _ in self.candidates]
        self.best_idx = 0
        self.ensemble = self.candidates[: self.ensemble_size]
        self.counter = 0

    def predict_one(self, x: dict[str, float]) -> Any:
        votes = Counter()
        for model in self.ensemble:
            try:
                pred = model.predict_one(x)
                if pred is not None:
                    votes[pred] += 1
            except Exception:
                continue
        if votes:
            return votes.most_common(1)[0][0]
        try:
            return self.candidates[self.best_idx].predict_one(x)
        except Exception:
            return None


    def predict_proba_one(self, x: dict[str, float]):
        probs = []
        for model in self.ensemble:
            try:
                fn = getattr(model, "predict_proba_one", None)
                if fn is None:
                    continue
                value = fn(x)
                if isinstance(value, dict) and value:
                    probs.append(value)
            except Exception:
                continue
        if not probs:
            try:
                fn = getattr(self.candidates[self.best_idx], "predict_proba_one", None)
                value = fn(x) if fn is not None else None
                return dict(value) if isinstance(value, dict) and value else None
            except Exception:
                return None
        labels = set()
        for p in probs:
            labels.update(p.keys())
        out = {label: float(np.mean([float(p.get(label, 0.0) or 0.0) for p in probs])) for label in labels}
        total = sum(max(0.0, v) for v in out.values())
        return ({k: max(0.0, v) / total for k, v in out.items()} if total > 0 else None)

    def learn_one(self, x: dict[str, float], y: Any) -> None:
        for i, model in enumerate(self.candidates):
            try:
                pred = model.predict_one(x)
                if pred is not None:
                    self.scores[i].update(y, pred)
                model.learn_one(x, y)
            except Exception:
                continue
        self.best_idx = int(np.argmax([safe_metric_value(m) for m in self.scores]))
        self.counter += 1
        if self.counter % self.exploration_window == 0:
            self._refresh_neighborhood()

    def _refresh_neighborhood(self):
        best = clone_model(self.candidates[self.best_idx])
        ranked = np.argsort([safe_metric_value(m) for m in self.scores])[::-1]
        keep = [clone_model(self.candidates[i]) for i in ranked[: min(2, len(ranked))]]
        fresh = [self._make_candidate(self.counter + i) for i in range(self.budget - len(keep))]
        self.candidates = keep + fresh
        self.scores = [metrics.Accuracy() for _ in self.candidates]
        self.best_idx = 0
        self.ensemble = [best] + [clone_model(m) for m in self.candidates[: max(0, self.ensemble_size - 1)]]
        self.ensemble = self.ensemble[: self.ensemble_size]

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._build()

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            "exploration_window": self.exploration_window,
            "budget": self.budget,
            "ensemble_size": self.ensemble_size,
        }

    def native_feature_importance(self):
        try:
            model = list(self.candidates[self.best_idx].steps.values())[-1]
            if hasattr(model, "feature_importances_") and isinstance(model.feature_importances_, dict):
                return normalize_importance(model.feature_importances_)
        except Exception:
            pass
        return None
