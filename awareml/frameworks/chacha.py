from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from river import linear_model, metrics, naive_bayes, preprocessing, tree

from .base import BaseStreamingFramework
from .common import safe_metric_value


class ChaChaAdapter(BaseStreamingFramework):
    """ChaCha/AutoVW adapter with an AwareML one-vs-rest multiclass extension.

    FLAML AutoVW is used as the binary base learner. For multiclass streams,
    AwareML maintains one native AutoVW learner per observed class and treats
    each learner as ``class vs. rest``.

    The multiclass layer is an AwareML extension around native FLAML AutoVW;
    it should not be described as upstream native multiclass ChaCha support.

    If AutoVW cannot initialize or fails at runtime, the adapter still exposes
    the previous transparent River/UCB fallback and records the failure.
    Production QC should reject fallback-backed rows when native AutoVW evidence
    is required.
    """

    name = "ChaCha"

    def __init__(self, seed: int = 42, exploration: float = 1.2):
        super().__init__(seed)
        self.exploration = float(exploration)

        self.native_error = None
        self._native_available = False
        self._fallback_active = False

        self._labels: list[Any] = []
        self._ovr_models: dict[Any, Any] = {}

        # Compatibility handle used by earlier AwareML code/tests. When native
        # AutoVW is available this points to the first AutoVW instance.
        self.autovw = None

        self._AutoVW = None
        self._loguniform = None

        self._try_native()
        if not self._native_available:
            self._build_fallback()

    @staticmethod
    def _safe_name(value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
        return value or "f"

    @staticmethod
    def _normalise_label(y: Any) -> Any:
        try:
            if hasattr(y, "item"):
                return y.item()
        except Exception:
            pass
        return y

    def _new_autovw(self, seed: int):
        if self._AutoVW is None or self._loguniform is None:
            raise RuntimeError("FLAML AutoVW backend is unavailable.")

        search_space = {
            "learning_rate": self._loguniform(lower=0.005, upper=0.8),
            "l1": self._loguniform(lower=1e-10, upper=1e-3),
            "l2": self._loguniform(lower=1e-10, upper=1e-2),
            "loss_function": "logistic",
            "link": "logistic",
        }

        return self._AutoVW(
            max_live_model_num=4,
            search_space=search_space,
            init_config={
                "learning_rate": 0.1,
                "l1": 1e-8,
                "l2": 1e-6,
                "loss_function": "logistic",
                "link": "logistic",
            },
            metric="mae_clipped",
            random_seed=int(seed),
            model_selection_mode="min",
        )

    def _try_native(self) -> None:
        try:
            from flaml.onlineml.autovw import AutoVW
            from flaml.tune import loguniform

            self._AutoVW = AutoVW
            self._loguniform = loguniform

            # Probe initialization before a production stream begins. The first
            # observed class reuses this instance as its OVR learner.
            self.autovw = self._new_autovw(self.seed)

            self._native_available = True
            self._fallback_active = False
            self.backend = "FLAML AutoVW / ChaCha OVR extension"
        except Exception as exc:
            self.autovw = None
            self._native_available = False
            self._fallback_active = True
            self.native_error = f"{type(exc).__name__}: {exc}"
            self.backend = "transparent-UCB-fallback (AutoVW init failed)"

    def _activate_fallback(self, error: str) -> None:
        self.native_error = error
        self._native_available = False
        self._fallback_active = True
        self.autovw = None
        self._ovr_models = {}
        self.backend = "transparent-UCB-fallback (AutoVW runtime failure)"
        self._build_fallback()

    def _register_label(self, y: Any) -> bool:
        y = self._normalise_label(y)

        if y in self._ovr_models:
            return True

        if not self._native_available:
            return False

        try:
            if not self._labels:
                model = self.autovw
                if model is None:
                    model = self._new_autovw(self.seed)
                    self.autovw = model
            else:
                # Deterministic class-index-specific seeds; avoid hash() because
                # hash randomization would harm reproducibility.
                model_seed = int(self.seed + 1009 * len(self._labels))
                model = self._new_autovw(model_seed)

            self._labels.append(y)
            self._ovr_models[y] = model
            return True
        except Exception as exc:
            self._activate_fallback(
                f"OVR model init: {type(exc).__name__}: {exc}"
            )
            return False

    def _vw_features(self, x: dict[str, float]) -> str:
        parts = []
        for k, v in x.items():
            try:
                fv = float(v)
                if np.isfinite(fv):
                    parts.append(f"{self._safe_name(k)}:{fv:.12g}")
            except Exception:
                continue
        return "|x " + " ".join(parts)

    @staticmethod
    def _raw_to_probability(raw: Any) -> float:
        if isinstance(raw, (list, tuple, np.ndarray)):
            raw = np.asarray(raw).reshape(-1)[0]

        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"Non-finite AutoVW prediction: {value!r}")

        if 0.0 <= value <= 1.0:
            return float(value)

        value = max(-60.0, min(60.0, value))
        return float(1.0 / (1.0 + math.exp(-value)))

    def _native_proba(self, x: dict[str, float]):
        if not self._native_available or not self._labels:
            return None

        example = self._vw_features(x)
        scores: dict[Any, float] = {}

        try:
            for label in self._labels:
                model = self._ovr_models[label]
                raw = model.predict(example)
                scores[label] = self._raw_to_probability(raw)
        except Exception as exc:
            self.native_error = f"predict: {type(exc).__name__}: {exc}"
            return None

        if len(scores) == 1:
            only = next(iter(scores))
            return {only: 1.0}

        total = sum(max(0.0, value) for value in scores.values())
        if total <= 0.0:
            uniform = 1.0 / len(scores)
            return {label: uniform for label in self._labels}

        return {
            label: float(max(0.0, scores[label]) / total)
            for label in self._labels
        }

    # ------------------------------ transparent fallback ------------------------------

    def _build_fallback(self):
        self.experts = [
            preprocessing.StandardScaler() | linear_model.LogisticRegression(),
            naive_bayes.GaussianNB(),
            tree.HoeffdingTreeClassifier(grace_period=100),
        ]
        self.metrics = [metrics.Accuracy() for _ in self.experts]
        self.pulls = np.zeros(len(self.experts), dtype=int)
        self.total = 0
        self.last_choice = 0

    def _choose(self) -> int:
        self.total += 1
        for i, n in enumerate(self.pulls):
            if n == 0:
                return i
        means = np.array(
            [safe_metric_value(m) for m in self.metrics],
            dtype=float,
        )
        bonus = self.exploration * np.sqrt(
            np.log(max(2, self.total)) / self.pulls
        )
        return int(np.argmax(means + bonus))

    def _fallback_predict(self, x: dict[str, float]) -> Any:
        self.last_choice = self._choose()
        try:
            return self.experts[self.last_choice].predict_one(x)
        except Exception:
            return None

    def _fallback_learn(self, x: dict[str, float], y: Any) -> None:
        for i, expert in enumerate(self.experts):
            try:
                pred = expert.predict_one(x)
                if pred is not None:
                    self.metrics[i].update(y, pred)
                expert.learn_one(x, y)
            except Exception:
                continue
        self.pulls[self.last_choice] += 1

    # -------------------------------- public API --------------------------------

    def predict_one(self, x: dict[str, float]) -> Any:
        if self._native_available:
            probs = self._native_proba(x)
            if probs:
                # Dict order follows class discovery order, giving deterministic
                # tie handling.
                return max(probs, key=probs.get)

            if not self._labels:
                return None

            if self.native_error:
                self._activate_fallback(self.native_error)

        return self._fallback_predict(x)

    def predict_proba_one(self, x: dict[str, float]):
        if self._native_available:
            probs = self._native_proba(x)
            if probs:
                return probs
            if self.native_error:
                self._activate_fallback(self.native_error)

        probs = []
        for expert in getattr(self, "experts", []):
            try:
                fn = getattr(expert, "predict_proba_one", None)
                value = fn(x) if fn is not None else None
                if isinstance(value, dict) and value:
                    probs.append(value)
            except Exception:
                continue

        if not probs:
            return None

        labels = set()
        for value in probs:
            labels.update(value.keys())

        out = {
            label: float(
                np.mean(
                    [
                        float(value.get(label, 0.0) or 0.0)
                        for value in probs
                    ]
                )
            )
            for label in labels
        }

        total = sum(max(0.0, value) for value in out.values())
        return (
            {
                key: max(0.0, value) / total
                for key, value in out.items()
            }
            if total > 0
            else None
        )

    def learn_one(self, x: dict[str, float], y: Any) -> None:
        y = self._normalise_label(y)

        if self._native_available:
            if not self._register_label(y):
                self._fallback_learn(x, y)
                return

            features = self._vw_features(x)

            try:
                # AutoVW.learn() expects predict() to have been called on that
                # AutoVW instance first. Prime every class-specific learner
                # immediately before its OVR update.
                for label in self._labels:
                    model = self._ovr_models[label]
                    model.predict(features)
                    binary_target = 1 if label == y else -1
                    model.learn(f"{binary_target} {features}")
                return
            except Exception as exc:
                self._activate_fallback(
                    f"learn: {type(exc).__name__}: {exc}"
                )

        self._fallback_learn(x, y)

    def reset(self) -> None:
        self.native_error = None
        self._native_available = False
        self._fallback_active = False

        self._labels = []
        self._ovr_models = {}

        self.autovw = None
        self._AutoVW = None
        self._loguniform = None

        self._try_native()
        if not self._native_available:
            self._build_fallback()

    def get_params(self) -> dict:
        label_count = len(self._labels)

        if label_count > 2:
            problem_mode = "multiclass"
        elif label_count == 2:
            problem_mode = "binary"
        elif label_count == 1:
            problem_mode = "single_class_warmup"
        else:
            problem_mode = "unobserved"

        data = {
            **super().get_params(),
            "native_autovw_active": bool(
                self._native_available and not self._fallback_active
            ),
            "native_error": self.native_error,
            "fallback_used": bool(self._fallback_active),
            "exploration": self.exploration,
            "label_count": label_count,
            "labels_seen": [str(label) for label in self._labels],
            "ovr_model_count": len(self._ovr_models),
            "multiclass_strategy": "one_vs_rest_native_autovw",
            "problem_mode": problem_mode,
            "extension_version": "awareml-chacha-ovr-v1",
        }

        if hasattr(self, "pulls"):
            data["arm_pulls"] = self.pulls.tolist()

        return data
