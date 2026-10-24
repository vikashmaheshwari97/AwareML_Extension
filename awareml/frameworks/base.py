from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any, Optional
import numpy as np
import pandas as pd


class BaseStreamingFramework(ABC):
    name: str = "Base"

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.backend = "unknown"
        self.research_error_count = 0
        self.research_strict = os.getenv("AWAREML_V3_STRICT", "0") == "1"

    def _research_failure(self, error: Exception) -> None:
        self.research_error_count += 1
        if self.research_strict:
            raise RuntimeError("{} backend error: {}".format(self.name, error)) from error

    @abstractmethod
    def predict_one(self, x: dict[str, float]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def learn_one(self, x: dict[str, float], y: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame | list[dict[str, float]]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            rows = X.to_dict(orient="records")
        else:
            rows = list(X)
        return np.asarray([self.predict_one(r) for r in rows])

    def predict_proba_one(self, x: dict[str, float]):
        """Optional River-style probability API. Return ``None`` when unsupported."""
        return None

    def predict_proba(self, X: pd.DataFrame | list[dict[str, float]]):
        """Optional batch probability API returning a list of probability dictionaries."""
        if isinstance(X, pd.DataFrame):
            rows = X.to_dict(orient="records")
        else:
            rows = list(X)
        out = []
        for row in rows:
            value = self.predict_proba_one(row)
            if not isinstance(value, dict) or not value:
                raise RuntimeError("Class probabilities are not available for this framework.")
            out.append(value)
        # Keep dictionary output because class labels can be strings/bools/ints.
        return out

    def get_params(self) -> dict:
        return {"seed": self.seed, "backend": self.backend,
                "research_error_count": self.research_error_count}

    def native_feature_importance(self) -> Optional[dict[str, float]]:
        return None

    def close(self) -> None:
        """Release optional external resources/processes. Local adapters use a no-op."""
        return None
