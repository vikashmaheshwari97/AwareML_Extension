from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, model_validator


class ObjectiveRequest(BaseModel):
    accuracy: float = Field(0.55, ge=0)
    runtime: float = Field(0.15, ge=0)
    energy: float = Field(0.10, ge=0)
    co2: float = Field(0.10, ge=0)
    fairness: float = Field(0.05, ge=0)
    interpretability: float = Field(0.05, ge=0)
    max_runtime_sec: Optional[float] = Field(None, gt=0)
    fairness_metric: Optional[str] = None

    @model_validator(mode="after")
    def validate_sum(self):
        total = self.accuracy + self.runtime + self.energy + self.co2 + self.fairness + self.interpretability
        if total <= 0:
            raise ValueError("At least one objective must have positive weight.")
        return self

    def normalized_weights(self) -> dict[str, float]:
        d = {k: getattr(self, k) for k in ["accuracy", "runtime", "energy", "co2", "fairness", "interpretability"]}
        s = sum(d.values())
        return {k: v / s for k, v in d.items()}
