from __future__ import annotations

from typing import Iterable
import pandas as pd

from awareml.engine.pareto import rank_results, objective_correlations
from awareml.types import FrameworkResult, ObjectiveWeights


class RecommendationService:
    def __init__(self, epsilon: float = 0.05, correlation_warning: float = 0.90, fairness_metric: str = "composite"):
        self.epsilon = float(epsilon)
        self.correlation_warning = float(correlation_warning)
        self.fairness_metric = fairness_metric

    def rank(self, results: Iterable[FrameworkResult], weights: ObjectiveWeights):
        frame, recs = rank_results(results, weights, epsilon=self.epsilon, fairness_metric=self.fairness_metric)
        corr = objective_correlations(frame)
        warnings = []
        if "energy_kwh" in corr.index and "co2_kg" in corr.columns:
            value = corr.loc["energy_kwh", "co2_kg"]
            if pd.notna(value) and abs(value) >= self.correlation_warning:
                warnings.append(
                    f"Energy and CO2 are strongly correlated (Spearman ρ={value:.2f}); "
                    "double weighting may over-count the same efficiency signal."
                )
        return frame, recs, corr, warnings
