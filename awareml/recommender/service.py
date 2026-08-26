from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from awareml.engine.pareto import rank_results, objective_correlations
from awareml.types import FrameworkResult, ObjectiveWeights

from .v2_service import V2Recommender


class RecommendationService:
    """Backward-compatible recommendation facade.

    `rank()` preserves the existing post-run ranking behavior.
    `recommend_v2_*()` exposes the Phase-6 learned pre-execution recommender.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        correlation_warning: float = 0.90,
        fairness_metric: str = "composite",
        v2_manifest: Optional[str] = None,
    ):
        self.epsilon = float(epsilon)
        self.correlation_warning = float(
            correlation_warning
        )
        self.fairness_metric = fairness_metric
        self.v2_manifest = v2_manifest
        self._v2 = None

    def rank(
        self,
        results: Iterable[FrameworkResult],
        weights: ObjectiveWeights,
    ):
        frame, recs = rank_results(
            results,
            weights,
            epsilon=self.epsilon,
            fairness_metric=self.fairness_metric,
        )
        corr = objective_correlations(frame)
        warnings = []
        if (
            "energy_kwh" in corr.index
            and "co2_kg" in corr.columns
        ):
            value = corr.loc[
                "energy_kwh",
                "co2_kg",
            ]
            if (
                pd.notna(value)
                and abs(value)
                >= self.correlation_warning
            ):
                warnings.append(
                    "Energy and CO2 are strongly correlated "
                    "(Spearman rho={:.2f}); double weighting may "
                    "over-count the same efficiency signal.".format(
                        value
                    )
                )
        return frame, recs, corr, warnings

    def _get_v2(self) -> V2Recommender:
        if self._v2 is None:
            path = (
                Path(self.v2_manifest)
                if self.v2_manifest
                else None
            )
            self._v2 = V2Recommender(
                manifest_path=path
            )
        return self._v2

    def recommend_v2_profile(
        self,
        profile,
        weights: Optional[Mapping[str, float]] = None,
        ranking_mode: str = "point",
        coverage: float = 0.90,
    ):
        return self._get_v2().recommend_profile(
            profile,
            weights=weights,
            ranking_mode=ranking_mode,
            coverage=coverage,
        )

    def recommend_v2_dataframe(
        self,
        df,
        target: str,
        weights: Optional[Mapping[str, float]] = None,
        window_size: int = 1000,
        time_budget_sec: float = 60.0,
        dataset_family: str = "unknown",
        source_type: str = "unknown",
        drift_type: str = "unknown",
        ranking_mode: str = "point",
        coverage: float = 0.90,
    ):
        return self._get_v2().recommend_dataframe(
            df,
            target=target,
            weights=weights,
            window_size=window_size,
            time_budget_sec=time_budget_sec,
            dataset_family=dataset_family,
            source_type=source_type,
            drift_type=drift_type,
            ranking_mode=ranking_mode,
            coverage=coverage,
        )
