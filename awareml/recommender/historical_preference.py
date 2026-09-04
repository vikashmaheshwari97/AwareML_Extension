from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from awareml.llm.configuration import FRAMEWORK_DEFAULTS
from awareml.recommender.v2_data import (
    EXPECTED_DATASETS,
    EXPECTED_FRAMEWORKS,
    EXPECTED_RUN_ROWS,
    load_canonical_runs,
    load_recommender_train,
)

OBJECTIVES = ("accuracy", "runtime", "energy", "co2")
DEFAULT_WEIGHTS = {
    "accuracy": 0.55,
    "runtime": 0.15,
    "energy": 0.15,
    "co2": 0.15,
}


class HistoricalPreferenceError(RuntimeError):
    """Raised when the historical meta-evidence cannot be ranked safely."""


def normalize_preference_weights(weights: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    raw = dict(weights or {})
    out = {}
    for key in OBJECTIVES:
        try:
            value = float(raw.get(key, 0.0) or 0.0)
        except Exception:
            value = 0.0
        out[key] = max(0.0, value)
    total = float(sum(out.values()))
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {key: value / total for key, value in out.items()}


def _within_dataset_desirability(
    frame: pd.DataFrame,
    metric: str,
    *,
    higher_is_better: bool,
) -> pd.Series:
    values = pd.to_numeric(frame[metric], errors="coerce")

    def transform(group: pd.Series) -> pd.Series:
        finite = group[np.isfinite(group)]
        if finite.empty:
            return pd.Series([np.nan] * len(group), index=group.index)
        lo = float(finite.min())
        hi = float(finite.max())
        if not np.isfinite(lo) or not np.isfinite(hi):
            return pd.Series([np.nan] * len(group), index=group.index)
        if hi - lo <= 1e-15:
            return pd.Series([0.5 if pd.notna(v) else np.nan for v in group], index=group.index)
        unit = (group - lo) / (hi - lo)
        unit = unit.clip(0.0, 1.0)
        return unit if higher_is_better else 1.0 - unit

    return values.groupby(frame["dataset_id"], group_keys=False).apply(transform)


def _attach_desirabilities(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["accuracy_des"] = _within_dataset_desirability(out, "accuracy", higher_is_better=True)
    out["runtime_des"] = _within_dataset_desirability(out, "runtime", higher_is_better=False)
    out["energy_des"] = _within_dataset_desirability(out, "energy", higher_is_better=False)
    out["co2_des"] = _within_dataset_desirability(out, "co2", higher_is_better=False)
    return out


def _score_rows(frame: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    out = _attach_desirabilities(frame)
    parts = []
    for objective in OBJECTIVES:
        parts.append(float(weights[objective]) * pd.to_numeric(out[f"{objective}_des"], errors="coerce"))
    utility = parts[0]
    for part in parts[1:]:
        utility = utility + part
    out["preference_utility"] = utility
    return out


def _rank_per_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["dataset_rank"] = out.groupby("dataset_id")["preference_utility"].rank(
        ascending=False,
        method="min",
    )
    return out


def _aggregate_frameworks(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise HistoricalPreferenceError("No historical rows are available after seed handling.")

    def q25(series):
        return float(pd.to_numeric(series, errors="coerce").quantile(0.25))

    def q75(series):
        return float(pd.to_numeric(series, errors="coerce").quantile(0.75))

    grouped = frame.groupby("framework", as_index=False).agg(
        historical_utility=("preference_utility", "median"),
        mean_utility=("preference_utility", "mean"),
        utility_q25=("preference_utility", q25),
        utility_q75=("preference_utility", q75),
        support_datasets=("dataset_id", "nunique"),
        win_count=("dataset_rank", lambda s: int((pd.to_numeric(s, errors="coerce") == 1).sum())),
        top3_count=("dataset_rank", lambda s: int((pd.to_numeric(s, errors="coerce") <= 3).sum())),
        accuracy_median=("accuracy", "median"),
        runtime_median_sec=("runtime", "median"),
        energy_median_kwh=("energy", "median"),
        co2_median_kg=("co2", "median"),
        accuracy_desirability=("accuracy_des", "mean"),
        runtime_desirability=("runtime_des", "mean"),
        energy_desirability=("energy_des", "mean"),
        co2_desirability=("co2_des", "mean"),
    )
    grouped["utility_iqr"] = grouped["utility_q75"] - grouped["utility_q25"]
    grouped["win_rate"] = grouped["win_count"] / grouped["support_datasets"].replace(0, np.nan)
    grouped["top3_rate"] = grouped["top3_count"] / grouped["support_datasets"].replace(0, np.nan)
    grouped = grouped.sort_values(
        ["historical_utility", "mean_utility", "win_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def _stable_three_seed_rows(train: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset_id",
        "framework",
        "seed_count",
        "accuracy_mean",
        "runtime_sec_mean",
        "energy_kwh_mean",
        "co2_kg_mean",
    }
    missing = sorted(required - set(train.columns))
    if missing:
        raise HistoricalPreferenceError(
            "The 3-seed aggregate snapshot is missing columns: {}".format(", ".join(missing))
        )
    rows = train[list(required)].copy()
    rows = rows.rename(
        columns={
            "accuracy_mean": "accuracy",
            "runtime_sec_mean": "runtime",
            "energy_kwh_mean": "energy",
            "co2_kg_mean": "co2",
        }
    )
    return rows


def _best_observed_seed_rows(runs: pd.DataFrame, weights: Mapping[str, float]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    required = {
        "dataset_id",
        "framework",
        "seed",
        "accuracy",
        "runtime_sec",
        "energy_kwh",
        "co2_kg",
    }
    missing = sorted(required - set(runs.columns))
    if missing:
        raise HistoricalPreferenceError(
            "The canonical run snapshot is missing columns: {}".format(", ".join(missing))
        )

    rows = runs[list(required)].copy().rename(
        columns={
            "runtime_sec": "runtime",
            "energy_kwh": "energy",
            "co2_kg": "co2",
        }
    )
    scored = _score_rows(rows, weights)
    scored = scored.dropna(subset=["preference_utility"])
    if scored.empty:
        raise HistoricalPreferenceError("No canonical run has complete preference evidence.")

    # IMPORTANT: choose one actual seed using the whole preference vector.
    # Never mix accuracy from one seed with energy/CO2/runtime from another seed.
    idx = scored.groupby(["dataset_id", "framework"])["preference_utility"].idxmax()
    selected = scored.loc[idx].copy().reset_index(drop=True)

    seed_usage: Dict[str, Dict[str, int]] = {}
    for framework, group in selected.groupby("framework"):
        counts = group["seed"].astype(str).value_counts().sort_index().to_dict()
        seed_usage[str(framework)] = {str(seed): int(count) for seed, count in counts.items()}

    # Re-normalize after seed selection across the five selected framework runs
    # in each dataset. This keeps the final cross-framework ranking comparable.
    selected = selected.drop(columns=[
        "accuracy_des", "runtime_des", "energy_des", "co2_des", "preference_utility"
    ])
    return selected, seed_usage


def _validated_algorithm(framework: str) -> Tuple[str, Dict[str, Any]]:
    defaults = FRAMEWORK_DEFAULTS.get(str(framework)) or {}
    return str(defaults.get("algorithm") or "N/A"), dict(defaults.get("parameters") or {})


@dataclass(frozen=True)
class HistoricalPreferenceResult:
    ranking: pd.DataFrame
    weights: Dict[str, float]
    seed_mode: str
    dataset_count: int
    run_count: int
    winner: str
    algorithm: str
    parameters: Dict[str, Any]
    seed_usage: Dict[str, Dict[str, int]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "seed_mode": self.seed_mode,
            "dataset_count": int(self.dataset_count),
            "run_count": int(self.run_count),
            "winner": self.winner,
            "algorithm": self.algorithm,
            "parameters": dict(self.parameters),
            "seed_usage": {k: dict(v) for k, v in self.seed_usage.items()},
        }


class HistoricalPreferenceRecommender:
    """Dataset-free historical prior over the frozen 705-run meta evidence.

    This is intentionally NOT ML Recommender V2. V2 is dataset-aware and requires
    a dataset meta-profile. This service gives a transparent global historical
    prior from the same 47-dataset development evidence.
    """

    STABLE = "stable_3seed_aggregate"
    BEST_SEED = "best_observed_single_seed"

    def recommend(
        self,
        weights: Optional[Mapping[str, Any]] = None,
        *,
        seed_mode: str = STABLE,
        train_frame: Optional[pd.DataFrame] = None,
        run_frame: Optional[pd.DataFrame] = None,
    ) -> HistoricalPreferenceResult:
        normalized = normalize_preference_weights(weights)
        seed_usage: Dict[str, Dict[str, int]] = {}

        if seed_mode == self.STABLE:
            train = train_frame.copy() if train_frame is not None else load_recommender_train()
            rows = _stable_three_seed_rows(train)
            scored = _rank_per_dataset(_score_rows(rows, normalized))
            run_count = EXPECTED_RUN_ROWS
        elif seed_mode == self.BEST_SEED:
            runs = run_frame.copy() if run_frame is not None else load_canonical_runs()
            selected, seed_usage = _best_observed_seed_rows(runs, normalized)
            scored = _rank_per_dataset(_score_rows(selected, normalized))
            run_count = int(len(runs))
        else:
            raise HistoricalPreferenceError("Unknown seed mode: {}".format(seed_mode))

        ranking = _aggregate_frameworks(scored)
        if ranking.empty:
            raise HistoricalPreferenceError("Historical ranking is empty.")

        winner = str(ranking.iloc[0]["framework"])
        algorithm, parameters = _validated_algorithm(winner)
        dataset_count = int(scored["dataset_id"].nunique())

        return HistoricalPreferenceResult(
            ranking=ranking,
            weights=normalized,
            seed_mode=seed_mode,
            dataset_count=dataset_count,
            run_count=run_count,
            winner=winner,
            algorithm=algorithm,
            parameters=parameters,
            seed_usage=seed_usage,
        )
