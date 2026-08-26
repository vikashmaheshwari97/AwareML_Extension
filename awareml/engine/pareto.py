from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd

from awareml.types import FrameworkResult, ObjectiveWeights, Recommendation


METRIC_SPECS = {
    "accuracy": "max",
    "runtime_sec": "min",
    "energy_kwh": "min",
    "co2_kg": "min",
    "fairness_score": "max",
    "interpretability_score": "max",
}

FAIRNESS_GAP_KEYS = {
    "composite": None,
    "demographic_parity": "dp_diff",
    "equal_opportunity": "equal_opportunity_diff",
    "equalized_odds": "equalized_odds_gap",
    "predictive_parity": "predictive_parity_diff",
    "error_rate": "error_rate_gap",
}


def _robust_unit(series: pd.Series, direction: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([np.nan] * len(s), index=s.index)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series([0.5 if pd.notna(v) else np.nan for v in s], index=s.index)
    unit = ((s.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)
    return unit if direction == "max" else 1.0 - unit


def fairness_score_from_result(fair: dict, fairness_metric: str = "composite"):
    if not fair or fair.get("status") != "ok":
        return None
    key = FAIRNESS_GAP_KEYS.get(fairness_metric)
    if key:
        gap = fair.get(key)
        return None if gap is None else max(0.0, 1.0 - float(gap))
    gaps = [
        fair.get(k) for k in [
            "dp_diff", "equal_opportunity_diff", "equalized_odds_gap",
            "predictive_parity_diff", "error_rate_gap",
        ] if fair.get(k) is not None
    ]
    return max(0.0, 1.0 - float(np.mean(gaps))) if gaps else None


def results_frame(results: Iterable[FrameworkResult], fairness_metric: str = "composite") -> pd.DataFrame:
    rows = []
    for r in results:
        fair = r.fairness or {}
        exp = r.explainability or {}
        rows.append({
            "framework": r.framework,
            "accuracy": r.accuracy,
            "runtime_sec": r.runtime_sec,
            "energy_kwh": r.energy_kwh,
            "co2_kg": r.co2_kg,
            "fairness_score": fairness_score_from_result(fair, fairness_metric),
            "interpretability_score": exp.get("consistency"),
        })
    return pd.DataFrame(rows)


def epsilon_pareto_mask(df: pd.DataFrame, epsilon: float = 0.05) -> pd.Series:
    """Return epsilon-nondominated rows in normalized all-higher-is-better objective space."""
    cols = [c for c in METRIC_SPECS if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]
    if not cols:
        return pd.Series([True] * len(df), index=df.index)
    norm = pd.DataFrame(index=df.index)
    for c in cols:
        norm[c] = _robust_unit(df[c], METRIC_SPECS[c])
    keep = []
    eps = max(0.0, float(epsilon))
    for i in norm.index:
        dominated = False
        for j in norm.index:
            if i == j:
                continue
            common = norm.loc[[i, j]].dropna(axis=1).columns
            if len(common) == 0:
                continue
            a = norm.loc[i, common].to_numpy(float)
            b = norm.loc[j, common].to_numpy(float)
            if np.all(b >= a - eps) and np.any(b > a + eps):
                dominated = True
                break
        keep.append(not dominated)
    return pd.Series(keep, index=df.index)


def objective_correlations(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df[[c for c in METRIC_SPECS if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method="spearman")


def rank_results(results: Iterable[FrameworkResult], weights: ObjectiveWeights, epsilon: float = 0.05, fairness_metric: str = "composite"):
    df = results_frame(results, fairness_metric=fairness_metric)
    w = weights.as_dict()
    mapping = {
        "accuracy": "accuracy", "runtime": "runtime_sec", "energy": "energy_kwh",
        "co2": "co2_kg", "fairness": "fairness_score", "interpretability": "interpretability_score",
    }
    norm = pd.DataFrame(index=df.index)
    available_weight = pd.Series(0.0, index=df.index)
    utility = pd.Series(0.0, index=df.index)
    for wk, col in mapping.items():
        if col not in df.columns:
            continue
        v = _robust_unit(df[col], METRIC_SPECS[col])
        norm[col] = v
        valid = v.notna().astype(float)
        utility += v.fillna(0.0) * w[wk]
        available_weight += valid * w[wk]
    df["utility"] = utility / available_weight.replace(0, np.nan)
    df["near_pareto"] = epsilon_pareto_mask(df, epsilon=epsilon)
    df["fairness_metric"] = fairness_metric
    df = df.sort_values(["utility", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    recs = []
    for _, row in df.iterrows():
        rationale = [f"weighted utility={row['utility']:.3f}"]
        if row["near_pareto"]:
            rationale.append(f"epsilon-Pareto candidate (ε={epsilon:.2f})")
        if pd.notna(row.get("accuracy")):
            rationale.append(f"accuracy={row['accuracy']:.3f}")
        if pd.notna(row.get("fairness_score")):
            rationale.append(f"{fairness_metric} fairness score={row['fairness_score']:.3f}")
        recs.append(Recommendation(
            framework=str(row["framework"]), utility=float(row["utility"]), rank=int(row["rank"]),
            rationale=rationale, near_pareto=bool(row["near_pareto"]),
        ))
    return df, recs
