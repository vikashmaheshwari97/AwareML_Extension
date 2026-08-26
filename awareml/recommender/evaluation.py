from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import spearmanr

from .models import available_baselines


REQUIRED = ["dataset_name", "framework", "n_features", "n_classes", "n_samples", "utility"]
FEATURES = ["framework", "n_features", "n_classes", "n_samples", "drift_rate", "window_size", "time_budget"]


def validate_meta_frame(df: pd.DataFrame):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Meta-log is missing columns: {missing}")


def grouped_benchmark(df: pd.DataFrame, include_xgboost: bool = True) -> pd.DataFrame:
    """Leave-one-dataset-out comparison to avoid row-level leakage across the same stream."""
    validate_meta_frame(df)
    work = df.copy()
    for col, default in [("drift_rate", 0.0), ("window_size", 500), ("time_budget", 60.0)]:
        if col not in work:
            work[col] = default
    work["dataset_name"] = work["dataset_name"].astype(str)
    groups = work["dataset_name"].to_numpy()
    logo = LeaveOneGroupOut()
    rows = []
    for name, model in available_baselines(include_xgboost).items():
        top1, regrets, rhos = [], [], []
        for train_idx, test_idx in logo.split(work, groups=groups):
            tr, te = work.iloc[train_idx], work.iloc[test_idx]
            if tr.empty or te.empty:
                continue
            estimator = clone(model) if hasattr(model, "get_params") else type(model)()
            estimator.fit(tr[FEATURES], tr["utility"].to_numpy(float))
            pred = np.asarray(estimator.predict(te[FEATURES]), dtype=float)
            local = te[["framework", "utility"]].copy()
            local["pred"] = pred
            best_true = local.loc[local["utility"].idxmax()]
            best_pred = local.loc[local["pred"].idxmax()]
            top1.append(float(best_true["framework"] == best_pred["framework"]))
            span = max(1e-9, float(local["utility"].max() - local["utility"].min()))
            regrets.append(float((best_true["utility"] - best_pred["utility"]) / span))
            if len(local) >= 3:
                rho = spearmanr(local["utility"], local["pred"]).statistic
                if np.isfinite(rho): rhos.append(float(rho))
        rows.append({
            "model": name,
            "top1_accuracy": float(np.mean(top1)) if top1 else np.nan,
            "normalized_regret": float(np.mean(regrets)) if regrets else np.nan,
            "spearman": float(np.mean(rhos)) if rhos else np.nan,
            "held_out_datasets": len(top1),
        })
    return pd.DataFrame(rows).sort_values(["top1_accuracy", "normalized_regret"], ascending=[False, True])
