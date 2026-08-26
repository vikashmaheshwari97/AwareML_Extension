from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class HistoricalMeanRegressor:
    def fit(self, X: pd.DataFrame, y):
        self.global_mean_ = float(np.mean(y))
        self.by_fw_ = pd.DataFrame({"framework": X["framework"].astype(str), "y": y}).groupby("framework")["y"].mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame):
        return np.asarray([self.by_fw_.get(str(fw), self.global_mean_) for fw in X["framework"]], dtype=float)


def _tabular_pipeline(model):
    cat = ["framework"]
    num = ["n_features", "n_classes", "n_samples", "drift_rate", "window_size", "time_budget"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", StandardScaler(), num),
    ], remainder="drop")
    return Pipeline([("prep", pre), ("model", model)])


def available_baselines(include_xgboost: bool = True) -> dict[str, object]:
    models = {
        "Historical mean": HistoricalMeanRegressor(),
        "kNN": _tabular_pipeline(KNeighborsRegressor(n_neighbors=7, weights="distance")),
        "Ridge": _tabular_pipeline(Ridge(alpha=1.0)),
        "Random Forest": _tabular_pipeline(RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)),
        "Extra Trees": _tabular_pipeline(ExtraTreesRegressor(n_estimators=250, random_state=42, n_jobs=-1)),
        "HistGradientBoosting": _tabular_pipeline(HistGradientBoostingRegressor(random_state=42)),
    }
    if include_xgboost:
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = _tabular_pipeline(XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
            ))
        except Exception:
            pass
    return models
