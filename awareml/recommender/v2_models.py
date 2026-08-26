from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .v2_data import V2_INPUT_FEATURES, V2_OPTIONAL_CATEGORICAL_FEATURES


TARGET_DIRECTIONS = {
    "accuracy": "maximize",
    "runtime": "minimize",
    "energy": "minimize",
    "co2": "minimize",
}

TARGET_COLUMNS = {
    "accuracy": "accuracy_mean",
    "runtime": "runtime_sec_mean",
    "energy": "energy_kwh_mean",
    "co2": "co2_kg_mean",
}


class FrameworkMeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, framework_column: str = "framework"):
        self.framework_column = framework_column

    def fit(self, X: pd.DataFrame, y):
        if self.framework_column not in X.columns:
            raise ValueError(
                "FrameworkMeanRegressor requires column {!r}.".format(self.framework_column)
            )
        y_arr = np.asarray(y, dtype=float)
        fw = X[self.framework_column].astype(str).to_numpy()
        self.global_mean_ = float(np.mean(y_arr))
        self.by_framework_ = {}
        for name in sorted(set(fw.tolist())):
            mask = fw == name
            self.by_framework_[name] = float(np.mean(y_arr[mask]))
        return self

    def predict(self, X: pd.DataFrame):
        fw = X[self.framework_column].astype(str)
        return np.asarray(
            [self.by_framework_.get(name, self.global_mean_) for name in fw],
            dtype=float,
        )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[str, int], object]
    requires_xgboost: bool = False


def categorical_features(include_optional_categoricals: bool = True) -> List[str]:
    cols = ["framework"]
    if include_optional_categoricals:
        cols.extend(V2_OPTIONAL_CATEGORICAL_FEATURES)
    return cols


def numeric_features() -> List[str]:
    return [c for c in V2_INPUT_FEATURES if c != "framework"]


def model_feature_columns(include_optional_categoricals: bool = True) -> List[str]:
    return categorical_features(include_optional_categoricals) + numeric_features()


def make_preprocessor(include_optional_categoricals: bool = True) -> ColumnTransformer:
    cat = categorical_features(include_optional_categoricals)
    num = numeric_features()

    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipe, cat),
            ("numeric", numeric_pipe, num),
        ],
        remainder="drop",
    )


def _wrap_target_if_needed(estimator, target: str):
    if target in {"runtime", "energy", "co2"}:
        return TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    return estimator


def _pipeline(regressor, target: str):
    estimator = Pipeline(
        steps=[
            ("prep", make_preprocessor(True)),
            ("model", regressor),
        ]
    )
    return _wrap_target_if_needed(estimator, target)


def _framework_mean_factory(target: str, seed: int):
    del target, seed
    return FrameworkMeanRegressor()


def _ridge_factory(target: str, seed: int):
    del seed
    return _pipeline(Ridge(alpha=1.0), target)


def _knn_factory(target: str, seed: int):
    del seed
    return _pipeline(
        KNeighborsRegressor(n_neighbors=7, weights="distance", p=2),
        target,
    )


def _rf_factory(target: str, seed: int):
    # Intentionally serial: 230-row folds are too small to justify Windows
    # process-spawn overhead from joblib/loky.
    return _pipeline(
        RandomForestRegressor(
            n_estimators=220,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=int(seed),
            n_jobs=1,
        ),
        target,
    )


def _extra_trees_factory(target: str, seed: int):
    return _pipeline(
        ExtraTreesRegressor(
            n_estimators=220,
            max_depth=None,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=int(seed),
            n_jobs=1,
        ),
        target,
    )


def _hgb_factory(target: str, seed: int):
    return _pipeline(
        HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=220,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=int(seed),
        ),
        target,
    )


def _xgb_factory(target: str, seed: int):
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError(
            "XGBoost is unavailable. Install xgboost==2.1.4 or run with --no-xgboost."
        ) from exc

    return _pipeline(
        XGBRegressor(
            n_estimators=260,
            max_depth=4,
            learning_rate=0.04,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="reg:squarederror",
            random_state=int(seed),
            n_jobs=1,
        ),
        target,
    )


def available_model_specs(include_xgboost: bool = True) -> List[ModelSpec]:
    specs = [
        ModelSpec("FrameworkMean", _framework_mean_factory),
        ModelSpec("Ridge", _ridge_factory),
        ModelSpec("kNN", _knn_factory),
        ModelSpec("RandomForest", _rf_factory),
        ModelSpec("ExtraTrees", _extra_trees_factory),
        ModelSpec("HistGradientBoosting", _hgb_factory),
    ]
    if include_xgboost:
        try:
            import xgboost  # noqa: F401
        except Exception:
            pass
        else:
            specs.append(ModelSpec("XGBoost", _xgb_factory, requires_xgboost=True))
    return specs


def target_column(target: str) -> str:
    if target not in TARGET_COLUMNS:
        raise KeyError("Unknown target {!r}.".format(target))
    return TARGET_COLUMNS[target]


def target_direction(target: str) -> str:
    if target not in TARGET_DIRECTIONS:
        raise KeyError("Unknown target {!r}.".format(target))
    return TARGET_DIRECTIONS[target]


def sanitize_predictions(target: str, values) -> np.ndarray:
    pred = np.asarray(values, dtype=float)
    if target == "accuracy":
        return np.clip(pred, 0.0, 1.0)
    return np.maximum(pred, 0.0)
