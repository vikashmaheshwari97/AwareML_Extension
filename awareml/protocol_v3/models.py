from __future__ import annotations

from pathlib import Path
import os
import tempfile
import warnings

import joblib
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
from threadpoolctl import threadpool_limits

from .common import (METADATA, NUMERIC_FEATURES, OBJECTIVES, TARGETS, code_fingerprint,
                     digest, environment, read_json, sha256, utc_now, write_json)
from .collection import load_collection


class FrameworkMean(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.means_ = pd.DataFrame({"framework": X.framework.astype(str), "y": np.asarray(y)}).groupby("framework").y.mean().to_dict()
        self.global_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.asarray([self.means_.get(str(v), self.global_) for v in X.framework])


def feature_columns(metadata=False):
    return ["framework"] + list(NUMERIC_FEATURES) + (list(METADATA) if metadata else [])


def estimator(name, target, metadata=False, seed=42, n_training_rows=100):
    if name == "FrameworkMean":
        return FrameworkMean()
    constructors = {
        "Ridge": lambda: Ridge(alpha=1.0),
        "kNN": lambda: KNeighborsRegressor(n_neighbors=min(7, n_training_rows), weights="distance"),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=160, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=1),
        "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=160, min_samples_leaf=2, max_features=1.0, random_state=seed, n_jobs=1),
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, learning_rate=0.06, l2_regularization=0.1, random_state=seed, early_stopping=False),
    }
    if name in ("XGBoost", "DemoXGBoostRefit"):
        from xgboost import XGBRegressor
        params = dict(n_estimators=260, max_depth=4, learning_rate=0.04, min_child_weight=2,
                      subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.0)
        if name == "DemoXGBoostRefit":
            # Original available_baselines XGBoost architecture, refit on the
            # revised deployable feature schema, not the old fitted model.
            params = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9)
        model = XGBRegressor(**params, objective="reg:squarederror", random_state=seed, n_jobs=1)
    else:
        if name not in constructors:
            raise ValueError("Unknown candidate family: {}".format(name))
        model = constructors[name]()
    cats = ["framework"] + (list(METADATA) if metadata else [])
    prep = ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                           ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(NUMERIC_FEATURES))], remainder="drop")
    model = Pipeline([("prep", prep), ("model", model)])
    if target != "accuracy":
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
    return model


def predict(model, frame, target, metadata):
    with threadpool_limits(limits=1):
        values = np.asarray(model.predict(frame[feature_columns(metadata)]), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predictions.")
    return np.clip(values, 0, 1) if target == "accuracy" else np.maximum(0, values)


def fit_model(frame, target, name, metadata, seed):
    model = estimator(name, target, metadata, seed, len(frame))
    values = pd.to_numeric(frame[TARGETS[target] + "_mean"], errors="raise").to_numpy()
    with threadpool_limits(limits=1):
        model.fit(frame[feature_columns(metadata)], values)
    return model


def prediction_score(frame, values, target):
    work = frame[["dataset_id", TARGETS[target] + "_mean"]].copy()
    work["prediction"] = values
    regrets, maes = [], []
    sign = 1 if target == "accuracy" else -1
    for _, part in work.groupby("dataset_id", sort=True):
        true = pd.to_numeric(part[TARGETS[target] + "_mean"]).to_numpy(dtype=float)
        pred = part.prediction.to_numpy(dtype=float)
        span = np.ptp(true)
        chosen = int(np.argmax(sign * pred))
        regret = (np.max(sign * true) - sign * true[chosen]) / span if span > 1e-12 else 0.0
        regrets.append(float(regret))
        maes.append(float(np.mean(np.abs(true - pred))))
    return float(np.mean(regrets)), float(np.mean(maes))


def select_family(frame, target, names, metadata=False, seed=42, cache=None):
    ids = sorted(frame.dataset_id.astype(str).unique())
    if len(ids) < 2:
        raise ValueError("Family selection needs at least two training datasets.")
    scores, all_predictions = [], {}
    source_hash = code_fingerprint()["sha256"] if cache else None
    package_versions = environment()["packages"] if cache else None
    for name in names:
        values = np.empty(len(frame))
        for held in ids:
            mask = frame.dataset_id.astype(str).eq(held).to_numpy()
            train, test = frame.loc[~mask], frame.loc[mask]
            # Input content and fitted family participate in the cache key.
            key = digest({"train": train.to_dict("records"), "test_features": test[feature_columns(metadata)].to_dict("records"),
                          "target": target, "name": name, "metadata": metadata, "seed": seed,
                          "code": source_hash, "packages": package_versions}) if cache else None
            path = Path(cache) / (key + ".joblib") if cache else None
            if path and path.exists():
                pred = joblib.load(path)
            else:
                model = fit_model(train, target, name, metadata, seed)
                pred = predict(model, test, target, metadata)
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fd, temporary = tempfile.mkstemp(prefix="fit_", dir=str(path.parent))
                    os.close(fd)
                    try:
                        joblib.dump(pred, temporary)
                        os.replace(temporary, str(path))
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
            values[mask] = pred
        regret, mae = prediction_score(frame, values, target)
        scores.append({"target": target, "model": name, "normalized_regret": regret, "mae": mae})
        all_predictions[name] = values
    order = {name: i for i, name in enumerate(names)}
    selected = min(scores, key=lambda r: (r["normalized_regret"], r["mae"], order[r["model"]]))["model"]
    return selected, scores, all_predictions[selected]


def nested_predictions(frame, names, metadata=False, seed=42, cache=None, progress=None, include_demo=False):
    ids = sorted(frame.dataset_id.astype(str).unique())
    if len(ids) < 3:
        raise ValueError("Nested LODO needs at least three development datasets.")
    output, decisions = [], []
    for held in ids:
        train = frame.loc[~frame.dataset_id.astype(str).eq(held)].reset_index(drop=True)
        test = frame.loc[frame.dataset_id.astype(str).eq(held)].copy().reset_index(drop=True)
        predictions = test[["dataset_id", "framework"]].copy()
        for target in OBJECTIVES:
            if progress:
                print("Selecting within outer training fold: {} / {}".format(held, target), flush=True)
            name, _, _ = select_family(train, target, names, metadata, seed, cache)
            model = fit_model(train, target, name, metadata, seed)
            predictions[target] = predict(model, test, target, metadata)
            predictions[TARGETS[target] + "_observed"] = test[TARGETS[target] + "_mean"].to_numpy()
            if include_demo:
                demo = fit_model(train, target, "DemoXGBoostRefit", metadata, seed)
                predictions["demo_" + target] = predict(demo, test, target, metadata)
            decisions.append({"outer_dataset": held, "target": target, "selected_family": name,
                              "selection_dataset_ids": sorted(train.dataset_id.astype(str).unique()),
                              "outer_dataset_used_for_selection": False})
            if progress:
                progress(held, target, name)
        output.extend(predictions.to_dict("records"))
    return pd.DataFrame(output), decisions


def train_bundle(collection_dir, output, portfolio="primary", metadata=False, complete_case=False, cache_directory=None):
    frame, collection = load_collection(collection_dir, complete_case)
    frozen = collection["frozen"]
    if set(frame.cohort.astype(str)) != {"dev"}:
        raise ValueError("Training accepts development-only collections; prepare/collect eval separately.")
    cfg = frozen["protocol"]
    frameworks = cfg["primary_portfolio"] if portfolio == "primary" else cfg["extended_portfolio"]
    if set(frameworks) - set(frame.framework):
        raise ValueError("The collection does not cover the requested portfolio. Collect that portfolio explicitly.")
    frame = frame.loc[frame.framework.isin(frameworks)].sort_values(["dataset_id", "framework"]).reset_index(drop=True)
    if not all(frame.groupby("dataset_id").framework.nunique().eq(len(frameworks))):
        raise ValueError("Portfolio is incomplete.")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Bundle directory exists. Never overwrite frozen models.")
    output.mkdir(parents=True)
    cache = Path(cache_directory).resolve() if cache_directory else output.parent / "nested_fit_cache"
    names = cfg["candidate_models"]
    seed = int(cfg["model_seed"])
    def progress(held, target, name):
        write_json(output / "training_progress.json", {"outer_dataset": held, "target": target, "selected_family": name, "updated_utc": utc_now()})
        print("Nested LODO: {} / {} -> {}".format(held, target, name), flush=True)
    nested, decisions = nested_predictions(frame, names, metadata, seed, cache, progress, include_demo=True)
    nested.to_csv(output / "nested_predictions.csv", index=False)
    write_json(output / "nested_selection.json", decisions)
    entries, calibrations, selection = {}, {}, []
    demo_entries = {}
    for target in OBJECTIVES:
        selected, scores, _ = select_family(frame, target, names, metadata, seed, cache)
        selection.extend(scores)
        model = fit_model(frame, target, selected, metadata, seed)
        filename = target + ".joblib"
        joblib.dump(model, output / filename, compress=3)
        entries[target] = {"file": filename, "sha256": sha256(output / filename), "family": selected}
        residuals = np.abs(nested[target].to_numpy() - nested[TARGETS[target] + "_observed"].to_numpy())
        calibrations[target] = {"coverage": 0.90, "absolute_radius": float(np.quantile(residuals, 0.90)),
                                "method": "empirical_nested_outer_residuals_not_a_conformal_guarantee"}
        demo = fit_model(frame, target, "DemoXGBoostRefit", metadata, seed)
        demo_name = "demo_xgboost_" + target + ".joblib"
        joblib.dump(demo, output / demo_name, compress=3)
        demo_entries[target] = {"file": demo_name, "sha256": sha256(output / demo_name)}
    pd.DataFrame(selection).to_csv(output / "final_family_selection.csv", index=False)
    frame.to_csv(output / "development_aggregates.csv", index=False)
    write_json(output / "training_collection.json", collection)
    manifest = {"schema": "awareml_meta_recommender_v3", "created_utc": utc_now(), "portfolio": portfolio,
                "frameworks": list(frameworks), "feature_set": "prefix_plus_metadata" if metadata else "prefix",
                "features": feature_columns(metadata), "profile_size": cfg["profile_size"], "protocol": cfg,
                "models": entries, "demo_models": demo_entries, "uncertainty": calibrations,
                "training_dataset_ids": sorted(frame.dataset_id.unique().tolist()),
                "training_sources": [{"dataset_id": d["dataset_id"], "canonical_sha256": d["source_canonical_sha256"]}
                                     for d in frozen["dataset_provenance"] if d["dataset_id"] in set(frame.dataset_id)],
                "collection_sha256": collection["collection_sha256"], "environment": environment(),
                "measurement_hardware": frozen["measurement_hardware"],
                "complete_case": bool(complete_case), "excluded_datasets": frozen["excluded_incomplete_datasets"],
                "demo_baseline_definition": "Original demo XGBoost architecture refit on the revised feature schema; not the original fitted model",
                "file_hashes": {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != "training_progress.json"}}
    write_json(output / "manifest.json", {"bundle_sha256": digest(manifest), "frozen": manifest})
    return manifest


def load_bundle(directory):
    directory = Path(directory)
    payload = read_json(directory / "manifest.json")
    manifest = payload["frozen"]
    if digest(manifest) != payload["bundle_sha256"]:
        raise ValueError("Model manifest integrity failure.")
    for name, expected in manifest["file_hashes"].items():
        if sha256(directory / name) != expected:
            raise ValueError("Model artifact changed: {}".format(name))
    if manifest["environment"]["packages"]["scikit-learn"] != environment()["packages"]["scikit-learn"]:
        raise RuntimeError("Use the training scikit-learn version to load this bundle.")
    models = {target: joblib.load(directory / entry["file"]) for target, entry in manifest["models"].items()}
    return manifest, models
