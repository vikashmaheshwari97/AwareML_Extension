from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


FRAMEWORKS = ["AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha"]


def locate_meta_logs(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate historical AwareML meta logs without asking the user to upload them again."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.getenv("AWAREML_META_LOGS")
    if env:
        candidates.append(Path(env))
    active = Path("data/meta/active_snapshot.txt")
    if active.exists():
        try:
            rel = active.read_text(encoding="utf-8").strip()
            if rel:
                candidates.append(Path("data/meta") / rel)
        except Exception:
            pass
    candidates.extend([
        Path("data/meta/legacy/meta_logs_v1_47.json"),
        Path("data/meta/meta_logs.json"),
        Path("meta_logs.json"),
    ])
    seen = set()
    for path in candidates:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists() and path.is_file():
            return path
    return None


def load_meta_logs(path: Optional[str] = None) -> pd.DataFrame:
    meta_path = locate_meta_logs(path)
    if meta_path is None:
        return pd.DataFrame()
    try:
        suffix = meta_path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(meta_path)
        elif suffix in {".jsonl", ".ndjson"}:
            df = pd.read_json(meta_path, lines=True)
        else:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                return pd.DataFrame()
            df = pd.DataFrame(payload)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df

    rename = {}
    if "best_algorithm" in df.columns and "algorithm" not in df.columns:
        rename["best_algorithm"] = "algorithm"
    if "CO2 Emission (µg)" in df.columns and "co2_ug" not in df.columns:
        rename["CO2 Emission (µg)"] = "co2_ug"
    if rename:
        df = df.rename(columns=rename)

    for col in [
        "n_samples", "n_features", "n_classes", "accuracy", "runtime_sec",
        "runtime_total", "energy_consumption_kwh", "co2_ug", "window_size", "time_budget",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "drift_detected" in df.columns:
        df["drift_detected"] = df["drift_detected"].fillna(False).astype(bool)
    else:
        df["drift_detected"] = False

    if "framework" in df.columns:
        df["framework"] = df["framework"].astype(str)
    if "dataset_name" in df.columns:
        df["dataset_name"] = df["dataset_name"].astype(str)

    # Historical zero carbon values frequently mean "not measured" rather than
    # a physically exact zero. Treat non-positive values as missing for model fitting.
    if "co2_ug" in df.columns:
        df.loc[df["co2_ug"] <= 0, "co2_ug"] = np.nan

    return df


def meta_log_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "rows": 0, "datasets": 0, "frameworks": {},
            "energy_coverage": 0.0, "co2_coverage": 0.0,
        }
    energy = pd.to_numeric(df.get("energy_consumption_kwh"), errors="coerce")
    co2 = pd.to_numeric(df.get("co2_ug"), errors="coerce")
    return {
        "rows": int(len(df)),
        "datasets": int(df["dataset_name"].nunique()) if "dataset_name" in df.columns else 0,
        "frameworks": df["framework"].value_counts().to_dict() if "framework" in df.columns else {},
        "energy_coverage": float(energy.notna().mean()) if energy is not None else 0.0,
        "co2_coverage": float(co2.notna().mean()) if co2 is not None else 0.0,
    }


def _safe_minmax(values: pd.Series, benefit: bool) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([0.5] * len(s), index=s.index, dtype=float)
    lo = float(s.min())
    hi = float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        out = pd.Series([0.5] * len(s), index=s.index, dtype=float)
    else:
        out = (s - lo) / (hi - lo)
        if not benefit:
            out = 1.0 - out
    return out.fillna(0.5).clip(0.0, 1.0)


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    clean = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        clean = {"accuracy": 0.55, "runtime": 0.15, "energy": 0.15, "co2": 0.15}
        total = 1.0
    return {k: v / total for k, v in clean.items()}


class HistoricalMLRecommender:
    """ML recommender backed by the historical AwareML meta-log schema.

    The class deliberately mirrors the useful scientific logic from the previous
    dashboard while keeping the new project modular. It predicts expected
    accuracy/runtime/energy/carbon for each framework and ranks candidates under
    explicit user preferences. The algorithm/hyperparameter suggestion comes from
    the nearest historical evidence for the selected framework.
    """

    def __init__(self, meta_df: Optional[pd.DataFrame] = None, meta_path: Optional[str] = None, seed: int = 42):
        self.df = meta_df.copy() if meta_df is not None else load_meta_logs(meta_path)
        self.seed = int(seed)
        self.models: Dict[str, RandomForestRegressor] = {}
        self.feature_cols: List[str] = []
        self.frameworks: List[str] = []
        self.metric_support: Dict[str, Dict[str, int]] = {}
        self.trained = False

    def _feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=df.index)
        x["n_features"] = pd.to_numeric(df.get("n_features"), errors="coerce").fillna(0.0)
        x["n_classes"] = pd.to_numeric(df.get("n_classes"), errors="coerce").fillna(0.0)
        n_samples = pd.to_numeric(df.get("n_samples"), errors="coerce").fillna(0.0)
        x["log_n_samples"] = np.log1p(n_samples.clip(lower=0.0))
        x["window_size"] = pd.to_numeric(df.get("window_size"), errors="coerce").fillna(0.0)
        x["time_budget"] = pd.to_numeric(df.get("time_budget"), errors="coerce").fillna(0.0)
        x["drift_detected"] = df.get("drift_detected", False).astype(bool).astype(float)
        fw = df.get("framework", pd.Series([""] * len(df), index=df.index)).astype(str)
        for name in FRAMEWORKS:
            x["fw_" + name] = (fw == name).astype(float)
        return x

    def train(self) -> "HistoricalMLRecommender":
        if self.df is None or self.df.empty:
            raise RuntimeError("No historical meta logs are available.")
        required = {"framework", "n_features", "n_classes", "accuracy", "runtime_sec", "energy_consumption_kwh"}
        missing = sorted(required - set(self.df.columns))
        if missing:
            raise RuntimeError("Historical meta logs are missing: " + ", ".join(missing))

        train_df = self.df[self.df["framework"].isin(FRAMEWORKS)].copy()
        self.frameworks = [fw for fw in FRAMEWORKS if fw in set(train_df["framework"])]
        X = self._feature_matrix(train_df)
        self.feature_cols = list(X.columns)

        targets = {
            "accuracy": pd.to_numeric(train_df["accuracy"], errors="coerce"),
            "runtime_sec": pd.to_numeric(train_df["runtime_sec"], errors="coerce"),
            "energy_kwh": pd.to_numeric(train_df["energy_consumption_kwh"], errors="coerce"),
            "co2_ug": pd.to_numeric(train_df.get("co2_ug"), errors="coerce"),
        }

        fw_counts = train_df["framework"].value_counts()
        base_weights = train_df["framework"].map(lambda fw: 1.0 / float(fw_counts.get(fw, 1))).astype(float)

        self.models = {}
        self.metric_support = {}
        for metric, y in targets.items():
            self.metric_support[metric] = {}
            for fw in self.frameworks:
                fw_mask = train_df["framework"].eq(fw) & y.notna()
                self.metric_support[metric][fw] = int(fw_mask.sum())
            mask = y.notna()
            if int(mask.sum()) < max(8, len(self.frameworks) * 2):
                continue
            model = RandomForestRegressor(
                n_estimators=240,
                max_depth=12,
                min_samples_leaf=2,
                random_state=self.seed,
                n_jobs=-1,
            )
            model.fit(X.loc[mask], y.loc[mask], sample_weight=base_weights.loc[mask].values)
            self.models[metric] = model
        self.trained = True
        return self

    def _profile_row(self, n_features: int, n_classes: int, n_samples: int, framework: str) -> pd.DataFrame:
        if not self.trained:
            raise RuntimeError("Call train() first.")
        med = self._feature_matrix(self.df).median(numeric_only=True).to_dict()
        row = {c: float(med.get(c, 0.0)) for c in self.feature_cols}
        row["n_features"] = float(n_features)
        row["n_classes"] = float(n_classes)
        row["log_n_samples"] = math.log1p(max(0, int(n_samples)))
        for fw in FRAMEWORKS:
            key = "fw_" + fw
            if key in row:
                row[key] = 1.0 if fw == framework else 0.0
        return pd.DataFrame([row], columns=self.feature_cols)

    def _nearest_historical(self, framework: str, n_features: int, n_classes: int, n_samples: int) -> Tuple[Optional[pd.Series], pd.DataFrame]:
        block = self.df[self.df["framework"] == framework].copy()
        if block.empty:
            return None, block
        nf = pd.to_numeric(block.get("n_features"), errors="coerce").fillna(0.0)
        nc = pd.to_numeric(block.get("n_classes"), errors="coerce").fillna(0.0)
        ns = pd.to_numeric(block.get("n_samples"), errors="coerce").fillna(0.0)
        d_feat = (nf - float(n_features)).abs() / max(1.0, float(n_features))
        d_cls = (nc - float(n_classes)).abs() / max(1.0, float(n_classes))
        d_n = (np.log1p(ns.clip(lower=0.0)) - math.log1p(max(0, int(n_samples)))).abs() / 10.0
        block["_distance"] = 0.45 * d_feat + 0.35 * d_cls + 0.20 * d_n
        block = block.sort_values("_distance")
        return block.iloc[0], block.head(5)

    def predict_candidates(self, n_features: int, n_classes: int, n_samples: int, weights: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not self.trained:
            self.train()
        weights = _normalize_weights(weights or {"accuracy": 0.55, "runtime": 0.15, "energy": 0.15, "co2": 0.15})
        rows: List[Dict[str, Any]] = []
        neighbor_map: Dict[str, List[Dict[str, Any]]] = {}

        for fw in self.frameworks:
            X = self._profile_row(n_features, n_classes, n_samples, fw)
            pred: Dict[str, Any] = {"framework": fw}
            for metric in ["accuracy", "runtime_sec", "energy_kwh", "co2_ug"]:
                model = self.models.get(metric)
                support = int((self.metric_support.get(metric) or {}).get(fw, 0))
                # Do not extrapolate a metric for a framework that has no historical measurements.
                pred[metric] = float(model.predict(X)[0]) if model is not None and support >= 2 else np.nan
                pred[metric + "_support"] = support
                if metric != "accuracy" and np.isfinite(pred[metric]):
                    pred[metric] = max(0.0, pred[metric])
            nearest, neighbors = self._nearest_historical(fw, n_features, n_classes, n_samples)
            if nearest is not None:
                pred["algorithm"] = nearest.get("algorithm", "N/A")
                pred["best_hyperparams"] = nearest.get("best_hyperparams", {})
                pred["nearest_dataset"] = nearest.get("dataset_name", "N/A")
                pred["evidence_distance"] = float(nearest.get("_distance", np.nan))
            else:
                pred["algorithm"] = "N/A"
                pred["best_hyperparams"] = {}
                pred["nearest_dataset"] = "N/A"
                pred["evidence_distance"] = np.nan
            keep = [c for c in ["dataset_name", "accuracy", "runtime_sec", "energy_consumption_kwh", "co2_ug", "algorithm", "_distance"] if c in neighbors.columns]
            neighbor_map[fw] = neighbors[keep].replace({np.nan: None}).to_dict(orient="records") if not neighbors.empty else []
            rows.append(pred)

        out = pd.DataFrame(rows)
        out["accuracy_score"] = _safe_minmax(out["accuracy"], benefit=True)
        out["runtime_score"] = _safe_minmax(out["runtime_sec"], benefit=False)
        out["energy_score"] = _safe_minmax(out["energy_kwh"], benefit=False)
        out["co2_score"] = _safe_minmax(out["co2_ug"], benefit=False)
        out["utility"] = (
            weights.get("accuracy", 0.0) * out["accuracy_score"]
            + weights.get("runtime", 0.0) * out["runtime_score"]
            + weights.get("energy", 0.0) * out["energy_score"]
            + weights.get("co2", 0.0) * out["co2_score"]
        )
        out = out.sort_values(["utility", "accuracy"], ascending=[False, False]).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
        margin = float(out.loc[0, "utility"] - out.loc[1, "utility"]) if len(out) > 1 else 1.0
        evidence_distance = float(out.loc[0, "evidence_distance"]) if np.isfinite(out.loc[0, "evidence_distance"]) else None
        meta = {
            "weights": weights,
            "utility_margin": margin,
            "evidence_distance": evidence_distance,
            "neighbors": neighbor_map,
            "coverage": meta_log_coverage(self.df),
            "model_metrics": sorted(self.models.keys()),
            "note": "Predictions are learned from historical meta logs; they are estimates, not guarantees for the current stream.",
        }
        return out, meta


def profile_from_dataframe(df: pd.DataFrame, target: str) -> Dict[str, int]:
    return {
        "n_samples": int(len(df)),
        "n_features": int(max(0, df.shape[1] - 1)),
        "n_classes": int(df[target].dropna().nunique()) if target in df.columns else 0,
    }
