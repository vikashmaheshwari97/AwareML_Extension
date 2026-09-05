from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from awareml.recommender.historical_preference import HistoricalPreferenceRecommender


EXPECTED_RUNS = 705
EXPECTED_DATASETS = 47
EXPECTED_FRAMEWORKS = 5
EXPECTED_SEEDS = {42, 43, 44}

SENSITIVITY_PROFILES = {
    "Accuracy only": {"accuracy": 1.0, "runtime": 0.0, "energy": 0.0, "co2": 0.0},
    "Runtime only": {"accuracy": 0.0, "runtime": 1.0, "energy": 0.0, "co2": 0.0},
    "Energy only": {"accuracy": 0.0, "runtime": 0.0, "energy": 1.0, "co2": 0.0},
    "CO2 only": {"accuracy": 0.0, "runtime": 0.0, "energy": 0.0, "co2": 1.0},
    "Balanced": {"accuracy": 0.25, "runtime": 0.25, "energy": 0.25, "co2": 0.25},
    "Accuracy focused": {"accuracy": 0.70, "runtime": 0.10, "energy": 0.10, "co2": 0.10},
    "Runtime focused": {"accuracy": 0.10, "runtime": 0.70, "energy": 0.10, "co2": 0.10},
    "Sustainability": {"accuracy": 0.20, "runtime": 0.10, "energy": 0.35, "co2": 0.35},
}


def _root(root: Optional[Path] = None) -> Path:
    return Path(root) if root is not None else Path(__file__).resolve().parents[2]


def load_v2_manifest_quality(root: Optional[Path] = None) -> Dict[str, Any]:
    path = _root(root) / "data" / "meta" / "models" / "recommender_v2" / "manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for objective, entry in (payload.get("models") or {}).items():
        bench = entry.get("benchmark_selection") or {}
        out[str(objective)] = {
            "model_name": entry.get("model_name"),
            "training_rows": entry.get("training_rows"),
            "training_datasets": entry.get("training_datasets"),
            "top1_accuracy": bench.get("top1_accuracy"),
            "top3_accuracy": bench.get("top3_accuracy"),
            "normalized_regret": bench.get("normalized_regret"),
            "spearman": bench.get("spearman"),
            "mae": bench.get("mae"),
            "rmse": bench.get("rmse"),
            "normalized_mae": bench.get("normalized_mae"),
        }
    return out


def audit_meta_logs_v2(root: Optional[Path] = None) -> Dict[str, Any]:
    path = _root(root) / "data" / "meta" / "snapshots" / "meta_logs_v2.parquet"
    if not path.exists():
        raise FileNotFoundError("Missing V2 meta-log snapshot: {}".format(path))

    df = pd.read_parquet(path)
    required = [
        "dataset_id", "framework", "seed",
        "accuracy", "runtime_sec", "energy_kwh", "co2_kg",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError("meta_logs_v2.parquet missing columns: {}".format(missing))

    dup = int(df.duplicated(["dataset_id", "framework", "seed"], keep=False).sum())
    primary_nulls = {
        c: int(df[c].isna().sum())
        for c in ["accuracy", "runtime_sec", "energy_kwh", "co2_kg"]
    }
    framework_counts = {
        str(k): int(v)
        for k, v in df["framework"].value_counts().sort_index().items()
    }
    seed_counts = {
        str(k): int(v)
        for k, v in df["seed"].value_counts().sort_index().items()
    }

    budget_values = sorted(pd.to_numeric(df.get("time_budget_sec"), errors="coerce").dropna().unique().tolist()) if "time_budget_sec" in df else []
    window_values = sorted(pd.to_numeric(df.get("window_size"), errors="coerce").dropna().unique().tolist()) if "window_size" in df else []
    max_sample_values = sorted(pd.to_numeric(df.get("max_samples_requested"), errors="coerce").dropna().unique().tolist()) if "max_samples_requested" in df else []

    near95 = None
    near99 = None
    if "time_budget_sec" in df.columns:
        runtime = pd.to_numeric(df["runtime_sec"], errors="coerce")
        budget = pd.to_numeric(df["time_budget_sec"], errors="coerce")
        valid = runtime.notna() & budget.notna() & (budget > 0)
        if valid.any():
            ratio = runtime[valid] / budget[valid]
            near95 = float((ratio >= 0.95).mean())
            near99 = float((ratio >= 0.99).mean())

    rho = None
    try:
        rho = float(df[["energy_kwh", "co2_kg"]].corr(method="spearman").loc["energy_kwh", "co2_kg"])
    except Exception:
        pass

    missing_variants = None
    for name in ("dataset_provenance__missing_fraction", "missing_fraction"):
        if name in df.columns:
            missing_variants = int(df[name].nunique(dropna=True))
            break

    core_pass = (
        len(df) == EXPECTED_RUNS
        and int(df["dataset_id"].nunique()) == EXPECTED_DATASETS
        and int(df["framework"].nunique()) == EXPECTED_FRAMEWORKS
        and set(int(x) for x in df["seed"].dropna().unique()) == EXPECTED_SEEDS
        and dup == 0
        and all(v == 0 for v in primary_nulls.values())
        and set(framework_counts.values()) == {141}
        and set(seed_counts.values()) == {235}
    )

    quality = load_v2_manifest_quality(root)
    warnings = []
    if len(budget_values) == 1:
        warnings.append(
            "time_budget_sec has only one development value. V2 cannot learn how outcomes change with different time budgets."
        )
    if len(window_values) == 1:
        warnings.append(
            "window_size has only one development value. V2 cannot learn window-size sensitivity."
        )
    if missing_variants == 1:
        warnings.append(
            "Development meta-data has no missingness variation; generalization to datasets with missing values is not directly covered."
        )
    if near95 is not None and near95 >= 0.50:
        warnings.append(
            "Many historical runtimes are close to the 60-second budget ceiling, so runtime is partly budget-censored."
        )
    if rho is not None and abs(rho) >= 0.95:
        warnings.append(
            "Energy and CO2 are highly correlated; simultaneous positive weights can emphasize the same efficiency signal twice."
        )

    runtime_q = quality.get("runtime") or {}
    try:
        if float(runtime_q.get("top1_accuracy")) < 0.50:
            warnings.append(
                "Runtime is the weakest frozen V2 objective model by LODO Top-1 accuracy ({:.3f}); runtime predictions should be interpreted more cautiously.".format(
                    float(runtime_q.get("top1_accuracy"))
                )
            )
    except Exception:
        pass

    return {
        "rows": int(len(df)),
        "datasets": int(df["dataset_id"].nunique()),
        "frameworks": int(df["framework"].nunique()),
        "seeds": sorted(int(x) for x in df["seed"].dropna().unique()),
        "duplicate_dataset_framework_seed_rows": dup,
        "primary_metric_nulls": primary_nulls,
        "framework_counts": framework_counts,
        "seed_counts": seed_counts,
        "time_budget_values": budget_values,
        "window_size_values": window_values,
        "max_samples_requested_values": max_sample_values,
        "runtime_near_budget_fraction_95": near95,
        "runtime_near_budget_fraction_99": near99,
        "energy_co2_spearman": rho,
        "missing_fraction_unique_count": missing_variants,
        "manifest_model_quality": quality,
        "warnings": warnings,
        "core_integrity_pass": bool(core_pass),
    }


def historical_winner_sensitivity() -> pd.DataFrame:
    service = HistoricalPreferenceRecommender()
    rows = []
    for profile, weights in SENSITIVITY_PROFILES.items():
        result = service.recommend(
            weights=weights,
            seed_mode=HistoricalPreferenceRecommender.STABLE,
        )
        top = result.ranking.iloc[0]
        second = result.ranking.iloc[1]
        rows.append({
            "Profile": profile,
            "Winner": str(top["framework"]),
            "Historical score": float(top["historical_utility"]),
            "Cross-dataset wins": int(top["win_count"]),
            "Runner-up": str(second["framework"]),
            "Score margin": float(top["historical_utility"] - second["historical_utility"]),
        })
    return pd.DataFrame(rows)
