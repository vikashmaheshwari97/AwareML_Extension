from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from awareml.engine.pareto_spec import (
    CANONICAL_EPSILON,
    NORMALIZATION_ID,
    PARETO_SPEC_ID,
    specification_dict,
)
from awareml.recommender.v2_data import (
    EXPECTED_DATASETS,
    EXPECTED_FRAMEWORKS,
    EXPECTED_RECOMMENDER_ROWS,
    V2_TARGETS,
    load_recommender_train,
    validate_recommender_train,
)
from awareml.recommender.v2_ranking import normalize_weights


PHASE13_RELEASE_ID = "recommender_multiobjective_validation_v1"
OBJECTIVES = ("accuracy", "runtime", "energy", "co2")
DIRECTIONS = {
    "accuracy": "max",
    "runtime": "min",
    "energy": "min",
    "co2": "min",
}

PRIMARY_PREFERENCE_SCENARIOS = {
    "balanced": {
        "accuracy": 0.25,
        "runtime": 0.25,
        "energy": 0.25,
        "co2": 0.25,
    },
    "accuracy_focus": {
        "accuracy": 0.70,
        "runtime": 0.10,
        "energy": 0.10,
        "co2": 0.10,
    },
    "speed_focus": {
        "accuracy": 0.20,
        "runtime": 0.50,
        "energy": 0.15,
        "co2": 0.15,
    },
    "sustainability_focus": {
        "accuracy": 0.20,
        "runtime": 0.10,
        "energy": 0.35,
        "co2": 0.35,
    },
}

SUSTAINABILITY_SCENARIOS = {
    "energy_only": {
        "accuracy": 0.0,
        "runtime": 0.0,
        "energy": 1.0,
        "co2": 0.0,
    },
    "co2_only": {
        "accuracy": 0.0,
        "runtime": 0.0,
        "energy": 0.0,
        "co2": 1.0,
    },
    "energy_plus_co2": {
        "accuracy": 0.0,
        "runtime": 0.0,
        "energy": 0.5,
        "co2": 0.5,
    },
}

BASELINES = (
    "ml_recommender_v2",
    "historical_best_framework",
    "historical_framework_mean",
    "accuracy_only",
    "runtime_only",
    "energy_only",
    "random_selection",
)


class Phase13Error(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def phase13_dir(root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else project_root()
    return base / "data" / "journal" / PHASE13_RELEASE_ID


def results_dir(root: Optional[Path] = None) -> Path:
    return phase13_dir(root) / "results"


def frozen_dir(root: Optional[Path] = None) -> Path:
    return phase13_dir(root) / "frozen"


def input_paths(root: Optional[Path] = None) -> Dict[str, Path]:
    base = Path(root) if root is not None else project_root()
    return {
        "recommender_train": (
            base / "data" / "meta" / "snapshots" / "recommender_train_v2.parquet"
        ),
        "oof_predictions": (
            base
            / "data"
            / "meta"
            / "snapshots"
            / "recommender_v2_oof_predictions.parquet"
        ),
        "recommender_manifest": (
            base / "data" / "meta" / "models" / "recommender_v2" / "manifest.json"
        ),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not np.isfinite(v) else v
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _read_manifest(root: Optional[Path] = None) -> Dict[str, object]:
    paths = input_paths(root)
    path = paths["recommender_manifest"]
    if not path.exists():
        raise Phase13Error("Missing frozen Recommender V2 manifest: {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_models(manifest: Mapping[str, object]) -> Dict[str, str]:
    models = manifest.get("models") or {}
    selected = {}
    for target in OBJECTIVES:
        entry = models.get(target) if isinstance(models, dict) else None
        if not entry:
            raise Phase13Error("Recommender manifest missing model for {}.".format(target))
        model_name = entry.get("model_name") or entry.get("name") or entry.get("model")
        if not model_name:
            raise Phase13Error(
                "Recommender manifest has no model_name for {}.".format(target)
            )
        selected[target] = str(model_name)
    return selected


def _candidate_model_names_from_oof(oof: pd.DataFrame) -> List[str]:
    names = sorted(oof["model"].astype(str).dropna().unique().tolist())
    if len(names) != 7:
        raise Phase13Error(
            "Frozen development OOF table must contain the seven candidate meta-models; got {}: {}".format(
                len(names), names
            )
        )
    return names


def _candidate_model_names(manifest: Mapping[str, object]) -> List[str]:
    for key in ("candidate_models", "candidate_model_names", "model_candidates"):
        value = manifest.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, dict):
            return sorted(str(v) for v in value)
    # Frozen Phase-6 candidate family. Kept explicit so Phase-13 validation can
    # still document the seven-model development search if an older manifest
    # omitted the convenience list.
    return [
        "frameworkmean",
        "ridge",
        "knn",
        "randomforest",
        "extratrees",
        "histgradientboosting",
        "xgboost",
    ]


def load_phase13_inputs(
    root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    paths = input_paths(root)
    for name, path in paths.items():
        if not path.exists():
            raise Phase13Error("Missing Phase-13 input {}: {}".format(name, path))

    train = pd.read_parquet(paths["recommender_train"])
    validate_recommender_train(train)
    if int(train["dataset_id"].nunique()) != EXPECTED_DATASETS:
        raise Phase13Error("Phase 13 must use exactly the 47 development datasets.")
    if len(train) != EXPECTED_RECOMMENDER_ROWS:
        raise Phase13Error("Phase 13 recommender snapshot must contain 235 rows.")

    oof = pd.read_parquet(paths["oof_predictions"])
    required = {
        "dataset_id",
        "framework",
        "target",
        "model",
        "y_true",
        "y_pred",
    }
    missing = sorted(required - set(oof.columns))
    if missing:
        raise Phase13Error("OOF prediction table missing columns: {}".format(missing))

    manifest = _read_manifest(root)
    return train.copy(), oof.copy(), manifest


def build_selected_oof_wide(
    oof: pd.DataFrame,
    manifest: Mapping[str, object],
) -> pd.DataFrame:
    selected = _selected_models(manifest)
    blocks = []
    for target in OBJECTIVES:
        model_name = selected[target]
        block = oof[
            oof["target"].astype(str).str.lower().eq(target)
            & oof["model"].astype(str).str.lower().eq(model_name.lower())
        ][["dataset_id", "framework", "y_true", "y_pred"]].copy()
        if len(block) != EXPECTED_RECOMMENDER_ROWS:
            raise Phase13Error(
                "Selected OOF block {} / {} has {} rows; expected {}.".format(
                    target,
                    model_name,
                    len(block),
                    EXPECTED_RECOMMENDER_ROWS,
                )
            )
        block = block.rename(
            columns={
                "y_true": target + "_true",
                "y_pred": target + "_pred",
            }
        )
        blocks.append(block)

    wide = blocks[0]
    for block in blocks[1:]:
        wide = wide.merge(
            block,
            on=["dataset_id", "framework"],
            how="inner",
            validate="one_to_one",
        )

    if len(wide) != EXPECTED_RECOMMENDER_ROWS:
        raise Phase13Error(
            "Selected OOF merge produced {} rows; expected {}.".format(
                len(wide), EXPECTED_RECOMMENDER_ROWS
            )
        )
    if int(wide["dataset_id"].nunique()) != EXPECTED_DATASETS:
        raise Phase13Error("Selected OOF merge must contain 47 development datasets.")
    return wide


def _unit(values: Sequence[float], maximize: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise Phase13Error("Objective values contain non-finite values.")
    if abs(hi - lo) <= 1e-12:
        return np.full(len(arr), 0.5, dtype=float)
    score = (arr - lo) / (hi - lo)
    if not maximize:
        score = 1.0 - score
    return np.clip(score, 0.0, 1.0)


def _utilities_from_wide(
    block: pd.DataFrame,
    weights: Mapping[str, float],
    suffix: str,
) -> np.ndarray:
    normalized = normalize_weights(weights)
    utility = np.zeros(len(block), dtype=float)
    for objective in OBJECTIVES:
        column = objective + suffix
        if column not in block.columns:
            raise Phase13Error("Missing utility column {}.".format(column))
        score = _unit(
            block[column].to_numpy(dtype=float),
            maximize=(DIRECTIONS[objective] == "max"),
        )
        utility += normalized[objective] * score
    return utility


def _training_target_column(objective: str) -> str:
    return V2_TARGETS[objective]


def _dataset_true_utilities(
    train: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    rows = []
    for dataset_id, block in train.groupby("dataset_id", sort=True):
        block = block.copy().reset_index(drop=True)
        normalized = normalize_weights(weights)
        utility = np.zeros(len(block), dtype=float)
        for objective in OBJECTIVES:
            values = pd.to_numeric(
                block[_training_target_column(objective)], errors="raise"
            ).to_numpy(dtype=float)
            utility += normalized[objective] * _unit(
                values,
                maximize=(DIRECTIONS[objective] == "max"),
            )
        for index, row in block.iterrows():
            rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "framework": str(row["framework"]),
                    "utility": float(utility[index]),
                }
            )
    return pd.DataFrame(rows)


def _historical_best_scores(
    train_fold: pd.DataFrame,
    weights: Mapping[str, float],
) -> Dict[str, float]:
    utility = _dataset_true_utilities(train_fold, weights)
    return {
        str(k): float(v)
        for k, v in utility.groupby("framework")["utility"].mean().to_dict().items()
    }


def _historical_mean_utility_scores(
    train_fold: pd.DataFrame,
    weights: Mapping[str, float],
) -> Dict[str, float]:
    frameworks = sorted(train_fold["framework"].astype(str).unique().tolist())
    summary = pd.DataFrame({"framework": frameworks})
    grouped = train_fold.groupby("framework", sort=False)
    normalized = normalize_weights(weights)
    total = np.zeros(len(summary), dtype=float)
    for objective in OBJECTIVES:
        means = grouped[_training_target_column(objective)].mean().to_dict()
        raw = np.asarray([float(means[framework]) for framework in frameworks])
        total += normalized[objective] * _unit(
            raw,
            maximize=(DIRECTIONS[objective] == "max"),
        )
    return {
        framework: float(total[index])
        for index, framework in enumerate(frameworks)
    }


def _single_objective_scores(
    train_fold: pd.DataFrame,
    objective: str,
) -> Dict[str, float]:
    means = (
        train_fold.groupby("framework")[_training_target_column(objective)]
        .mean()
        .to_dict()
    )
    frameworks = sorted(str(k) for k in means)
    raw = np.asarray([float(means[framework]) for framework in frameworks])
    unit = _unit(raw, maximize=(DIRECTIONS[objective] == "max"))
    return {
        framework: float(unit[index])
        for index, framework in enumerate(frameworks)
    }


def _stable_random_scores(
    frameworks: Sequence[str],
    dataset_id: str,
    scenario: str,
    seed: int = 42,
) -> Dict[str, float]:
    token = "{}|{}|{}".format(seed, dataset_id, scenario).encode("utf-8")
    derived = int(hashlib.sha256(token).hexdigest()[:16], 16) % (2 ** 32)
    rng = np.random.RandomState(derived)
    order = list(frameworks)
    rng.shuffle(order)
    return {
        framework: float(len(order) - index)
        for index, framework in enumerate(order)
    }


def _score_order(
    frameworks: Sequence[str],
    scores: Mapping[str, float],
) -> List[str]:
    return sorted(
        [str(framework) for framework in frameworks],
        key=lambda name: (-float(scores[name]), name),
    )


def _safe_spearman_from_orders(
    true_order: Sequence[str],
    predicted_order: Sequence[str],
) -> Optional[float]:
    true_pos = {name: index for index, name in enumerate(true_order)}
    pred_pos = {name: index for index, name in enumerate(predicted_order)}
    names = sorted(set(true_pos) & set(pred_pos))
    if len(names) < 2:
        return None
    a = [true_pos[name] for name in names]
    b = [pred_pos[name] for name in names]
    rho = spearmanr(a, b).statistic
    if rho is None or not np.isfinite(rho):
        return None
    return float(rho)


def _evaluate_order(
    frameworks: Sequence[str],
    true_utility: np.ndarray,
    predicted_order: Sequence[str],
) -> Dict[str, object]:
    true_by_framework = {
        str(framework): float(true_utility[index])
        for index, framework in enumerate(frameworks)
    }
    true_order = sorted(
        [str(v) for v in frameworks],
        key=lambda name: (-true_by_framework[name], name),
    )
    predicted_order = [str(v) for v in predicted_order]
    selected = predicted_order[0]
    oracle = true_order[0]
    oracle_utility = true_by_framework[oracle]
    selected_utility = true_by_framework[selected]
    values = np.asarray(list(true_by_framework.values()), dtype=float)
    span = float(np.max(values) - np.min(values))
    regret = (
        0.0
        if span <= 1e-12
        else max(0.0, (oracle_utility - selected_utility) / span)
    )
    rho = _safe_spearman_from_orders(true_order, predicted_order)
    return {
        "true_best_framework": oracle,
        "selected_framework": selected,
        "ranking": "|".join(predicted_order),
        "true_ranking": "|".join(true_order),
        "top1_correct": float(selected == oracle),
        "top3_correct": float(oracle in set(predicted_order[:3])),
        "normalized_regret": float(regret),
        "selected_true_utility": float(selected_utility),
        "oracle_true_utility": float(oracle_utility),
        "ranking_spearman": rho,
    }


def _aggregate_detail(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (baseline, scenario), block in detail.groupby(
        ["baseline", "scenario"], sort=True
    ):
        rows.append(
            {
                "baseline": str(baseline),
                "scenario": str(scenario),
                "evaluations": int(len(block)),
                "top1_accuracy": float(block["top1_correct"].mean()),
                "top3_accuracy": float(block["top3_correct"].mean()),
                "normalized_regret": float(block["normalized_regret"].mean()),
                "mean_selected_utility": float(block["selected_true_utility"].mean()),
                "mean_oracle_utility": float(block["oracle_true_utility"].mean()),
                "mean_ranking_spearman": float(
                    pd.to_numeric(block["ranking_spearman"], errors="coerce").mean()
                ),
            }
        )
    per_scenario = pd.DataFrame(rows)

    overall_rows = []
    for baseline, block in detail.groupby("baseline", sort=True):
        overall_rows.append(
            {
                "baseline": str(baseline),
                "scenario": "OVERALL",
                "evaluations": int(len(block)),
                "top1_accuracy": float(block["top1_correct"].mean()),
                "top3_accuracy": float(block["top3_correct"].mean()),
                "normalized_regret": float(block["normalized_regret"].mean()),
                "mean_selected_utility": float(block["selected_true_utility"].mean()),
                "mean_oracle_utility": float(block["oracle_true_utility"].mean()),
                "mean_ranking_spearman": float(
                    pd.to_numeric(block["ranking_spearman"], errors="coerce").mean()
                ),
            }
        )
    return pd.concat(
        [per_scenario, pd.DataFrame(overall_rows)],
        ignore_index=True,
    )


def _baseline_conclusion(table: pd.DataFrame) -> Dict[str, object]:
    overall = table[table["scenario"].eq("OVERALL")].copy()
    ml = overall[overall["baseline"].eq("ml_recommender_v2")]
    simple = overall[~overall["baseline"].eq("ml_recommender_v2")].copy()
    if len(ml) != 1 or simple.empty:
        raise Phase13Error("Cannot derive baseline conclusion.")
    simple = simple.sort_values(
        ["normalized_regret", "top1_accuracy", "baseline"],
        ascending=[True, False, True],
    )
    best = simple.iloc[0]
    ml_row = ml.iloc[0]
    regret_delta = float(best["normalized_regret"] - ml_row["normalized_regret"])
    top1_delta = float(ml_row["top1_accuracy"] - best["top1_accuracy"])

    if regret_delta > 1e-12 and top1_delta >= -1e-12:
        verdict = "YES"
        interpretation = (
            "ML Recommender V2 has lower normalized regret than the strongest "
            "simple baseline without sacrificing Top-1 accuracy."
        )
    elif regret_delta < -1e-12 and top1_delta <= 1e-12:
        verdict = "NO"
        interpretation = (
            "A simple baseline matches or exceeds ML Recommender V2 on both "
            "normalized regret and Top-1 accuracy."
        )
    else:
        verdict = "MIXED"
        interpretation = (
            "ML Recommender V2 and the strongest simple baseline trade off "
            "Top-1 accuracy and normalized regret; report both metrics."
        )

    return {
        "question": "Does ML Recommender V2 outperform simple selection rules?",
        "verdict": verdict,
        "best_simple_baseline": str(best["baseline"]),
        "ml_top1_accuracy": float(ml_row["top1_accuracy"]),
        "best_simple_top1_accuracy": float(best["top1_accuracy"]),
        "top1_delta_ml_minus_simple": top1_delta,
        "ml_normalized_regret": float(ml_row["normalized_regret"]),
        "best_simple_normalized_regret": float(best["normalized_regret"]),
        "normalized_regret_improvement_simple_minus_ml": regret_delta,
        "interpretation": interpretation,
    }


def run_baseline_validation(
    root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    random_seed: int = 42,
) -> Dict[str, object]:
    train, oof, manifest = load_phase13_inputs(root)
    wide = build_selected_oof_wide(oof, manifest)
    selected_models = _selected_models(manifest)

    detail_rows = []
    for scenario, raw_weights in PRIMARY_PREFERENCE_SCENARIOS.items():
        weights = normalize_weights(raw_weights)
        for dataset_id, test_block in wide.groupby("dataset_id", sort=True):
            dataset_id = str(dataset_id)
            test_block = test_block.copy().reset_index(drop=True)
            frameworks = test_block["framework"].astype(str).tolist()
            if len(frameworks) != len(EXPECTED_FRAMEWORKS):
                raise Phase13Error(
                    "Dataset {} does not contain five framework candidates.".format(
                        dataset_id
                    )
                )

            true_utility = _utilities_from_wide(test_block, weights, "_true")
            pred_utility = _utilities_from_wide(test_block, weights, "_pred")
            ml_scores = {
                framework: float(pred_utility[index])
                for index, framework in enumerate(frameworks)
            }

            train_fold = train[~train["dataset_id"].astype(str).eq(dataset_id)].copy()
            if int(train_fold["dataset_id"].nunique()) != EXPECTED_DATASETS - 1:
                raise Phase13Error("LODO baseline fold is not leakage-free.")

            score_sets = {
                "ml_recommender_v2": ml_scores,
                "historical_best_framework": _historical_best_scores(
                    train_fold, weights
                ),
                "historical_framework_mean": _historical_mean_utility_scores(
                    train_fold, weights
                ),
                "accuracy_only": _single_objective_scores(train_fold, "accuracy"),
                "runtime_only": _single_objective_scores(train_fold, "runtime"),
                "energy_only": _single_objective_scores(train_fold, "energy"),
                "random_selection": _stable_random_scores(
                    frameworks,
                    dataset_id=dataset_id,
                    scenario=scenario,
                    seed=random_seed,
                ),
            }

            for baseline in BASELINES:
                order = _score_order(frameworks, score_sets[baseline])
                metrics = _evaluate_order(frameworks, true_utility, order)
                detail_rows.append(
                    {
                        "baseline": baseline,
                        "scenario": scenario,
                        "dataset_id": dataset_id,
                        **metrics,
                    }
                )

    detail = pd.DataFrame(detail_rows)
    table = _aggregate_detail(detail)
    conclusion = _baseline_conclusion(table)

    out = Path(output_dir) if output_dir is not None else results_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "baseline_comparison_detail.csv", index=False)
    table.to_csv(out / "baseline_comparison_table.csv", index=False)
    _write_json(out / "baseline_conclusion.json", _json_safe(conclusion))

    candidate_models = _candidate_model_names_from_oof(oof)
    metadata = {
        "release_id": PHASE13_RELEASE_ID,
        "analysis": "explicit_recommender_baselines",
        "development_dataset_count": int(train["dataset_id"].nunique()),
        "framework_count": int(train["framework"].nunique()),
        "evaluation_protocol": "leave-one-development-dataset-out",
        "held_out_23_dataset_contents_used": False,
        "selected_models": selected_models,
        "candidate_model_family": candidate_models,
        "candidate_model_count": len(candidate_models),
        "baselines": list(BASELINES),
        "preference_scenarios": PRIMARY_PREFERENCE_SCENARIOS,
        "random_seed": int(random_seed),
        "conclusion": conclusion,
    }
    _write_json(out / "baseline_validation_metadata.json", _json_safe(metadata))
    return {
        "table": table,
        "detail": detail,
        "conclusion": conclusion,
        "metadata": metadata,
    }


def _ranking_positions(ranking: str) -> Dict[str, int]:
    names = [value for value in str(ranking).split("|") if value]
    return {name: index for index, name in enumerate(names)}


def _kendall_between_rankings(a: str, b: str) -> Optional[float]:
    pa = _ranking_positions(a)
    pb = _ranking_positions(b)
    names = sorted(set(pa) & set(pb))
    if len(names) < 2:
        return None
    result = kendalltau(
        [pa[name] for name in names],
        [pb[name] for name in names],
    ).statistic
    if result is None or not np.isfinite(result):
        return None
    return float(result)


def run_energy_co2_sensitivity(
    root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    train, oof, manifest = load_phase13_inputs(root)
    wide = build_selected_oof_wide(oof, manifest)

    energy_values = pd.to_numeric(train[V2_TARGETS["energy"]], errors="raise")
    co2_values = pd.to_numeric(train[V2_TARGETS["co2"]], errors="raise")
    rho = spearmanr(energy_values, co2_values).statistic
    rho_value = float(rho) if rho is not None and np.isfinite(rho) else None

    detail_rows = []
    for scenario, raw_weights in SUSTAINABILITY_SCENARIOS.items():
        weights = normalize_weights(raw_weights)
        for dataset_id, block in wide.groupby("dataset_id", sort=True):
            block = block.copy().reset_index(drop=True)
            frameworks = block["framework"].astype(str).tolist()
            true_utility = _utilities_from_wide(block, weights, "_true")
            pred_utility = _utilities_from_wide(block, weights, "_pred")
            scores = {
                framework: float(pred_utility[index])
                for index, framework in enumerate(frameworks)
            }
            order = _score_order(frameworks, scores)
            metrics = _evaluate_order(frameworks, true_utility, order)

            canonical_weights = SUSTAINABILITY_SCENARIOS["energy_plus_co2"]
            canonical_true = _utilities_from_wide(
                block, canonical_weights, "_true"
            )
            canonical_map = {
                framework: float(canonical_true[index])
                for index, framework in enumerate(frameworks)
            }
            metrics["canonical_sustainability_utility"] = canonical_map[
                str(metrics["selected_framework"])
            ]
            detail_rows.append(
                {
                    "scenario": scenario,
                    "dataset_id": str(dataset_id),
                    **metrics,
                }
            )

    detail = pd.DataFrame(detail_rows)
    summary_rows = []
    for scenario, block in detail.groupby("scenario", sort=True):
        summary_rows.append(
            {
                "scenario": str(scenario),
                "datasets": int(len(block)),
                "top1_accuracy": float(block["top1_correct"].mean()),
                "top3_accuracy": float(block["top3_correct"].mean()),
                "normalized_regret": float(block["normalized_regret"].mean()),
                "mean_selected_true_utility": float(
                    block["selected_true_utility"].mean()
                ),
                "mean_canonical_sustainability_utility": float(
                    block["canonical_sustainability_utility"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    by_scenario = {
        scenario: block.set_index("dataset_id")
        for scenario, block in detail.groupby("scenario", sort=False)
    }
    comparisons = [
        ("energy_only", "co2_only"),
        ("energy_only", "energy_plus_co2"),
        ("co2_only", "energy_plus_co2"),
    ]
    pair_rows = []
    for left, right in comparisons:
        a = by_scenario[left]
        b = by_scenario[right]
        ids = sorted(set(a.index) & set(b.index))
        top1_changes = []
        ranking_changes = []
        taus = []
        utility_deltas = []
        regret_deltas = []
        for dataset_id in ids:
            row_a = a.loc[dataset_id]
            row_b = b.loc[dataset_id]
            top1_changes.append(
                float(row_a["selected_framework"] != row_b["selected_framework"])
            )
            ranking_changes.append(float(row_a["ranking"] != row_b["ranking"]))
            tau = _kendall_between_rankings(row_a["ranking"], row_b["ranking"])
            if tau is not None:
                taus.append(tau)
            utility_deltas.append(
                float(
                    row_b["canonical_sustainability_utility"]
                    - row_a["canonical_sustainability_utility"]
                )
            )
            regret_deltas.append(
                float(row_b["normalized_regret"] - row_a["normalized_regret"])
            )
        pair_rows.append(
            {
                "comparison": "{}_vs_{}".format(left, right),
                "left": left,
                "right": right,
                "datasets": int(len(ids)),
                "top1_change_count": int(sum(top1_changes)),
                "top1_change_rate": float(np.mean(top1_changes)),
                "full_ranking_change_count": int(sum(ranking_changes)),
                "full_ranking_change_rate": float(np.mean(ranking_changes)),
                "mean_kendall_tau": float(np.mean(taus)) if taus else np.nan,
                "mean_canonical_utility_delta_right_minus_left": float(
                    np.mean(utility_deltas)
                ),
                "mean_normalized_regret_delta_right_minus_left": float(
                    np.mean(regret_deltas)
                ),
            }
        )
    pairwise = pd.DataFrame(pair_rows)

    e_vs_c = pairwise[pairwise["comparison"].eq("energy_only_vs_co2_only")].iloc[0]
    high_corr = rho_value is not None and abs(rho_value) >= 0.95
    rank_equivalent = (
        float(e_vs_c["top1_change_rate"]) <= 0.10
        and float(e_vs_c["mean_kendall_tau"]) >= 0.90
    )

    if high_corr and rank_equivalent:
        decision_code = "retain_semantics_avoid_double_counting"
        journal_decision = (
            "Retain Energy and CO2 as distinct measurable and user-selectable "
            "objectives, but do not treat simultaneous Energy+CO2 weighting as "
            "independent evidence in the primary journal recommender claim. "
            "Report the sensitivity analysis and retain the double-counting warning."
        )
    else:
        decision_code = "retain_both_independent"
        journal_decision = (
            "Retain Energy and CO2 as independent objectives in the primary journal "
            "formulation because the development sensitivity analysis shows material "
            "ranking differences."
        )

    decision = {
        "development_spearman_energy_co2": rho_value,
        "high_correlation_threshold": 0.95,
        "high_correlation": bool(high_corr),
        "energy_vs_co2_top1_change_rate": float(e_vs_c["top1_change_rate"]),
        "energy_vs_co2_mean_kendall_tau": float(e_vs_c["mean_kendall_tau"]),
        "rank_equivalent_rule": "top1 change <= 0.10 and mean Kendall tau >= 0.90",
        "rank_equivalent": bool(rank_equivalent),
        "decision_code": decision_code,
        "journal_decision": journal_decision,
        "held_out_23_dataset_contents_used": False,
    }

    out = Path(output_dir) if output_dir is not None else results_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "energy_co2_sensitivity_detail.csv", index=False)
    summary.to_csv(out / "energy_co2_sensitivity_report.csv", index=False)
    pairwise.to_csv(out / "energy_co2_ranking_changes.csv", index=False)
    _write_json(out / "energy_co2_decision.json", _json_safe(decision))

    return {
        "detail": detail,
        "summary": summary,
        "pairwise": pairwise,
        "decision": decision,
    }


def write_near_pareto_specification(
    root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    out = Path(output_dir) if output_dir is not None else results_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    spec = specification_dict(CANONICAL_EPSILON)
    spec.update(
        {
            "journal_scope": [
                "ML Recommender V2 / 3D Decision Space",
                "observed post-run Decision Lab",
                "journal paper/reporting",
            ],
            "legacy_column_note": (
                "The recommender keeps pareto_efficient as a backward-compatible "
                "column alias, but from Phase 13 ranking outputs it denotes the same "
                "canonical epsilon-Pareto membership as near_pareto."
            ),
        }
    )
    _write_json(out / "near_pareto_specification.json", spec)

    markdown = "# Canonical near-Pareto specification\n\n"
    markdown += "**Specification ID:** `{}`  \n".format(PARETO_SPEC_ID)
    markdown += "**Canonical epsilon:** `{:.2f}`  \n".format(CANONICAL_EPSILON)
    markdown += "**Normalization:** `{}`\n\n".format(NORMALIZATION_ID)
    markdown += "## Mathematical definition\n\n"
    markdown += (
        "Let `z_i in [0,1]^m` be the normalized desirability vector for candidate "
        "`i`, after converting every objective to higher-is-better. Candidate `j` "
        "epsilon-dominates candidate `i` when:\n\n"
    )
    markdown += "1. `z_jk >= z_ik - epsilon` for every jointly available objective `k`; and\n"
    markdown += "2. `z_jk > z_ik + epsilon` for at least one objective `k`.\n\n"
    markdown += (
        "Candidate `i` is **epsilon-Pareto / near-Pareto** when no other candidate "
        "epsilon-dominates it. The journal uses `epsilon = 0.05`. At `epsilon = 0` "
        "the relation reduces to ordinary Pareto nondominance.\n\n"
    )
    markdown += "## Canonical use\n\n"
    markdown += "- Pre-run ML Recommender V2 / 3D Decision Space: epsilon = 0.05.\n"
    markdown += "- Post-run Decision Lab: epsilon = 0.05 for journal reporting.\n"
    markdown += "- Paper: report this definition and epsilon exactly.\n"
    markdown += "- Sensitivity views may vary epsilon, but must be labelled as sensitivity analysis.\n"
    (out / "near_pareto_specification.md").write_text(markdown, encoding="utf-8")
    return spec


def _required_result_files(root: Optional[Path] = None) -> List[Path]:
    out = results_dir(root)
    return [
        out / "baseline_comparison_table.csv",
        out / "baseline_comparison_detail.csv",
        out / "baseline_conclusion.json",
        out / "baseline_validation_metadata.json",
        out / "energy_co2_sensitivity_report.csv",
        out / "energy_co2_sensitivity_detail.csv",
        out / "energy_co2_ranking_changes.csv",
        out / "energy_co2_decision.json",
        out / "near_pareto_specification.json",
        out / "near_pareto_specification.md",
    ]


def freeze_phase13_validation(root: Optional[Path] = None) -> Dict[str, object]:
    base = Path(root) if root is not None else project_root()
    missing = [str(path) for path in _required_result_files(base) if not path.exists()]
    if missing:
        raise Phase13Error(
            "Phase-13 outputs are incomplete before freeze: {}".format(missing)
        )

    baseline_table = pd.read_csv(results_dir(base) / "baseline_comparison_table.csv")
    if set(baseline_table["baseline"].astype(str).unique()) != set(BASELINES):
        raise Phase13Error("Baseline table does not contain all seven Phase-13 methods.")
    if set(PRIMARY_PREFERENCE_SCENARIOS) - set(
        baseline_table["scenario"].astype(str).unique()
    ):
        raise Phase13Error("Baseline table is missing preference scenarios.")

    energy_decision = json.loads(
        (results_dir(base) / "energy_co2_decision.json").read_text(encoding="utf-8")
    )
    baseline_conclusion = json.loads(
        (results_dir(base) / "baseline_conclusion.json").read_text(encoding="utf-8")
    )
    pareto_spec = json.loads(
        (results_dir(base) / "near_pareto_specification.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_v2 = _read_manifest(base)
    baseline_metadata = json.loads(
        (results_dir(base) / "baseline_validation_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    result_hashes = {
        path.name: sha256_file(path) for path in _required_result_files(base)
    }
    source_paths = {
        "canonical_pareto_spec_code": base / "awareml" / "engine" / "pareto_spec.py",
        "post_run_pareto_code": base / "awareml" / "engine" / "pareto.py",
        "recommender_ranking_code": base / "awareml" / "recommender" / "v2_ranking.py",
        "phase13_validation_code": (
            base / "awareml" / "journal" / "recommender_validation.py"
        ),
    }
    for path in source_paths.values():
        if not path.exists():
            raise Phase13Error("Missing Phase-13 source file: {}".format(path))

    inputs = input_paths(base)
    manifest = {
        "artifact": PHASE13_RELEASE_ID,
        "phase": 13,
        "release_status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_dataset_count": EXPECTED_DATASETS,
        "framework_count": len(EXPECTED_FRAMEWORKS),
        "development_protocol": "47-dataset leave-one-dataset-out validation",
        "held_out_23_dataset_contents_used": False,
        "input_artifacts": {
            name: {
                "path": str(path.relative_to(base)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
            for name, path in inputs.items()
        },
        "source_code": {
            name: {
                "path": str(path.relative_to(base)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
            for name, path in source_paths.items()
        },
        "candidate_meta_models": baseline_metadata["candidate_model_family"],
        "candidate_meta_model_count": int(baseline_metadata["candidate_model_count"]),
        "selected_meta_models": _selected_models(manifest_v2),
        "baseline_methods": list(BASELINES),
        "primary_preference_scenarios": PRIMARY_PREFERENCE_SCENARIOS,
        "baseline_conclusion": baseline_conclusion,
        "energy_co2_decision": energy_decision,
        "near_pareto": pareto_spec,
        "result_hashes": result_hashes,
    }

    frozen = frozen_dir(base)
    frozen.mkdir(parents=True, exist_ok=True)
    manifest_path = frozen / "manifest.json"
    _write_json(manifest_path, _json_safe(manifest))
    digest = sha256_file(manifest_path)
    (frozen / "manifest.json.sha256").write_text(
        "{}  manifest.json\n".format(digest), encoding="utf-8"
    )
    marker = base / "data" / "journal" / "active_recommender_validation.txt"
    marker.write_text(
        "{}/frozen/manifest.json\n".format(PHASE13_RELEASE_ID),
        encoding="utf-8",
    )
    return {"manifest": manifest_path, "sha256": digest, "payload": manifest}


def validate_phase13_complete(root: Optional[Path] = None) -> Dict[str, object]:
    base = Path(root) if root is not None else project_root()
    marker = base / "data" / "journal" / "active_recommender_validation.txt"
    if not marker.exists():
        raise Phase13Error("Phase-13 active artifact marker is missing.")
    rel = marker.read_text(encoding="utf-8").strip()
    manifest_path = base / "data" / "journal" / rel
    if not manifest_path.exists():
        raise Phase13Error("Frozen Phase-13 manifest is missing: {}".format(manifest_path))

    checksum_path = manifest_path.parent / "manifest.json.sha256"
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(manifest_path)
    if actual != expected:
        raise Phase13Error("Frozen Phase-13 manifest checksum mismatch.")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("release_status") != "frozen":
        raise Phase13Error("Phase-13 release is not frozen.")
    if int(payload.get("development_dataset_count", -1)) != EXPECTED_DATASETS:
        raise Phase13Error("Phase-13 frozen dataset count is not 47.")
    if bool(payload.get("held_out_23_dataset_contents_used")):
        raise Phase13Error("Phase 13 must not consume the reserved 23-dataset test split.")
    if int(payload.get("candidate_meta_model_count", -1)) != 7:
        raise Phase13Error("Phase 13 must preserve the seven development meta-models.")
    if set(payload.get("baseline_methods") or []) != set(BASELINES):
        raise Phase13Error("Frozen Phase-13 baseline set is incomplete.")

    near = payload.get("near_pareto") or {}
    if near.get("spec_id") != PARETO_SPEC_ID:
        raise Phase13Error("Frozen near-Pareto specification ID is incorrect.")
    if abs(float(near.get("epsilon", -1.0)) - CANONICAL_EPSILON) > 1e-12:
        raise Phase13Error("Frozen near-Pareto epsilon is not 0.05.")

    for name, entry in (payload.get("input_artifacts") or {}).items():
        path = base / str(entry["path"])
        if sha256_file(path) != str(entry["sha256"]):
            raise Phase13Error("Frozen input changed: {}".format(name))
    for name, entry in (payload.get("source_code") or {}).items():
        path = base / str(entry["path"])
        if sha256_file(path) != str(entry["sha256"]):
            raise Phase13Error("Frozen source changed: {}".format(name))
    for filename, digest in (payload.get("result_hashes") or {}).items():
        path = results_dir(base) / filename
        if sha256_file(path) != str(digest):
            raise Phase13Error("Frozen Phase-13 result changed: {}".format(filename))

    return {
        "artifact": payload["artifact"],
        "sha256": actual,
        "development_dataset_count": payload["development_dataset_count"],
        "candidate_meta_model_count": payload["candidate_meta_model_count"],
        "baseline_conclusion": payload["baseline_conclusion"],
        "energy_co2_decision": payload["energy_co2_decision"],
        "near_pareto": payload["near_pareto"],
        "held_out_23_dataset_contents_used": False,
        "release_status": payload["release_status"],
    }
