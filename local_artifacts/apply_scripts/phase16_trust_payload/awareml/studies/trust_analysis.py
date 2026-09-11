from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .trust import Phase16Error, Phase16Store, load_phase16_protocol


ANALYSIS_VERSION = "phase16_trust_analysis_v1"


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _mean(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return None if arr.size == 0 else float(arr.mean())


def _sd(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return None if arr.size < 2 else float(arr.std(ddof=1))


def _rate(mask: pd.Series) -> Optional[float]:
    if len(mask) == 0:
        return None
    return float(mask.astype(float).mean())


def _participant_bootstrap_gap(
    participant_gaps: pd.Series,
    n_bootstrap: int = 10000,
    seed: int = 16042,
) -> Dict[str, Any]:
    gaps = pd.to_numeric(participant_gaps, errors="coerce").dropna().to_numpy(dtype=float)
    if len(gaps) == 0:
        return {"n_participants": 0, "mean_gap": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        means[i] = float(rng.choice(gaps, size=len(gaps), replace=True).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_participants": int(len(gaps)),
        "mean_gap": float(gaps.mean()),
        "sd_gap": float(gaps.std(ddof=1)) if len(gaps) > 1 else None,
        "ci95": [float(lo), float(hi)],
        "bootstrap_iterations": int(n_bootstrap),
        "bootstrap_seed": int(seed),
    }


def _paired_sign_flip_permutation(
    participant_gaps: pd.Series,
    n_permutations: int = 50000,
    seed: int = 16043,
) -> Dict[str, Any]:
    gaps = pd.to_numeric(participant_gaps, errors="coerce").dropna().to_numpy(dtype=float)
    if len(gaps) == 0:
        return {"n_participants": 0, "observed_mean_gap": None, "p_two_sided": None}
    observed = abs(float(gaps.mean()))
    if np.allclose(gaps, 0.0):
        return {
            "n_participants": int(len(gaps)),
            "observed_mean_gap": 0.0,
            "p_two_sided": 1.0,
            "permutations": 0,
            "method": "paired_sign_flip",
        }
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(gaps), replace=True)
        permuted = abs(float((gaps * signs).mean()))
        if permuted >= observed - 1e-15:
            extreme += 1
    p = (extreme + 1.0) / (n_permutations + 1.0)
    return {
        "n_participants": int(len(gaps)),
        "observed_mean_gap": float(gaps.mean()),
        "p_two_sided": float(p),
        "permutations": int(n_permutations),
        "seed": int(seed),
        "method": "paired_sign_flip",
    }


def _spearman(frame: pd.DataFrame, x: str, y: str) -> Dict[str, Any]:
    temp = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(temp) < 3 or temp[x].nunique() < 2 or temp[y].nunique() < 2:
        return {"n": int(len(temp)), "rho": None, "p": None}
    rho, p = spearmanr(temp[x], temp[y])
    return {"n": int(len(temp)), "rho": _finite_or_none(rho), "p": _finite_or_none(p)}


def _within_participant_fixed_effects(frame: pd.DataFrame) -> Dict[str, Any]:
    """Participant-centered OLS for the central RQ without a statsmodels dependency.

    Outcome and continuous predictors are centered within participant. Correctness is a
    within-participant binary predictor and is centered the same way. Standardized beta
    coefficients make correctness, fluency and perceived confidence directly comparable.
    This is a supportive model; the primary predeclared contrast remains the paired trust gap.
    """

    required = [
        "participant_hash",
        "trust_rating",
        "correctness_binary",
        "fluency_rating",
        "perceived_confidence",
        "order_index",
    ]
    work = frame[required].copy()
    for col in required[1:]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna()
    if work.empty or work["participant_hash"].nunique() < 2:
        return {"n_rows": int(len(work)), "n_participants": int(work["participant_hash"].nunique()), "betas": {}}

    predictors = ["correctness_binary", "fluency_rating", "perceived_confidence", "order_index"]
    group = work.groupby("participant_hash", sort=False)
    y = work["trust_rating"] - group["trust_rating"].transform("mean")
    centered = pd.DataFrame(index=work.index)
    for col in predictors:
        centered[col] = work[col] - group[col].transform("mean")

    y_sd = float(y.std(ddof=0))
    if not math.isfinite(y_sd) or y_sd <= 0:
        return {
            "n_rows": int(len(work)),
            "n_participants": int(work["participant_hash"].nunique()),
            "betas": {},
            "note": "Trust outcome has no within-participant variance.",
        }
    y_std = y.to_numpy(dtype=float) / y_sd
    X_cols: List[np.ndarray] = []
    kept: List[str] = []
    predictor_sds: Dict[str, float] = {}
    for col in predictors:
        sd = float(centered[col].std(ddof=0))
        predictor_sds[col] = sd
        if math.isfinite(sd) and sd > 0:
            X_cols.append(centered[col].to_numpy(dtype=float) / sd)
            kept.append(col)
    if not X_cols:
        return {"n_rows": int(len(work)), "n_participants": int(work["participant_hash"].nunique()), "betas": {}}
    X = np.column_stack(X_cols)
    betas, residuals, rank, singular = np.linalg.lstsq(X, y_std, rcond=None)
    fitted = X.dot(betas)
    ss_res = float(np.sum((y_std - fitted) ** 2))
    ss_tot = float(np.sum(y_std ** 2))
    r2 = None if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return {
        "n_rows": int(len(work)),
        "n_participants": int(work["participant_hash"].nunique()),
        "betas": {name: float(beta) for name, beta in zip(kept, betas)},
        "within_participant_r2": r2,
        "matrix_rank": int(rank),
        "predictor_within_sd": predictor_sds,
        "interpretation": (
            "Standardized participant-centered coefficients. correctness_binary estimates the trust shift "
            "for actually correct vs incorrect explanations after accounting for reported fluency, "
            "perceived confidence and order."
        ),
    }


def _validate_complete_participants(
    participants: pd.DataFrame,
    responses: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    if participants.empty:
        return participants.copy(), responses.iloc[0:0].copy(), issues
    complete = participants[participants["completed_at"].notna()].copy()
    complete_hashes = set(complete["participant_hash"].astype(str))
    response_complete = responses[responses["participant_hash"].astype(str).isin(complete_hashes)].copy()

    valid_hashes: List[str] = []
    for _, participant in complete.iterrows():
        ph = str(participant["participant_hash"])
        expected = int(participant["n_items"])
        rows = response_complete[response_complete["participant_hash"].astype(str) == ph]
        correct = int((rows["correctness_condition"] == "correct").sum()) if not rows.empty else 0
        incorrect = int((rows["correctness_condition"] == "incorrect").sum()) if not rows.empty else 0
        orders = rows["order_index"].tolist() if not rows.empty else []
        ok = (
            len(rows) == expected
            and correct == expected // 2
            and incorrect == expected // 2
            and sorted(int(x) for x in orders) == list(range(1, expected + 1))
        )
        if ok:
            valid_hashes.append(ph)
        else:
            issues.append(
                {
                    "participant_hash": ph,
                    "expected_rows": expected,
                    "observed_rows": int(len(rows)),
                    "correct": correct,
                    "incorrect": incorrect,
                    "orders": [int(x) for x in orders],
                }
            )
    valid_participants = complete[complete["participant_hash"].astype(str).isin(valid_hashes)].copy()
    valid_responses = response_complete[
        response_complete["participant_hash"].astype(str).isin(valid_hashes)
    ].copy()
    return valid_participants, valid_responses, issues


def analyze_phase16_frames(
    participant_rows: Iterable[Dict[str, Any]],
    response_rows: Iterable[Dict[str, Any]],
    protocol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    protocol = protocol or load_phase16_protocol()
    participants = pd.DataFrame(list(participant_rows))
    responses = pd.DataFrame(list(response_rows))
    if participants.empty or responses.empty:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "status": "insufficient_data",
            "participants_registered": int(len(participants)),
            "responses": int(len(responses)),
        }

    valid_participants, frame, completeness_issues = _validate_complete_participants(participants, responses)
    if frame.empty:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "status": "no_complete_participants",
            "participants_registered": int(len(participants)),
            "participants_complete_valid": 0,
            "responses": int(len(responses)),
            "completeness_issues": completeness_issues,
        }

    numeric_cols = [
        "trust_rating",
        "perceived_correctness",
        "fluency_rating",
        "perceived_confidence",
        "response_time_sec",
        "order_index",
        "accepted",
    ]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["correctness_binary"] = (frame["correctness_condition"] == "correct").astype(float)

    condition_summary: Dict[str, Any] = {}
    for condition in ("correct", "incorrect"):
        group = frame[frame["correctness_condition"] == condition]
        condition_summary[condition] = {
            "n_trials": int(len(group)),
            "mean_trust": _mean(group["trust_rating"].dropna().tolist()),
            "sd_trust": _sd(group["trust_rating"].dropna().tolist()),
            "acceptance_rate": _rate(group["accepted"].dropna() > 0.5),
            "mean_perceived_correctness": _mean(group["perceived_correctness"].dropna().tolist()),
            "mean_fluency": _mean(group["fluency_rating"].dropna().tolist()),
            "mean_perceived_confidence": _mean(group["perceived_confidence"].dropna().tolist()),
            "mean_response_time_sec": _mean(group["response_time_sec"].dropna().tolist()),
        }

    pivot = frame.pivot_table(
        index="participant_hash",
        columns="correctness_condition",
        values="trust_rating",
        aggfunc="mean",
    )
    if "correct" not in pivot.columns:
        pivot["correct"] = np.nan
    if "incorrect" not in pivot.columns:
        pivot["incorrect"] = np.nan
    pivot["trust_gap_correct_minus_incorrect"] = pivot["correct"] - pivot["incorrect"]
    gaps = pivot["trust_gap_correct_minus_incorrect"].dropna()

    primary = {
        "estimand": "within-participant mean trust(correct) - mean trust(incorrect)",
        "paired_gap": _participant_bootstrap_gap(gaps),
        "paired_permutation_test": _paired_sign_flip_permutation(gaps),
        "direction": "positive values indicate better trust calibration",
    }

    incorrect = frame[frame["correctness_condition"] == "incorrect"].copy()
    correct = frame[frame["correctness_condition"] == "correct"].copy()
    overtrust = {
        "definition": "Accepting a known-incorrect explanation/recommendation stimulus",
        "incorrect_trials": int(len(incorrect)),
        "incorrect_acceptance_count": int((incorrect["accepted"] > 0.5).sum()),
        "overtrust_rate": _rate(incorrect["accepted"] > 0.5),
        "incorrect_override_rate": _rate(incorrect["decision_action"] == "Override"),
        "incorrect_reject_rate": _rate(incorrect["decision_action"] == "Reject"),
    }
    undertrust = {
        "definition": "Overriding or rejecting a known-correct explanation/recommendation stimulus",
        "correct_trials": int(len(correct)),
        "correct_nonaccept_count": int((correct["accepted"] <= 0.5).sum()),
        "undertrust_rate": _rate(correct["accepted"] <= 0.5),
    }
    appropriate = ((frame["correctness_condition"] == "correct") & (frame["accepted"] > 0.5)) | (
        (frame["correctness_condition"] == "incorrect") & (frame["accepted"] <= 0.5)
    )
    reliance = {
        "appropriate_reliance_rate": _rate(appropriate),
        "appropriate_reliance_count": int(appropriate.sum()),
        "total_trials": int(len(frame)),
    }

    style = {
        "trust_vs_fluency_spearman": _spearman(frame, "trust_rating", "fluency_rating"),
        "trust_vs_perceived_confidence_spearman": _spearman(
            frame, "trust_rating", "perceived_confidence"
        ),
        "trust_vs_perceived_correctness_spearman": _spearman(
            frame, "trust_rating", "perceived_correctness"
        ),
        "participant_centered_regression": _within_participant_fixed_effects(frame),
    }

    expertise_results: Dict[str, Any] = {}
    for expertise in sorted(frame["expertise_group"].dropna().astype(str).unique()):
        group = frame[frame["expertise_group"].astype(str) == expertise]
        gpivot = group.pivot_table(
            index="participant_hash",
            columns="correctness_condition",
            values="trust_rating",
            aggfunc="mean",
        )
        if "correct" in gpivot.columns and "incorrect" in gpivot.columns:
            group_gaps = gpivot["correct"] - gpivot["incorrect"]
        else:
            group_gaps = pd.Series(dtype=float)
        incorrect_group = group[group["correctness_condition"] == "incorrect"]
        expertise_results[expertise] = {
            "participants": int(group["participant_hash"].nunique()),
            "trials": int(len(group)),
            "mean_trust_gap": _mean(group_gaps.dropna().tolist()),
            "overtrust_rate": _rate(incorrect_group["accepted"] > 0.5),
        }

    source_stage_results: Dict[str, Any] = {}
    for stage in sorted(frame["source_stage"].dropna().astype(str).unique()):
        group = frame[frame["source_stage"].astype(str) == stage]
        source_stage_results[stage] = {
            "trials": int(len(group)),
            "mean_trust_correct": _mean(
                group[group["correctness_condition"] == "correct"]["trust_rating"].dropna().tolist()
            ),
            "mean_trust_incorrect": _mean(
                group[group["correctness_condition"] == "incorrect"]["trust_rating"].dropna().tolist()
            ),
            "incorrect_acceptance_rate": _rate(
                group[group["correctness_condition"] == "incorrect"]["accepted"] > 0.5
            ),
        }

    order_effects = {
        "trust_vs_order_spearman": _spearman(frame, "trust_rating", "order_index"),
        "response_time_vs_order_spearman": _spearman(frame, "response_time_sec", "order_index"),
    }

    power = protocol.get("power_calculation") or {}
    required_n = power.get("required_completed_participants")
    completion_gate = {
        "required_completed_participants": required_n,
        "valid_completed_participants": int(valid_participants["participant_hash"].nunique()),
        "target_met": (
            bool(isinstance(required_n, int) and int(valid_participants["participant_hash"].nunique()) >= required_n)
            if required_n is not None
            else False
        ),
        "completeness_issues": completeness_issues,
    }

    result = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "ok",
        "central_rq": protocol.get("central_rq"),
        "participants_registered": int(len(participants)),
        "participants_completed_valid": int(valid_participants["participant_hash"].nunique()),
        "trials_analyzed": int(len(frame)),
        "condition_summary": condition_summary,
        "primary_calibration": primary,
        "overtrust": overtrust,
        "undertrust": undertrust,
        "reliance": reliance,
        "style_vs_correctness": style,
        "expertise_groups": expertise_results,
        "source_stages": source_stage_results,
        "order_effects": order_effects,
        "completion_gate": completion_gate,
    }
    return _json_safe(result)


def analyze_phase16_store(
    store: Phase16Store,
    collection_mode: str = "final",
    protocol_path: Optional[Path] = None,
) -> Dict[str, Any]:
    protocol = load_phase16_protocol(protocol_path)
    return analyze_phase16_frames(
        store.participant_rows(collection_mode),
        store.response_rows(collection_mode),
        protocol=protocol,
    )


def write_analysis_outputs(result: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_results.json"
    calibration_path = output_dir / "calibration_results.json"
    overtrust_path = output_dir / "overtrust_results.json"

    with analysis_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(result), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    calibration_payload = {
        "analysis_version": result.get("analysis_version"),
        "participants_completed_valid": result.get("participants_completed_valid"),
        "condition_summary": result.get("condition_summary"),
        "primary_calibration": result.get("primary_calibration"),
        "style_vs_correctness": result.get("style_vs_correctness"),
        "expertise_groups": result.get("expertise_groups"),
        "order_effects": result.get("order_effects"),
    }
    with calibration_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(calibration_payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    overtrust_payload = {
        "analysis_version": result.get("analysis_version"),
        "participants_completed_valid": result.get("participants_completed_valid"),
        "overtrust": result.get("overtrust"),
        "undertrust": result.get("undertrust"),
        "reliance": result.get("reliance"),
        "expertise_groups": result.get("expertise_groups"),
        "source_stages": result.get("source_stages"),
    }
    with overtrust_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(overtrust_payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return {
        "analysis": analysis_path,
        "calibration": calibration_path,
        "overtrust": overtrust_path,
    }
