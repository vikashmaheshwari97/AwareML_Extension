"""Supplementary trust pilot: behaviour and probability calibration, not Likert-as-probability."""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED = ["participant_id", "trial_id", "dataset_id", "condition", "ground_truth",
            "ai_answer", "initial_answer", "final_answer", "ai_correct_probability"]
CONDITIONS = ("answer_only", "rationale", "rationale_and_diagnostics")


def validate_responses(frame):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    out = frame.copy()
    if out.empty:
        raise ValueError("No responses to analyze.")
    if out[REQUIRED].isna().any().any():
        raise ValueError("Required response fields cannot be missing.")
    for col in REQUIRED[:-1]:
        if out[col].astype(str).str.strip().eq("").any():
            raise ValueError("Blank response identifiers or labels are not allowed.")
        out[col] = out[col].astype(str)
    if out.duplicated(["participant_id", "trial_id"]).any():
        raise ValueError("Duplicate participant/trial rows must be resolved before analysis.")
    p = pd.to_numeric(out.ai_correct_probability, errors="raise")
    if not np.isfinite(p).all() or not p.between(0, 1).all():
        raise ValueError("ai_correct_probability must be a probability in [0,1], not a Likert rating.")
    out["ai_correct_probability"] = p
    out["ai_correct"] = out.ai_answer == out.ground_truth
    out["initial_correct"] = out.initial_answer == out.ground_truth
    out["final_correct"] = out.final_answer == out.ground_truth
    out["disagreement"] = out.initial_answer != out.ai_answer
    out["adopted_ai"] = out.final_answer == out.ai_answer
    return out


def _conditional_rate(frame, eligible, outcome):
    sub = frame.loc[eligible]
    return {"value": float(sub[outcome].mean()) if len(sub) else None, "denominator": len(sub)}


def summarize_behavior(frame, bins=5):
    f = validate_responses(frame)
    if isinstance(bins, bool) or int(bins) != bins or bins < 2:
        raise ValueError("bins must be an integer >= 2.")
    p = f.ai_correct_probability.to_numpy(float)
    y = f.ai_correct.to_numpy(float)
    bin_id = np.minimum((p * bins).astype(int), bins - 1)
    reliability, ece = [], 0.0
    for i in range(bins):
        mask = bin_id == i
        n = int(mask.sum())
        if n:
            confidence, accuracy = float(p[mask].mean()), float(y[mask].mean())
            ece += n / len(f) * abs(confidence - accuracy)
            reliability.append({"bin": i, "n": n, "mean_probability": confidence, "ai_accuracy": accuracy})
    return {
        "n_trials": len(f), "n_participants": f.participant_id.nunique(),
        "initial_accuracy": float(f.initial_correct.mean()), "final_accuracy": float(f.final_correct.mean()),
        "accuracy_change": float(f.final_correct.mean() - f.initial_correct.mean()),
        "beneficial_reliance": _conditional_rate(f, ~f.initial_correct & f.ai_correct & f.disagreement, "adopted_ai"),
        "overreliance": _conditional_rate(f, ~f.ai_correct & f.disagreement, "adopted_ai"),
        "harmful_switch": _conditional_rate(f, f.initial_correct & ~f.ai_correct, "adopted_ai"),
        "appropriate_self_reliance": _conditional_rate(f, f.initial_correct & ~f.ai_correct, "final_correct"),
        "brier_ai_correctness": float(np.mean((p - y) ** 2)),
        "ece_ai_correctness": float(ece), "reliability_bins": reliability,
        "note": "Brier/ECE evaluate the participant's explicit probability that AI advice is correct. Trials are descriptive, not independent participants.",
    }


def analyze_behavior(frame):
    f = validate_responses(frame)
    by_condition = {str(name): summarize_behavior(group) for name, group in f.groupby("condition")}
    return {"schema_version": "behavioral_calibration_v1", "overall": summarize_behavior(f),
            "conditions": by_condition,
            "participant_condition": [{"participant_id": str(pid), "condition": str(cond),
                                       **summarize_behavior(g)}
                                      for (pid, cond), g in f.groupby(["participant_id", "condition"])],
            "inference": "Use participant and stimulus/dataset clustering for confirmatory comparisons; no trial-level significance test is performed here."}


def paired_participant_interval(frame, condition_a, condition_b, seed=42, draws=2000):
    """Paired participant bootstrap of final accuracy, conditional on sampled stimuli."""
    if condition_a == condition_b or draws < 100:
        raise ValueError("Choose different conditions and at least 100 draws.")
    f = validate_responses(frame)
    means = f.groupby(["participant_id", "condition"]).final_correct.mean().unstack()
    if condition_a not in means or condition_b not in means:
        raise ValueError("Both conditions are required.")
    paired = means[[condition_a, condition_b]].dropna()
    if len(paired) < 2:
        return {"status": "unavailable", "reason": "At least two paired participants required."}
    delta = (paired[condition_b] - paired[condition_a]).to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(draws)])
    return {"status": "ok", "contrast": condition_b + " minus " + condition_a,
            "mean_difference": float(delta.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "paired_participants": len(delta), "seed": seed, "draws": draws,
            "limitation": "Participant bootstrap only; does not generalize over new stimuli. Small-sample intervals are unstable."}


def validate_pilot_bank(bank):
    if bank.get("schema_version") != "xai_behavioral_pilot_v1":
        raise ValueError("Unknown pilot-bank schema.")
    cases = bank.get("cases") or []
    if not cases:
        raise ValueError("Provide reviewed study cases; no cases are synthesized by the collector.")
    seen = set()
    for c in cases:
        fields = ("case_id", "dataset_id", "question", "choices", "ground_truth", "ai_answer",
                  "rationale", "diagnostics", "task_evidence", "ground_truth_policy")
        if any(k not in c for k in fields) or c.get("reviewed") is not True:
            raise ValueError("Each case must be complete and explicitly reviewed.")
        if c["case_id"] in seen:
            raise ValueError("A participant must not see the same case twice.")
        seen.add(c["case_id"])
        if not isinstance(c["choices"], list) or len(c["choices"]) < 2 or len(set(c["choices"])) != len(c["choices"]):
            raise ValueError("Cases require distinct choices.")
        if any(not isinstance(x, str) or not x.strip() for x in c["choices"]):
            raise ValueError("Choice labels must be nonempty strings.")
        if c["ground_truth"] not in c["choices"] or c["ai_answer"] not in c["choices"]:
            raise ValueError("Ground truth and AI answer must be among the choices.")
        for key in ("case_id", "dataset_id", "question", "rationale", "ground_truth_policy"):
            if not isinstance(c[key], str) or not c[key].strip():
                raise ValueError(key + " must be nonempty.")
    return bank


def pilot_assignment(bank, participant_index):
    """Latin rotation across successive participant indices, one case per participant.

    Counterbalances presentation, not advice correctness. Researchers must balance
    correctness and difficulty in the bank; this draft is never Phase-16 final data.
    """
    validate_pilot_bank(bank)
    if isinstance(participant_index, bool) or int(participant_index) != participant_index or participant_index < 0:
        raise ValueError("participant_index must be a nonnegative integer.")
    cases = bank["cases"]
    order = list(range(len(cases)))
    np.random.default_rng(42 + participant_index).shuffle(order)
    return [{"case": cases[i], "condition": CONDITIONS[(i + participant_index) % len(CONDITIONS)]} for i in order]
