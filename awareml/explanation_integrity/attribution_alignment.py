"""Independent attribution-vector analysis inspired by CC-SHAP and ConstLLM.

Consumes measured, aligned attributions; never infers them from words in a rationale.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _vector(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or len(arr) < 2 or not np.isfinite(arr).all():
        raise ValueError("Attributions must be finite one-dimensional vectors with at least two features.")
    return arr


def alignment_scores(decision, explanation, top_k=5):
    a, b = _vector(decision), _vector(explanation)
    if a.shape != b.shape:
        raise ValueError("Decision and explanation feature axes must match exactly.")
    if isinstance(top_k, bool) or int(top_k) != top_k or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return {"status": "unavailable", "reason": "zero attribution vector", "cosine_signed": None,
                "spearman_signed": None, "spearman_absolute": None, "top_k_jaccard": None, "absolute_mass_overlap": None}
    cosine = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    rho = None if np.ptp(a) == 0 or np.ptp(b) == 0 else float(spearmanr(a, b)[0])
    # ConstLLM's inspected implementation ranks absolute magnitudes; preserve
    # signed rank as a separate diagnostic instead of silently conflating them.
    abs_rho = None if np.ptp(np.abs(a)) == 0 or np.ptp(np.abs(b)) == 0 else float(spearmanr(np.abs(a), np.abs(b))[0])
    k = min(int(top_k), len(a))
    # Include every boundary tie, avoiding arbitrarily favourable top-k sets.
    def tops(v):
        mass = np.abs(v)
        threshold = np.sort(mass)[-k]
        return set(np.flatnonzero((mass >= threshold) & (mass > 0)).tolist())
    sa, sb = tops(a), tops(b)
    pa, pb = np.abs(a) / np.abs(a).sum(), np.abs(b) / np.abs(b).sum()
    return {"status": "ok", "cosine_signed": cosine, "spearman_signed": rho, "spearman_absolute": abs_rho,
            "spearman_note": "Undefined for a constant vector" if rho is None else None,
            "top_k_jaccard": len(sa & sb) / len(sa | sb),
            "effective_top_k_decision": len(sa), "effective_top_k_explanation": len(sb),
            "absolute_mass_overlap": float(np.minimum(pa, pb).sum()),
            "top_k": k, "n_features": len(a),
            "interpretation": "Attribution self-consistency; not task accuracy or proof of internal faithfulness."}


def validate_record(record):
    required = ("case_id", "model", "attribution_method", "feature_ids", "decision_attributions",
                "explanation_attributions", "decision", "explanation", "input_sha256", "target_definition")
    missing = [k for k in required if k not in record]
    if missing:
        raise ValueError("Missing attribution fields: " + ", ".join(missing))
    ids = record["feature_ids"]
    if not isinstance(ids, list) or not all(isinstance(x, str) and x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("feature_ids must be unique non-empty strings (use token positions, not token text).")
    a, b = _vector(record["decision_attributions"]), _vector(record["explanation_attributions"])
    if len(ids) != len(a) or a.shape != b.shape:
        raise ValueError("Feature IDs and both vectors must have identical lengths.")
    digest = str(record["input_sha256"])
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError("input_sha256 must identify the original input.")
    for field in ("case_id", "model", "attribution_method", "decision", "explanation", "target_definition"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(field + " must be a non-empty string.")
    return record


def analyze_records(records):
    output = []
    for record in records:
        validate_record(record)
        output.append(dict(record, alignment=alignment_scores(
            record["decision_attributions"], record["explanation_attributions"])))
    return output


def preference_pairs(records, minimum_gap=0.1):
    """Export measured train-only pairs; no training or correctness guarantee."""
    if not 0 < minimum_gap <= 2:
        raise ValueError("minimum_gap must be in (0, 2].")
    groups = {}
    evaluated = analyze_records(records)
    held_out_inputs = {r["input_sha256"] for r in evaluated if r.get("split") != "train"}
    for record in evaluated:
        if record.get("split") != "train" or record["input_sha256"] in held_out_inputs:
            continue
        key = (record["case_id"], record["input_sha256"], record["model"], record["decision"],
               record["attribution_method"], record["target_definition"], tuple(record["feature_ids"]),
               tuple(record["decision_attributions"]))
        score = record["alignment"].get("spearman_absolute")
        if score is not None:
            groups.setdefault(key, []).append(record)
    pairs = []
    for group in groups.values():
        ordered = sorted(group, key=lambda x: x["alignment"]["spearman_absolute"])
        low, high = ordered[0], ordered[-1]
        gap = high["alignment"]["spearman_absolute"] - low["alignment"]["spearman_absolute"]
        if gap >= minimum_gap and high["explanation"] != low["explanation"]:
            pairs.append({"case_id": high["case_id"], "input_sha256": high["input_sha256"],
                          "decision": high["decision"], "chosen": high["explanation"],
                          "rejected": low["explanation"], "spearman_gap": gap, "split": "train",
                          "criterion": "absolute-magnitude Spearman alignment, not factual correctness"})
    return pairs
