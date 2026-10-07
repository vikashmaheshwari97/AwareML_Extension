"""Measured, model-level diagnostics. No LLM-faithfulness claims."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


METRIC_GUIDE = {
    "stability": "Top-k overlap between resampled explanation vectors; may be trivially 1 when k equals feature count.",
    "consistency": "Cosine agreement between explanation vectors; not decision/explanation alignment.",
    "sensitivity": "Normalized L1 difference across resampled explanations; lower means less variation.",
    "sparsity": "Hoyer sparsity of mean absolute importance; concentration, not explanation correctness.",
    "fidelity": "Legacy clipped accuracy drop after replacing up to five top features; not surrogate fidelity.",
    "deletion_drop_signed": "Replay accuracy minus accuracy after replacing top-k features. Negative values are retained.",
    "sufficiency_gap_signed": "Replay accuracy minus accuracy with only top-k features retained; closer to zero means similar accuracy.",
    "top_k_mass": "Share of recorded absolute attribution mass in top-k; not causal completeness.",
    "features_for_90pct_mass": "Fewest features covering at least 90% of recorded attribution mass.",
    "top_k_fraction": "k divided by feature count. A value of 1 makes top-k comparisons uninformative.",
}


def intervention_diagnostics(model, X, y, importances, categorical_features=None):
    # Imported at call time to avoid a circular import.
    from .explainability import _safe_predict, _safe_accuracy
    values = np.asarray(importances, dtype=float)
    if values.ndim != 1 or len(values) != X.shape[1] or not np.isfinite(values).all():
        raise ValueError("A finite importance for every feature is required.")
    mass = np.abs(values)
    if mass.sum() <= 1e-12:
        raise ValueError("Zero attribution mass cannot support coverage metrics.")
    order = np.argsort(-mass, kind="stable")
    k = min(5, max(1, int(math.ceil(len(values) / 2))))
    selected = set(order[:k])
    categorical = set(categorical_features or [])
    deleted, retained = X.copy(), X.copy()
    for i, col in enumerate(X.columns):
        if col in categorical:
            modes = X[col].mode(dropna=True)
            baseline = modes.iloc[0] if len(modes) else 0.0
        else:
            baseline = pd.to_numeric(X[col], errors="coerce").median()
            baseline = 0.0 if pd.isna(baseline) else float(baseline)
        if i in selected:
            deleted[col] = baseline
        else:
            retained[col] = baseline
    base = _safe_accuracy(y, _safe_predict(model, X))
    deletion = _safe_accuracy(y, _safe_predict(model, deleted))
    sufficiency = _safe_accuracy(y, _safe_predict(model, retained))
    normalized = mass / mass.sum()
    return {
        "status": "ok", "schema_version": "multilevel_xai_v1",
        "scope": "current_model_replay", "window_n": len(X),
        "feature_count": len(values), "top_k": k,
        "top_k_features": [str(X.columns[i]) for i in order[:k]],
        "top_k_fraction": k / len(values),
        "top_k_mass": float(normalized[order[:k]].sum()),
        "features_for_90pct_mass": int(np.searchsorted(np.cumsum(normalized[order]), .9) + 1),
        "deletion_drop_signed": float(base - deletion),
        "sufficiency_gap_signed": float(base - sufficiency),
        "replay_accuracy": base, "deleted_accuracy": deletion,
        "retained_accuracy": sufficiency,
        "baseline": "categorical mode / numeric median on explanation window",
        "limitations": "Replacement can be out of distribution; correlated features can mask effects. Final-model replay is not prequential performance.",
    }
