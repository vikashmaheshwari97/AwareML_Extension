from __future__ import annotations

"""Descriptive dataset-composition shift audit for Streaming Observatory.

This is deliberately separate from predictive-error drift.  It helps explain
cases where ADWIN/Page-Hinkley report no performance drift even though users
expect a non-stationary dataset.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _js_from_probabilities(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask]))) if mask.any() else 0.0
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _series_js(reference: pd.Series, current: pd.Series) -> float:
    a = reference.dropna(); b = current.dropna()
    if a.empty or b.empty:
        return np.nan
    numeric = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
    combined_unique = pd.concat([a, b], ignore_index=True).nunique(dropna=True)
    if numeric and combined_unique > 30:
        # Reference-anchored quantile bins are stable across windows and avoid
        # treating each unique continuous value as a category.
        quantiles = np.unique(a.quantile(np.linspace(0, 1, 11)).to_numpy(dtype=float))
        if len(quantiles) >= 3:
            bins = np.r_[-np.inf, quantiles[1:-1], np.inf]
            pa, _ = np.histogram(a.to_numpy(dtype=float), bins=bins)
            pb, _ = np.histogram(b.to_numpy(dtype=float), bins=bins)
            return _js_from_probabilities(pa.astype(float), pb.astype(float))
    cats = sorted(set(a.astype(str).tolist()) | set(b.astype(str).tolist()))
    pa = a.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    pb = b.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    return _js_from_probabilities(pa, pb)


def window_distribution_shift(
    df: pd.DataFrame,
    target: Optional[str],
    window_size: int,
    processed_samples: Optional[int] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    window_size = max(50, int(window_size or 500))
    n = min(len(df), int(processed_samples)) if processed_samples else len(df)
    work = df.iloc[:n].copy()
    if len(work) < window_size:
        return pd.DataFrame()
    reference = work.iloc[:window_size]
    feature_cols = [c for c in work.columns if c != target]
    rows = []
    for start in range(0, len(work), window_size):
        current = work.iloc[start:start + window_size]
        if len(current) < max(50, window_size // 4):
            continue
        js = {}
        for col in feature_cols:
            try:
                js[col] = _series_js(reference[col], current[col])
            except Exception:
                js[col] = np.nan
        finite = {k: v for k, v in js.items() if v is not None and np.isfinite(v)}
        target_js = None
        if target and target in work.columns:
            try:
                target_js = _series_js(reference[target], current[target])
            except Exception:
                target_js = np.nan
        rows.append({
            "Window": len(rows) + 1,
            "Sample": int(start + len(current)),
            "Mean feature JSD": float(np.mean(list(finite.values()))) if finite else np.nan,
            "Max feature JSD": float(max(finite.values())) if finite else np.nan,
            "Most shifted feature": max(finite, key=finite.get) if finite else "N/A",
            "Target JSD": target_js,
        })
    return pd.DataFrame(rows)


def distribution_shift_figure(audit: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if audit is None or audit.empty:
        return fig
    fig.add_trace(go.Scatter(
        x=audit["Sample"], y=audit["Mean feature JSD"], mode="lines+markers",
        name="Mean feature JSD", line=dict(width=3),
    ))
    fig.add_trace(go.Scatter(
        x=audit["Sample"], y=audit["Max feature JSD"], mode="lines+markers",
        name="Max feature JSD", line=dict(width=2.5, dash="dash"),
        customdata=audit[["Most shifted feature"]].to_numpy(),
        hovertemplate="Sample %{x}<br>Max JSD %{y:.4f}<br>Feature %{customdata[0]}<extra></extra>",
    ))
    if "Target JSD" in audit and pd.to_numeric(audit["Target JSD"], errors="coerce").notna().any():
        fig.add_trace(go.Scatter(
            x=audit["Sample"], y=audit["Target JSD"], mode="lines+markers",
            name="Target JSD", line=dict(width=2.5, dash="dot"),
        ))
    fig.update_layout(
        title="Dataset-composition shift audit · Jensen–Shannon divergence from first window",
        xaxis_title="Processed stream sample",
        yaxis_title="JSD (0 = same empirical distribution)",
        height=390,
        margin=dict(l=65, r=28, t=58, b=62),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig
