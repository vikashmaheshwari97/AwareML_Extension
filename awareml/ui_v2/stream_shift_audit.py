from __future__ import annotations

"""Descriptive dataset-composition shift audit for Streaming Observatory.

Jensen-Shannon divergence (JSD) is used here to compare empirical feature/target
distributions in each stream window with the first processed window. It is
intentionally separate from prediction-error drift detection.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _js_from_probabilities(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask]))) if mask.any() else 0.0

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _series_js(reference: pd.Series, current: pd.Series) -> float:
    a = reference.dropna()
    b = current.dropna()
    if a.empty or b.empty:
        return np.nan
    numeric = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
    combined_unique = pd.concat([a, b], ignore_index=True).nunique(dropna=True)
    if numeric and combined_unique > 30:
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


def window_feature_distribution_shift(
    df: pd.DataFrame,
    target: Optional[str],
    window_size: int,
    processed_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Return one row per feature/window JSD against the first window."""
    if df is None or df.empty:
        return pd.DataFrame()
    window_size = max(50, int(window_size or 500))
    n = min(len(df), int(processed_samples)) if processed_samples else len(df)
    work = df.iloc[:n].copy()
    if len(work) < window_size:
        return pd.DataFrame()
    reference = work.iloc[:window_size]
    columns = list(work.columns)
    rows = []
    for start in range(0, len(work), window_size):
        current = work.iloc[start:start + window_size]
        if len(current) < max(50, window_size // 4):
            continue
        window_id = len({r["Window"] for r in rows}) + 1 if rows else 1
        sample = int(start + len(current))
        for col in columns:
            try:
                value = _series_js(reference[col], current[col])
            except Exception:
                value = np.nan
            rows.append({
                "Window": int(start // window_size) + 1,
                "Sample": sample,
                "Feature": str(col),
                "JSD": value,
                "Kind": "Target" if target and col == target else "Feature",
            })
    return pd.DataFrame(rows)


def window_distribution_shift(
    df: pd.DataFrame,
    target: Optional[str],
    window_size: int,
    processed_samples: Optional[int] = None,
) -> pd.DataFrame:
    long = window_feature_distribution_shift(df, target, window_size, processed_samples)
    if long.empty:
        return pd.DataFrame()
    rows = []
    for (window, sample), group in long.groupby(["Window", "Sample"], sort=True):
        features = group[group["Kind"].eq("Feature")].copy()
        vals = pd.to_numeric(features["JSD"], errors="coerce").dropna()
        target_rows = group[group["Kind"].eq("Target")]
        target_jsd = pd.to_numeric(target_rows["JSD"], errors="coerce").dropna()
        most = "N/A"
        if not features.empty:
            tmp = features.assign(_v=pd.to_numeric(features["JSD"], errors="coerce")).dropna(subset=["_v"])
            if not tmp.empty:
                most = str(tmp.sort_values("_v", ascending=False).iloc[0]["Feature"])
        rows.append({
            "Window": int(window),
            "Sample": int(sample),
            "Mean feature JSD": float(vals.mean()) if not vals.empty else np.nan,
            "Median feature JSD": float(vals.median()) if not vals.empty else np.nan,
            "P95 feature JSD": float(vals.quantile(0.95)) if not vals.empty else np.nan,
            "Max feature JSD": float(vals.max()) if not vals.empty else np.nan,
            "Most shifted feature": most,
            "Target JSD": float(target_jsd.iloc[0]) if not target_jsd.empty else np.nan,
        })
    return pd.DataFrame(rows)


def distribution_shift_figure(audit: pd.DataFrame) -> go.Figure:
    """Restore the original Observatory JSD trend with clearer research styling."""
    fig = go.Figure()
    if audit is None or audit.empty:
        return fig

    fig.add_trace(go.Scatter(
        x=audit["Sample"], y=audit["Mean feature JSD"],
        mode="lines+markers", name="Mean feature JSD",
        line=dict(width=3.2, color="#0f6fd6"),
        marker=dict(size=7, color="#0f6fd6", line=dict(width=1, color="white")),
        customdata=np.c_[pd.to_numeric(audit.get("P95 feature JSD"), errors="coerce")],
        hovertemplate="Sample %{x:,}<br>Mean feature JSD %{y:.4f}<br>P95 feature JSD %{customdata[0]:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=audit["Sample"], y=audit["Max feature JSD"],
        mode="lines+markers", name="Max feature JSD",
        line=dict(width=2.8, dash="dash", color="#6db7f4"),
        marker=dict(size=7, color="#6db7f4"),
        customdata=audit[["Most shifted feature"]].to_numpy(),
        hovertemplate="Sample %{x:,}<br>Max feature JSD %{y:.4f}<br>Most shifted feature %{customdata[0]}<extra></extra>",
    ))
    if "Target JSD" in audit and pd.to_numeric(audit["Target JSD"], errors="coerce").notna().any():
        fig.add_trace(go.Scatter(
            x=audit["Sample"], y=audit["Target JSD"],
            mode="lines+markers", name="Target JSD",
            line=dict(width=2.6, dash="dot", color="#ef4444"),
            marker=dict(size=6, color="#ef4444"),
            hovertemplate="Sample %{x:,}<br>Target JSD %{y:.4f}<extra></extra>",
        ))
    fig.add_hline(y=0, line_width=1, line_color="rgba(100,116,139,.35)")
    fig.update_layout(
        title="Dataset-composition shift audit · Jensen–Shannon divergence from first window",
        xaxis_title="Processed stream sample",
        yaxis_title="JSD (0 = same empirical distribution)",
        height=430,
        margin=dict(l=72, r=30, t=62, b=92),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig

def _kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros_like(grid)
    spread = float(np.std(values))
    bw = max(0.0025, min(0.05, 1.06 * (spread if spread > 1e-12 else 0.01) * values.size ** (-0.2)))
    z = (grid[:, None] - values[None, :]) / bw
    density = np.exp(-0.5 * z * z).mean(axis=1) / (bw * np.sqrt(2.0 * np.pi))
    return density


def jsd_density_figure(feature_audit: pd.DataFrame, max_windows: int = 6) -> go.Figure:
    """Density-style view similar to drift literature, but using observed feature JSDs."""
    fig = go.Figure()
    if feature_audit is None or feature_audit.empty:
        return fig
    work = feature_audit[feature_audit["Kind"].eq("Feature")].copy()
    work["JSD"] = pd.to_numeric(work["JSD"], errors="coerce")
    work = work.dropna(subset=["JSD"])
    if work.empty:
        return fig
    windows = sorted(work["Window"].unique().tolist())
    nonref = [w for w in windows if int(w) != 1] or windows
    if len(nonref) > max_windows:
        idx = np.linspace(0, len(nonref) - 1, max_windows).round().astype(int)
        selected = [nonref[i] for i in sorted(set(idx.tolist()))]
    else:
        selected = nonref
    upper = max(0.05, float(work["JSD"].quantile(0.995)) * 1.25, float(work["JSD"].max()) * 1.08)
    upper = min(1.0, upper)
    grid = np.linspace(0.0, upper, 220)
    for i, window in enumerate(selected):
        vals = work.loc[work["Window"].eq(window), "JSD"].to_numpy(dtype=float)
        density = _kde(vals, grid)
        sample = work.loc[work["Window"].eq(window), "Sample"].iloc[0]
        latest = window == selected[-1]
        fig.add_trace(go.Scatter(
            x=grid,
            y=density,
            mode="lines",
            name="W{} · n={}".format(int(window), len(vals)),
            line=dict(width=3.0 if latest else 1.8),
            fill="tozeroy" if latest else None,
            opacity=0.95 if latest else 0.68,
            hovertemplate="Window {} · sample {}<br>JSD %{{x:.4f}}<br>Density %{{y:.2f}}<extra></extra>".format(int(window), int(sample)),
        ))
    observed = work.loc[work["Window"].ne(1), "JSD"]
    if observed.notna().any():
        p95 = float(observed.quantile(0.95))
        fig.add_vline(x=p95, line_dash="dot", line_width=2, line_color="#111827")
        fig.add_annotation(x=p95, y=1.0, yref="paper", text="Observed P95", showarrow=False, xanchor="left", yanchor="bottom")
    fig.update_layout(
        title="Feature-shift distribution · JSD density across stream windows",
        xaxis_title="Jensen–Shannon divergence from reference window",
        yaxis_title="Density across features",
        height=430,
        margin=dict(l=65, r=30, t=60, b=86),
        legend=dict(orientation="h", y=-0.24, x=0.5, xanchor="center"),
    )
    return fig


def jsd_feature_ranking_figure(feature_audit: pd.DataFrame, top_n: int = 12) -> go.Figure:
    fig = go.Figure()
    if feature_audit is None or feature_audit.empty:
        return fig
    work = feature_audit[feature_audit["Kind"].eq("Feature")].copy()
    work["JSD"] = pd.to_numeric(work["JSD"], errors="coerce")
    work = work.dropna(subset=["JSD"])
    work = work[work["Window"].ne(1)]
    if work.empty:
        return fig
    summary = work.groupby("Feature")["JSD"].agg(["median", "max", "mean"]).reset_index()
    summary = summary.sort_values(["max", "mean"], ascending=False).head(top_n).sort_values("max")
    for _, row in summary.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["median"], row["max"]], y=[row["Feature"], row["Feature"]],
            mode="lines", line=dict(width=4, color="#cbd5e1"), showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=summary["median"], y=summary["Feature"], mode="markers", name="Median window JSD",
        marker=dict(size=10, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=summary["max"], y=summary["Feature"], mode="markers+text", name="Maximum window JSD",
        marker=dict(size=12, symbol="diamond"), text=["{:.3f}".format(v) for v in summary["max"]], textposition="middle right",
    ))
    fig.update_layout(
        title="Most shifted features · median → maximum observed JSD",
        xaxis_title="Jensen–Shannon divergence",
        yaxis_title=None,
        height=max(380, 30 * len(summary) + 120),
        margin=dict(l=145, r=70, t=58, b=72),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    )
    return fig
