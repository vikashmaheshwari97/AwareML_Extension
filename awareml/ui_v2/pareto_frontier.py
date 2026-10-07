from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .plots import FRAMEWORK_COLORS, apply_research_layout
from .page_utils import plot


def _unit_lower_better(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.dropna()
    if valid.empty:
        return out
    lo, hi = float(valid.min()), float(valid.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return out
    if hi <= lo:
        out.loc[valid.index] = 0.0
    else:
        out.loc[valid.index] = (valid - lo) / (hi - lo)
    return out


def _two_dimensional_pareto(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact nondominance: lower x and higher y are preferred."""
    n = len(x)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            keep[i] = False
            continue
        for j in range(n):
            if i == j or not np.isfinite(x[j]) or not np.isfinite(y[j]):
                continue
            no_worse = x[j] <= x[i] and y[j] >= y[i]
            strictly = x[j] < x[i] or y[j] > y[i]
            if no_worse and strictly:
                keep[i] = False
                break
    return keep


def build_sustainability_pareto_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "framework" not in df.columns:
        raise ValueError("Decision Lab frame must contain 'framework'.")
    df["energy_norm"] = _unit_lower_better(df.get("energy_kwh", pd.Series(index=df.index, dtype=float)))
    df["co2_norm"] = _unit_lower_better(df.get("co2_kg", pd.Series(index=df.index, dtype=float)))
    # Both measured resources are required; missing one does not silently collapse
    # the comparison to the other.
    df["sustainability_burden"] = df[["energy_norm", "co2_norm"]].mean(axis=1, skipna=False)
    df["accuracy_numeric"] = pd.to_numeric(df.get("accuracy"), errors="coerce")
    df["pareto_2d"] = _two_dimensional_pareto(
        df["sustainability_burden"].to_numpy(dtype=float),
        df["accuracy_numeric"].to_numpy(dtype=float),
    )

    # Explicit, reproducible visual compromise point: among the exact 2D Pareto
    # points, choose the point closest to the run-relative ideal corner
    # (minimum burden, maximum accuracy). This is a visual 2D compromise, not the
    # canonical multi-objective AwareML recommendation.
    valid = df.dropna(subset=["sustainability_burden", "accuracy_numeric"]).copy()
    df["optimal_2d"] = False
    if not valid.empty:
        x = valid["sustainability_burden"]
        y = valid["accuracy_numeric"]
        x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else pd.Series(0.0, index=valid.index)
        y_loss = (y.max() - y) / (y.max() - y.min()) if y.max() > y.min() else pd.Series(0.0, index=valid.index)
        distance = np.sqrt(x_norm ** 2 + y_loss ** 2)
        candidates = valid.index[valid["pareto_2d"]]
        chosen = distance.loc[candidates].idxmin() if len(candidates) else distance.idxmin()
        df.loc[chosen, "optimal_2d"] = True
    return df


def render_sustainability_accuracy_pareto(frame: pd.DataFrame, key: str = "decision_pareto_frontier") -> None:
    df = build_sustainability_pareto_frame(frame)
    valid = df.dropna(subset=["sustainability_burden", "accuracy_numeric"]).copy()
    if valid.empty:
        st.info("Pareto frontier requires measured Accuracy, Energy and CO₂ for the current run.")
        return

    fig = go.Figure()
    frontier = valid[valid["pareto_2d"]].sort_values(["sustainability_burden", "accuracy_numeric"])
    if len(frontier) >= 2:
        fig.add_trace(go.Scatter(
            x=frontier["sustainability_burden"],
            y=frontier["accuracy_numeric"],
            mode="lines",
            name="Exact 2D Pareto frontier",
            line=dict(color="#0f766e", width=3.5, dash="dash"),
            hoverinfo="skip",
        ))

    for _, row in valid.iterrows():
        fw = str(row["framework"])
        optimal = bool(row.get("optimal_2d"))
        pareto = bool(row.get("pareto_2d"))
        symbol = "star" if optimal else ("diamond" if pareto else "circle")
        size = 24 if optimal else (17 if pareto else 13)
        color = "#f59e0b" if optimal else FRAMEWORK_COLORS.get(fw, "#64748b")
        fig.add_trace(go.Scatter(
            x=[float(row["sustainability_burden"])],
            y=[float(row["accuracy_numeric"])],
            mode="markers+text",
            name=fw,
            text=[fw],
            textposition="top center",
            marker=dict(
                size=size,
                symbol=symbol,
                color=color,
                line=dict(width=2.5 if pareto or optimal else 1.0, color="#0f172a" if pareto or optimal else "#ffffff"),
                opacity=0.95,
            ),
            customdata=[[row.get("energy_kwh"), row.get("co2_kg"), row.get("runtime_sec"), row.get("near_pareto"), optimal]],
            hovertemplate=(
                "<b>%{text}</b><br>Accuracy: %{y:.4f}<br>Normalized Energy+CO₂ burden: %{x:.3f}"
                "<br>Energy: %{customdata[0]} kWh<br>CO₂: %{customdata[1]} kg"
                "<br>Runtime: %{customdata[2]} s<br>Canonical near-Pareto: %{customdata[3]}"
                "<br>2D optimal compromise: %{customdata[4]}<extra></extra>"
            ),
        ))

    x_med = float(valid["sustainability_burden"].median())
    y_med = float(valid["accuracy_numeric"].median())
    fig.add_vline(x=x_med, line_dash="dot", line_color="rgba(100,116,139,.45)")
    fig.add_hline(y=y_med, line_dash="dot", line_color="rgba(100,116,139,.45)")

    optimal_row = valid[valid["optimal_2d"]].iloc[0] if valid["optimal_2d"].any() else None
    if optimal_row is not None:
        fig.add_annotation(
            x=float(optimal_row["sustainability_burden"]), y=float(optimal_row["accuracy_numeric"]),
            text="2D optimal compromise", showarrow=True, arrowhead=2, ax=66, ay=-48,
            bgcolor="rgba(255,255,255,.92)", bordercolor="#f59e0b", borderwidth=1,
            font=dict(size=11, color="#92400e"),
        )

    fig.add_annotation(
        x=0.01, y=float(valid["accuracy_numeric"].max()), xref="x", yref="y",
        text="ideal direction ↖", showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=11, color="#0f766e"),
    )

    apply_research_layout(
        fig,
        height=520,
        legend="right",
        title="Interactive Pareto Frontier · Sustainability vs Accuracy",
        bottom_margin=72,
    )
    fig.update_layout(
        margin=dict(l=70, r=210, t=62, b=78),
        xaxis_title="Normalized CO₂ + Energy burden  ← lower is greener",
        yaxis_title="Observed prequential accuracy  ↑ higher is better",
    )
    plot(fig, key)

    pareto_names = valid.loc[valid["pareto_2d"], "framework"].astype(str).tolist()
    best_acc = str(valid.sort_values("accuracy_numeric", ascending=False).iloc[0]["framework"])
    greenest = str(valid.sort_values("sustainability_burden").iloc[0]["framework"])
    optimal_name = str(optimal_row["framework"]) if optimal_row is not None else "N/A"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("2D optimal compromise", optimal_name)
    c2.metric("Accuracy leader", best_acc)
    c3.metric("Lowest normalized burden", greenest)
    c4.metric("2D Pareto set size", len(pareto_names))
    st.caption(
        "The star marks the **2D optimal compromise**: the exact Pareto point closest to the run-relative ideal corner (minimum normalized Energy+CO₂ burden, maximum accuracy). "
        "It is a transparent visual compromise for this two-axis plot—not a replacement for the canonical multi-objective ε-Pareto result or the current weighted Decision Lab recommendation. "
        "Point size is fixed; runtime is available on hover rather than encoded as bubble area."
    )
