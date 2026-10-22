from __future__ import annotations

"""Research-grade window-level temporal views for the Streaming Observatory.

The v6.2 design deliberately avoids heatmaps. Window-level evidence is encoded by
quantitative position: within-framework deviation, distribution intervals and
cross-framework rank flow.
"""

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .plots import FRAMEWORK_COLORS, apply_research_layout
from .page_utils import plot


def _window_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results or []:
        fw = str(result.get("framework") or "Unknown")
        for point in result.get("points") or []:
            if hasattr(point, "__dict__"):
                point = dict(point.__dict__)
            if not isinstance(point, dict):
                continue
            value = point.get("rolling_accuracy")
            if value is None:
                value = point.get("accuracy")
            try:
                sample = int(point.get("sample"))
                value = float(value)
            except Exception:
                continue
            if np.isfinite(value):
                rows.append({"Framework": fw, "Sample": sample, "Accuracy": value})
    return pd.DataFrame(rows)


def temporal_comparison_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Align window accuracy and derive within-run deviation and cross-framework rank."""
    frame = _window_frame(results)
    if frame.empty:
        return frame
    frame = frame.sort_values(["Sample", "Framework"]).reset_index(drop=True)
    medians = frame.groupby("Framework")["Accuracy"].transform("median")
    frame["Framework median"] = medians
    frame["Deviation pp"] = 100.0 * (frame["Accuracy"] - medians)
    frame["Window rank"] = frame.groupby("Sample")["Accuracy"].rank(
        method="min", ascending=False
    )
    return frame


def temporal_deviation_figure(results: Sequence[Mapping[str, Any]]) -> go.Figure:
    frame = temporal_comparison_frame(results)
    fig = go.Figure()
    if frame.empty:
        return fig

    for fw, group in frame.groupby("Framework", sort=False):
        group = group.sort_values("Sample")
        color = FRAMEWORK_COLORS.get(str(fw), "#64748b")
        fig.add_trace(go.Scatter(
            x=group["Sample"],
            y=group["Deviation pp"],
            mode="lines+markers",
            name=str(fw),
            line=dict(width=2.7, color=color),
            marker=dict(size=7, color=color, line=dict(width=1, color="white")),
            customdata=np.c_[group["Accuracy"], group["Framework median"], group["Window rank"]],
            hovertemplate=(
                "%{x:,} samples<br>Deviation %{y:+.2f} pp"
                "<br>Rolling accuracy %{customdata[0]:.4f}"
                "<br>Run median %{customdata[1]:.4f}"
                "<br>Window rank %{customdata[2]:.0f}<extra>" + str(fw) + "</extra>"
            ),
        ))

    fig.add_hline(y=0.0, line_width=1.4, line_dash="dot", line_color="#64748b")
    fig.update_layout(
        title="Window-level performance deviation · each framework centred on its own run median",
        xaxis_title="Processed stream sample",
        yaxis_title="Rolling-accuracy deviation (percentage points)",
        height=430,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        margin=dict(l=76, r=28, t=64, b=92),
    )
    return fig


def temporal_stability_summary(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = temporal_comparison_frame(results)
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for fw, group in frame.groupby("Framework"):
        acc = pd.to_numeric(group["Accuracy"], errors="coerce").dropna()
        rank = pd.to_numeric(group["Window rank"], errors="coerce").dropna()
        if acc.empty:
            continue
        q25, median, q75 = acc.quantile([0.25, 0.50, 0.75])
        rows.append({
            "Framework": fw,
            "Worst window": float(acc.min()),
            "Q1": float(q25),
            "Median rolling accuracy": float(median),
            "Q3": float(q75),
            "Best window": float(acc.max()),
            "IQR accuracy": float(q75 - q25),
            "Mean rank": float(rank.mean()) if not rank.empty else None,
            "Rank volatility": float(rank.std(ddof=0)) if len(rank) > 1 else 0.0,
            "Windows": int(len(acc)),
        })
    return pd.DataFrame(rows).sort_values(
        ["Median rolling accuracy", "Mean rank"], ascending=[False, True]
    )


def temporal_stability_interval_figure(results: Sequence[Mapping[str, Any]]) -> go.Figure:
    summary = temporal_stability_summary(results)
    fig = go.Figure()
    if summary.empty:
        return fig

    ordered = summary.sort_values("Median rolling accuracy", ascending=True)
    for _, row in ordered.iterrows():
        fw = str(row["Framework"])
        color = FRAMEWORK_COLORS.get(fw, "#64748b")
        # Full observed range.
        fig.add_trace(go.Scatter(
            x=[row["Worst window"], row["Best window"]],
            y=[fw, fw],
            mode="lines",
            line=dict(width=2.2, color="#cbd5e1"),
            showlegend=False,
            hoverinfo="skip",
        ))
        # IQR as the stronger inner interval.
        fig.add_trace(go.Scatter(
            x=[row["Q1"], row["Q3"]],
            y=[fw, fw],
            mode="lines",
            line=dict(width=8, color=color),
            opacity=0.45,
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[row["Median rolling accuracy"]],
            y=[fw],
            mode="markers+text",
            marker=dict(size=13, symbol="diamond", color=color, line=dict(width=1.2, color="white")),
            text=["{:.3f}".format(row["Median rolling accuracy"])],
            textposition="middle right",
            name=fw,
            showlegend=False,
            customdata=[[row["Worst window"], row["Q1"], row["Q3"], row["Best window"], row["IQR accuracy"]]],
            hovertemplate=(
                "Median %{x:.4f}<br>Worst %{customdata[0]:.4f}<br>Q1 %{customdata[1]:.4f}"
                "<br>Q3 %{customdata[2]:.4f}<br>Best %{customdata[3]:.4f}<br>IQR %{customdata[4]:.4f}<extra>" + fw + "</extra>"
            ),
        ))

    fig.update_layout(
        title="Window stability profile · observed range, IQR and median",
        xaxis_title="Rolling accuracy",
        yaxis_title=None,
        height=max(360, 64 * len(ordered) + 120),
        margin=dict(l=120, r=70, t=64, b=70),
    )
    return fig


def temporal_rank_flow_figure(results: Sequence[Mapping[str, Any]]) -> go.Figure:
    frame = temporal_comparison_frame(results)
    fig = go.Figure()
    if frame.empty:
        return fig

    for fw, group in frame.groupby("Framework", sort=False):
        group = group.sort_values("Sample")
        color = FRAMEWORK_COLORS.get(str(fw), "#64748b")
        fig.add_trace(go.Scatter(
            x=group["Sample"],
            y=group["Window rank"],
            mode="lines+markers",
            name=str(fw),
            line=dict(width=2.5, color=color),
            marker=dict(size=7, color=color, line=dict(width=1, color="white")),
            customdata=np.c_[group["Accuracy"], group["Deviation pp"]],
            hovertemplate=(
                "%{x:,} samples<br>Rank %{y:.0f}<br>Rolling accuracy %{customdata[0]:.4f}"
                "<br>Own-median deviation %{customdata[1]:+.2f} pp<extra>" + str(fw) + "</extra>"
            ),
        ))

    n_fw = max(1, int(frame["Framework"].nunique()))
    fig.update_yaxes(
        autorange="reversed", tickmode="linear", dtick=1,
        range=[n_fw + 0.35, 0.65],
        title="Rank (1 = highest rolling accuracy)",
    )
    fig.update_layout(
        title="Temporal rank flow · competitive ordering across windows",
        xaxis_title="Processed stream sample",
        height=430,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        margin=dict(l=76, r=28, t=64, b=92),
    )
    return fig


def render_window_research_comparison(results: Sequence[Mapping[str, Any]]) -> None:
    """Render quantitative window evidence without any heatmap encoding."""
    fig = temporal_deviation_figure(results)
    apply_research_layout(
        fig,
        height=430,
        legend="bottom",
        title="Window-level performance deviation · each framework centred on its own run median",
        bottom_margin=104,
    )
    plot(fig, "r122_window_deviation")
    st.caption(
        "Each framework is centred on its own median rolling accuracy, so sustained deterioration or improvement is visible in percentage points even when absolute accuracies are close. Zero means that framework's own typical run level."
    )

    left, right = st.columns([1.0, 1.12])
    with left:
        fig = temporal_stability_interval_figure(results)
        apply_research_layout(
            fig,
            height=430,
            legend="none",
            title="Window stability profile · observed range, IQR and median",
            bottom_margin=62,
        )
        plot(fig, "r122_window_stability_profile")
        st.caption(
            "Thin line = worst-to-best observed window; thick segment = interquartile range; diamond = median. Narrower intervals indicate more stable window-level performance."
        )
    with right:
        fig = temporal_rank_flow_figure(results)
        apply_research_layout(
            fig,
            height=430,
            legend="bottom",
            title="Temporal rank flow · competitive ordering across windows",
            bottom_margin=104,
        )
        plot(fig, "r122_window_rank_flow")
        st.caption(
            "Rank 1 is the highest rolling accuracy in that window. Crossings reveal changes in relative ordering; they are descriptive and do not by themselves imply statistical significance."
        )

    summary = temporal_stability_summary(results)
    if not summary.empty:
        with st.expander("Exact window-level stability summary", expanded=False):
            st.dataframe(summary, use_container_width=True, hide_index=True)
