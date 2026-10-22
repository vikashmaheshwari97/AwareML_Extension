from __future__ import annotations

"""Observed hyperparameter-history analysis.

This module intentionally avoids calling observational association 'causal
impact'. It analyses parameters that were actually recorded during a run and
keeps next-run modification separate from already observed evidence.
"""

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .plots import FRAMEWORK_COLORS, apply_research_layout
from .page_utils import plot


def _flatten_tunable(params: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(params, Mapping):
        return {}
    tunable = params.get("tunable")
    source = tunable if isinstance(tunable, Mapping) else params
    out = {}
    for key, value in source.items():
        if isinstance(value, bool):
            out[str(key)] = value
        elif isinstance(value, (str, int, float)) or value is None:
            out[str(key)] = value
    return out


def parameter_history_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for record in result.get("parameter_history") or []:
        if not isinstance(record, Mapping):
            continue
        base = {
            "Sample": record.get("sample"),
            "Active algorithm": record.get("active_algorithm"),
            "Rolling accuracy": record.get("rolling_accuracy"),
            "Rolling Macro-F1": record.get("rolling_f1_macro"),
        }
        params = _flatten_tunable(record.get("parameters") or {})
        for name, value in params.items():
            rows.append({**base, "Parameter": str(name), "Value": value})
    return pd.DataFrame(rows)


def parameter_association_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    hist = parameter_history_frame(result)
    if hist.empty:
        return pd.DataFrame()
    rows = []
    for parameter, group in hist.groupby("Parameter", sort=True):
        x = pd.to_numeric(group["Value"], errors="coerce")
        y = pd.to_numeric(group["Rolling accuracy"], errors="coerce")
        valid = x.notna() & y.notna()
        x = x[valid]
        y = y[valid]
        unique = int(x.nunique()) if not x.empty else 0
        rho = None
        if len(x) >= 4 and unique >= 2 and y.nunique() >= 2:
            rho = float(x.rank(method="average").corr(y.rank(method="average")))
            if not np.isfinite(rho):
                rho = None
        rows.append({
            "Parameter": parameter,
            "Windows": int(len(x)),
            "Unique values": unique,
            "Spearman rho": rho,
            "Direction": (
                "higher associated with higher accuracy" if rho is not None and rho > 0.05 else
                "higher associated with lower accuracy" if rho is not None and rho < -0.05 else
                "weak / unavailable"
            ),
        })
    return pd.DataFrame(rows).sort_values(
        "Spearman rho", key=lambda s: pd.to_numeric(s, errors="coerce").abs(), ascending=False
    )


def parameter_change_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    hist = parameter_history_frame(result)
    if hist.empty:
        return pd.DataFrame()
    rows = []
    for parameter, group in hist.groupby("Parameter", sort=True):
        group = group.sort_values("Sample").reset_index(drop=True)
        for i in range(1, len(group)):
            before, after = group.iloc[i - 1], group.iloc[i]
            if str(before["Value"]) == str(after["Value"]):
                continue
            acc_before = pd.to_numeric(pd.Series([before["Rolling accuracy"]]), errors="coerce").iloc[0]
            acc_after = pd.to_numeric(pd.Series([after["Rolling accuracy"]]), errors="coerce").iloc[0]
            rows.append({
                "Parameter": parameter,
                "Sample": after["Sample"],
                "Before": before["Value"],
                "After": after["Value"],
                "Accuracy before": acc_before,
                "Accuracy after": acc_after,
                "Observed Δ accuracy (pp)": (
                    100.0 * (float(acc_after) - float(acc_before))
                    if pd.notna(acc_before) and pd.notna(acc_after) else None
                ),
                "Algorithm": after["Active algorithm"],
            })
    return pd.DataFrame(rows)


def parameter_association_figure(result: Mapping[str, Any], framework: str) -> go.Figure:
    assoc = parameter_association_frame(result)
    fig = go.Figure()
    if assoc.empty:
        return fig
    display = assoc.dropna(subset=["Spearman rho"]).head(14).sort_values("Spearman rho")
    if display.empty:
        return fig
    for _, row in display.iterrows():
        fig.add_trace(go.Scatter(
            x=[0.0, row["Spearman rho"]],
            y=[row["Parameter"], row["Parameter"]],
            mode="lines",
            line=dict(width=4, color="#cbd5e1"),
            hoverinfo="skip",
            showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=display["Spearman rho"],
        y=display["Parameter"],
        mode="markers+text",
        text=["{:+.2f}".format(v) for v in display["Spearman rho"]],
        textposition="middle right",
        marker=dict(size=13, color=FRAMEWORK_COLORS.get(framework, "#2563eb"), line=dict(width=1, color="white")),
        name="Observed association",
    ))
    fig.add_vline(x=0, line_width=1.2, line_color="#64748b")
    fig.update_xaxes(range=[-1.08, 1.08], title="Spearman association with same-window rolling accuracy")
    fig.update_yaxes(title=None)
    fig.update_layout(
        title="Observed hyperparameter association · descriptive, not causal",
        height=max(360, 30 * len(display) + 140),
        margin=dict(l=170, r=70, t=60, b=70),
        showlegend=False,
    )
    return fig


def parameter_change_figure(result: Mapping[str, Any], framework: str) -> go.Figure:
    changes = parameter_change_frame(result)
    fig = go.Figure()
    if changes.empty:
        return fig
    display = changes.dropna(subset=["Observed Δ accuracy (pp)"]).copy()
    if display.empty:
        return fig
    display["Label"] = display.apply(
        lambda r: "{} @ {:,}".format(r["Parameter"], int(r["Sample"])), axis=1
    )
    display = display.tail(18)
    fig.add_trace(go.Bar(
        x=display["Observed Δ accuracy (pp)"],
        y=display["Label"],
        orientation="h",
        customdata=display[["Before", "After", "Algorithm"]].astype(str).to_numpy(),
        hovertemplate=(
            "%{y}<br>Observed Δ accuracy %{x:+.2f} pp"
            "<br>%{customdata[0]} → %{customdata[1]}"
            "<br>Algorithm %{customdata[2]}<extra></extra>"
        ),
        marker=dict(color=FRAMEWORK_COLORS.get(framework, "#2563eb")),
        name="Observed change",
    ))
    fig.add_vline(x=0, line_width=1.2, line_color="#64748b")
    fig.update_layout(
        title="Parameter-change event audit · adjacent observed accuracy change",
        xaxis_title="Observed rolling-accuracy change (percentage points)",
        yaxis_title=None,
        height=max(380, 28 * len(display) + 130),
        margin=dict(l=180, r=30, t=60, b=70),
        showlegend=False,
    )
    return fig


def render_inline_hyperparameter_evidence(result: Mapping[str, Any], framework: str) -> None:
    hist = parameter_history_frame(result)
    st.markdown("### Hyperparameter evolution evidence")
    st.caption(
        "AwareML records the active algorithm and tunable parameter state at each evaluation window. Associations and before→after changes below are observational diagnostics only; they do not prove that a parameter caused the performance change."
    )
    if hist.empty:
        st.info(
            "No window-level parameter history exists for this run. Run a new benchmark after applying v6 to collect this evidence automatically."
        )
        return
    a1, a2, a3 = st.columns(3)
    a1.metric("Parameter-window records", str(len(hist)))
    a2.metric("Distinct recorded parameters", str(hist["Parameter"].nunique()))
    a3.metric("Observed parameter changes", str(len(parameter_change_frame(result))))

    assoc = parameter_association_frame(result)
    if not assoc.empty and assoc["Spearman rho"].notna().any():
        fig = parameter_association_figure(result, framework)
        apply_research_layout(fig, height=fig.layout.height or 420, legend="none", title="Observed hyperparameter association · descriptive, not causal", bottom_margin=70)
        plot(fig, "r120_hparam_assoc_{}".format(framework))
    else:
        st.info("The run did not contain enough variation in numeric parameters to estimate a window-level Spearman association.")

    changes = parameter_change_frame(result)
    if not changes.empty:
        fig = parameter_change_figure(result, framework)
        apply_research_layout(fig, height=fig.layout.height or 420, legend="none", title="Parameter-change event audit · adjacent observed accuracy change", bottom_margin=70)
        plot(fig, "r120_hparam_changes_{}".format(framework))
        with st.expander("Exact parameter-change records", expanded=False):
            st.dataframe(changes, use_container_width=True, hide_index=True)

    with st.expander("Exact parameter history", expanded=False):
        st.dataframe(hist, use_container_width=True, hide_index=True)
