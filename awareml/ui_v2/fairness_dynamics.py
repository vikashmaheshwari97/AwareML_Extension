from __future__ import annotations

"""FairStream-inspired *audit* dynamics for AwareML fairness evidence.

This module does not turn AwareML into FairStream and does not change model
selection.  It replays already-observed window-level fairness gaps through a
transparent diagnostic envelope/debt calculation so persistent violations can
be inspected after a run.
"""

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots import FRAMEWORK_COLORS


def fairness_series_frame(results: Sequence[Mapping[str, Any]], metric_key: str) -> pd.DataFrame:
    rows = []
    for result in results or []:
        fw = str(result.get("framework"))
        for point in result.get("points") or []:
            value = point.get(metric_key)
            sample = point.get("sample")
            if value is None or sample is None:
                continue
            try:
                value = float(value); sample = int(sample)
            except Exception:
                continue
            if np.isfinite(value):
                rows.append({"Framework": fw, "Sample": sample, "Gap": max(0.0, value)})
    return pd.DataFrame(rows).sort_values(["Framework", "Sample"]).reset_index(drop=True) if rows else pd.DataFrame()


def replay_fairness_audit(
    series: pd.DataFrame,
    initial_upper: float = 0.10,
    debt_decay: float = 0.90,
    warning_threshold: float = 0.02,
    drift_threshold: float = 0.04,
    min_upper: float = 0.02,
    max_upper: float = 0.25,
    alpha_drift: float = 0.20,
    alpha_violation: float = 0.25,
    beta_relax: float = 0.005,
) -> pd.DataFrame:
    """Replay a lightweight envelope + fairness-debt audit.

    The update mirrors the FairStream paper's concepts (time-varying upper
    feasibility reference and D_t=lambda*D_{t-1}+max(0,c_t-u_t)) while staying
    honest about missing FairStream-only uncertainty/support signals.
    """
    if series is None or series.empty:
        return pd.DataFrame()
    initial_upper = float(np.clip(initial_upper, min_upper, max_upper))
    debt_decay = float(np.clip(debt_decay, 0.0, 1.0))
    rows = []
    for fw, group in series.groupby("Framework", sort=False):
        upper = initial_upper
        debt = 0.0
        prev_gap = None
        for _, row in group.sort_values("Sample").iterrows():
            gap = float(row["Gap"])
            delta = 0.0 if prev_gap is None else gap - prev_gap
            abs_delta = abs(delta)
            prior_upper = upper
            if prev_gap is not None and delta > 0:
                upper -= alpha_drift * abs(delta)
            elif prev_gap is not None:
                upper += beta_relax
            if gap > prior_upper:
                upper -= alpha_violation * (gap - prior_upper)
            upper = float(np.clip(upper, min_upper, max_upper))
            violation = max(0.0, gap - upper)
            debt = debt_decay * debt + violation
            if prev_gap is None:
                event = "baseline"
            elif abs_delta >= drift_threshold:
                event = "fairness_change"
            elif abs_delta >= warning_threshold:
                event = "warning"
            else:
                event = "stable"
            rows.append({
                "Framework": fw,
                "Sample": int(row["Sample"]),
                "Gap": gap,
                "Envelope": upper,
                "Violation": violation,
                "Debt": debt,
                "Delta gap": delta,
                "Event": event,
            })
            prev_gap = gap
    return pd.DataFrame(rows)


def fairness_debt_summary(audit: pd.DataFrame) -> pd.DataFrame:
    if audit is None or audit.empty:
        return pd.DataFrame()
    rows = []
    for fw, group in audit.groupby("Framework", sort=False):
        group = group.sort_values("Sample")
        rows.append({
            "Framework": fw,
            "Mean gap": float(group["Gap"].mean()),
            "Worst gap": float(group["Gap"].max()),
            "Violation windows": int((group["Violation"] > 0).sum()),
            "Fairness-change flags": int((group["Event"] == "fairness_change").sum()),
            "Peak debt": float(group["Debt"].max()),
            "Final debt": float(group["Debt"].iloc[-1]),
        })
    return pd.DataFrame(rows).sort_values(["Final debt", "Worst gap"]).reset_index(drop=True)


def fairness_dynamics_figure(audit: pd.DataFrame, framework: str) -> go.Figure:
    data = audit[audit["Framework"].astype(str).eq(str(framework))].sort_values("Sample").copy()
    if data.empty:
        return go.Figure()
    color = FRAMEWORK_COLORS.get(str(framework), "#2563eb")
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            "(a) Fairness gap and adaptive audit envelope",
            "(b) Accumulated fairness debt",
            "(c) Scale-normalized temporal profile",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
    )
    # Feasible envelope area.
    fig.add_trace(go.Scatter(
        x=data["Sample"], y=data["Envelope"], mode="lines", name="Audit envelope uₜ",
        line=dict(color="#0f766e", width=2.5, dash="dash"),
        fill="tozeroy", fillcolor="rgba(20,184,166,.10)",
        hovertemplate="Sample %{x}<br>Envelope %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data["Sample"], y=data["Gap"], mode="lines+markers", name="Fairness gap cₜ",
        line=dict(color=color, width=3), marker=dict(size=7),
        hovertemplate="Sample %{x}<br>Gap %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    viol = data[data["Violation"] > 0]
    if not viol.empty:
        fig.add_trace(go.Scatter(
            x=viol["Sample"], y=viol["Gap"], mode="markers", name="Envelope violation",
            marker=dict(size=12, color="#dc2626", symbol="diamond", line=dict(color="white", width=1.5)),
            hovertemplate="Violation at %{x}<br>Gap %{y:.4f}<extra></extra>",
        ), row=1, col=1)
    shifts = data[data["Event"].eq("fairness_change")]
    if not shifts.empty:
        fig.add_trace(go.Scatter(
            x=shifts["Sample"], y=shifts["Gap"], mode="markers", name="Fairness-change flag",
            marker=dict(size=13, color="#f97316", symbol="x", line=dict(width=2)),
            hovertemplate="Fairness-change flag at %{x}<br>Gap %{y:.4f}<extra></extra>",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data["Sample"], y=data["Debt"], mode="lines+markers", name="Debt Dₜ",
        line=dict(color="#6d28d9", width=3), fill="tozeroy", fillcolor="rgba(109,40,217,.12)",
        hovertemplate="Sample %{x}<br>Debt %{y:.4f}<extra></extra>",
    ), row=2, col=1)

    gap_max = max(float(data["Gap"].max()), 1e-12)
    debt_max = max(float(data["Debt"].max()), 1e-12)
    fig.add_trace(go.Scatter(
        x=data["Sample"], y=data["Gap"] / gap_max, mode="lines+markers", name="Normalized gap",
        line=dict(color=color, width=2.5),
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=data["Sample"], y=data["Debt"] / debt_max if debt_max > 1e-11 else np.zeros(len(data)),
        mode="lines+markers", name="Normalized debt", line=dict(color="#6d28d9", width=2.5, dash="dash"),
    ), row=2, col=2)

    fig.update_yaxes(title_text="Gap ↓", row=1, col=1)
    fig.update_yaxes(title_text="Debt", row=2, col=1)
    fig.update_yaxes(title_text="Relative level (0–1)", range=[0, 1.05], row=2, col=2)
    fig.update_xaxes(title_text="Stream sample", row=1, col=1)
    fig.update_xaxes(title_text="Stream sample", row=2, col=1)
    fig.update_xaxes(title_text="Stream sample", row=2, col=2)
    fig.update_layout(
        title="FairStream-inspired fairness dynamics audit · {}".format(framework),
        height=720,
        margin=dict(l=60, r=28, t=80, b=70),
        legend=dict(orientation="h", y=-0.13, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig
