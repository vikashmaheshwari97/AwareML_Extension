from __future__ import annotations

"""Reviewer-facing Responsible-AI visual builders (v6.4).

The design intentionally avoids block/heatmap encodings for heterogeneous
Responsible-AI evidence. Exact raw values remain available and no aggregate
Responsible-AI or fairness score is manufactured.
"""

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots import FRAMEWORK_COLORS


FAIRNESS_GROUP_METRICS = {
    "Accuracy ↑": "accuracy",
    "TPR ↑": "true_positive_rate",
    "FPR ↓": "false_positive_rate",
    "Selection rate ↔": "predicted_positive_rate",
    "PPV ↑": "positive_predictive_value",
    "Error rate ↓": "error_rate",
}

FAIRNESS_TEMPORAL_METRICS = {
    "Demographic parity gap": "dp_diff",
    "Equal opportunity gap": "eo_diff",
    "Equalized odds gap": "equalized_odds_gap",
    "Error-rate gap": "error_rate_gap",
}

_GROUP_COLORS = [
    "#2563eb", "#16a34a", "#f97316", "#7c3aed", "#0f766e",
    "#dc2626", "#ca8a04", "#0891b2", "#be123c", "#4f46e5",
]


def _finite(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _normalise(values: pd.Series, higher_is_better: bool) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    finite = vals.dropna()
    if finite.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    lo, hi = float(finite.min()), float(finite.max())
    if abs(hi - lo) <= 1e-12:
        return pd.Series(0.5, index=values.index, dtype=float).where(vals.notna())
    score = (vals - lo) / (hi - lo)
    return score if higher_is_better else 1.0 - score


def _pareto_high_y_low_x(frame: pd.DataFrame, x: str, y: str) -> pd.Series:
    work = frame[[x, y]].apply(pd.to_numeric, errors="coerce")
    flags = []
    for i, row in work.iterrows():
        if row.isna().any():
            flags.append(False)
            continue
        dominated = False
        for j, other in work.iterrows():
            if i == j or other.isna().any():
                continue
            if (
                other[x] <= row[x]
                and other[y] >= row[y]
                and (other[x] < row[x] or other[y] > row[y])
            ):
                dominated = True
                break
        flags.append(not dominated)
    return pd.Series(flags, index=frame.index, dtype=bool)


def fairness_group_frame(fair: Mapping[str, Any]) -> pd.DataFrame:
    perf = fair.get("group_performance") or {}
    support_map = fair.get("groups") or {}
    rows = []
    for group, values in perf.items():
        row = {"Group": str(group)}
        for label, key in FAIRNESS_GROUP_METRICS.items():
            row[label] = _finite((values or {}).get(key))
        support = support_map.get(str(group), support_map.get(group))
        row["N"] = support
        rows.append(row)
    return pd.DataFrame(rows)


def _group_palette(groups: Sequence[str]) -> dict[str, str]:
    return {str(g): _GROUP_COLORS[i % len(_GROUP_COLORS)] for i, g in enumerate(groups)}


def group_metric_profile_figure(fair: Mapping[str, Any]) -> go.Figure:
    """Six small multiples: exact group values on common 0–1 axes.

    Unlike the previous square-marker matrix, position carries the value and the
    shaded vertical band shows only the observed between-group range.
    """
    frame = fairness_group_frame(fair)
    if frame.empty:
        return go.Figure()

    metrics = list(FAIRNESS_GROUP_METRICS)
    groups = frame["Group"].astype(str).tolist()
    palette = _group_palette(groups)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=metrics,
        horizontal_spacing=0.065,
        vertical_spacing=0.23,
    )

    for idx, metric in enumerate(metrics):
        row = idx // 3 + 1
        col = idx % 3 + 1
        values = pd.to_numeric(frame[metric], errors="coerce")
        finite_vals = values.dropna()
        if not finite_vals.empty:
            lo, hi = float(finite_vals.min()), float(finite_vals.max())
            if hi > lo + 1e-12:
                fig.add_vrect(
                    x0=lo, x1=hi,
                    fillcolor="rgba(148,163,184,.13)",
                    line_width=0,
                    row=row, col=col,
                    layer="below",
                )

        for gi, group in enumerate(groups):
            value = _finite(frame.loc[frame["Group"].astype(str).eq(group), metric].iloc[0])
            if value is None:
                continue
            support = frame.loc[frame["Group"].astype(str).eq(group), "N"].iloc[0]
            support_text = "N/A" if pd.isna(support) else f"{int(support):,}"
            # A short stem makes the exact horizontal position easy to read
            # without using marker area or colored boxes as a quantitative cue.
            fig.add_trace(go.Scatter(
                x=[0.0, value], y=[group, group], mode="lines",
                line=dict(color="rgba(148,163,184,.34)", width=2),
                hoverinfo="skip", showlegend=False,
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=[value], y=[group], mode="markers+text",
                text=[f"{value:.3f}"], textposition="middle right",
                marker=dict(
                    size=14, color=palette[group],
                    line=dict(color="white", width=2),
                ),
                name=f"Group {group}",
                legendgroup=group,
                showlegend=(idx == 0),
                customdata=[[support_text]],
                hovertemplate=(
                    f"<b>Group {group}</b><br>{metric}: %{{x:.4f}}"
                    "<br>Support: %{customdata[0]}<extra></extra>"
                ),
            ), row=row, col=col)

        fig.update_xaxes(
            range=[0, 1.04], tickvals=[0, .25, .5, .75, 1.0],
            tickformat=".2f", gridcolor="rgba(148,163,184,.18)",
            zeroline=False, row=row, col=col,
        )
        fig.update_yaxes(
            categoryorder="array", categoryarray=groups[::-1],
            title=None, row=row, col=col,
        )

    fig.update_layout(
        margin=dict(l=72, r=28, t=58, b=88),
        legend=dict(
            orientation="h", y=-0.12, x=0.5, xanchor="center",
            title=None,
        ),
        hovermode="closest",
    )
    return fig


def fairness_disparity_structure_figure(fair: Mapping[str, Any]) -> go.Figure:
    """Observed between-group range for every available fairness diagnostic."""
    frame = fairness_group_frame(fair)
    rows = []
    for metric in FAIRNESS_GROUP_METRICS:
        vals = pd.to_numeric(frame.get(metric), errors="coerce") if not frame.empty else pd.Series(dtype=float)
        valid = frame.loc[vals.dropna().index].copy() if not vals.empty else pd.DataFrame()
        if valid.empty:
            continue
        valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
        low = valid.loc[valid[metric].idxmin()]
        high = valid.loc[valid[metric].idxmax()]
        rows.append({
            "Metric": metric,
            "Gap": float(high[metric] - low[metric]),
            "Low group": str(low["Group"]),
            "Low": float(low[metric]),
            "High group": str(high["Group"]),
            "High": float(high[metric]),
        })
    data = pd.DataFrame(rows)
    fig = go.Figure()
    if data.empty:
        return fig
    data = data.sort_values("Gap", ascending=True)
    max_gap = max(float(data["Gap"].max()), 0.02)
    for _, r in data.iterrows():
        fig.add_trace(go.Scatter(
            x=[0, r["Gap"]], y=[r["Metric"], r["Metric"]], mode="lines",
            line=dict(color="rgba(100,116,139,.38)", width=4),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=data["Gap"], y=data["Metric"], mode="markers+text",
        text=[f"{v:.3f}" for v in data["Gap"]], textposition="middle right",
        marker=dict(size=13, color="#2563eb", line=dict(color="white", width=2)),
        customdata=data[["Low group", "Low", "High group", "High"]].to_numpy(),
        hovertemplate=(
            "<b>%{y}</b><br>Observed group range: %{x:.4f}"
            "<br>Low: Group %{customdata[0]} · %{customdata[1]:.4f}"
            "<br>High: Group %{customdata[2]} · %{customdata[3]:.4f}<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.update_layout(
        xaxis=dict(
            title="Observed between-group range · smaller means closer parity",
            range=[0, max_gap * 1.22], gridcolor="rgba(148,163,184,.18)",
        ),
        yaxis=dict(title=None),
        margin=dict(l=138, r=52, t=18, b=60),
        showlegend=False,
    )
    return fig


def fairness_tpr_fpr_plane(fair: Mapping[str, Any]) -> go.Figure:
    """One operating point per group: false-positive vs true-positive rate."""
    frame = fairness_group_frame(fair)
    fig = go.Figure()
    if frame.empty:
        return fig
    groups = frame["Group"].astype(str).tolist()
    palette = _group_palette(groups)
    valid_xy = []
    for _, r in frame.iterrows():
        tpr = _finite(r.get("TPR ↑"))
        fpr = _finite(r.get("FPR ↓"))
        if tpr is None or fpr is None:
            continue
        group = str(r["Group"])
        support = r.get("N")
        valid_xy.append((fpr, tpr))
        fig.add_trace(go.Scatter(
            x=[fpr], y=[tpr], mode="markers+text",
            text=[f"Group {group}"], textposition="top center",
            marker=dict(size=17, color=palette[group], line=dict(color="white", width=2)),
            customdata=[["N/A" if pd.isna(support) else f"{int(support):,}"]],
            hovertemplate=(
                f"<b>Group {group}</b><br>TPR: %{{y:.4f}}<br>FPR: %{{x:.4f}}"
                "<br>Support: %{customdata[0]}<extra></extra>"
            ),
            showlegend=False,
        ))
    if valid_xy:
        fig.add_annotation(
            x=0.01, y=0.99, xref="paper", yref="paper",
            text="preferred direction ↖ higher TPR · lower FPR",
            showarrow=False, xanchor="left",
            font=dict(size=11, color="#0f766e"),
            bgcolor="rgba(255,255,255,.85)",
        )
    fig.update_layout(
        xaxis=dict(title="False-positive rate ↓", range=[0, 1], gridcolor="rgba(148,163,184,.18)"),
        yaxis=dict(title="True-positive rate ↑", range=[0, 1], gridcolor="rgba(148,163,184,.18)"),
        margin=dict(l=68, r=30, t=30, b=60),
        showlegend=False,
    )
    return fig


def fairness_over_time_figure(result: Mapping[str, Any]) -> go.Figure:
    rows = []
    for p in result.get("points") or []:
        if hasattr(p, "__dict__"):
            p = vars(p)
        if not isinstance(p, Mapping):
            continue
        sample = _finite(p.get("sample"))
        if sample is None:
            continue
        for label, key in FAIRNESS_TEMPORAL_METRICS.items():
            value = _finite(p.get(key))
            if value is not None:
                rows.append({"Sample": sample, "Metric": label, "Gap": value})
    fig = go.Figure()
    if not rows:
        return fig
    data = pd.DataFrame(rows)
    palette = ["#2563eb", "#16a34a", "#f97316", "#7c3aed"]
    for color, (metric, group) in zip(palette, data.groupby("Metric", sort=False)):
        fig.add_trace(go.Scatter(
            x=group["Sample"], y=group["Gap"], mode="lines+markers", name=metric,
            line=dict(width=2.5, color=color), marker=dict(size=7),
            hovertemplate="Sample %{x:,.0f}<br>Gap %{y:.4f}<extra>" + metric + "</extra>",
        ))
    for i, sample in enumerate(result.get("drift_events") or []):
        fig.add_vline(x=float(sample), line=dict(color="#dc2626", width=1.5, dash="dot"), opacity=0.45)
        if i == 0:
            fig.add_annotation(
                x=float(sample), y=1.02, yref="paper", text="predictive-change marker",
                showarrow=False, font=dict(size=10, color="#b91c1c"),
            )
    fig.update_layout(
        xaxis=dict(title="Stream sample", gridcolor="rgba(148,163,184,.20)"),
        yaxis=dict(title="Observed fairness gap ↓", rangemode="tozero", gridcolor="rgba(148,163,184,.20)"),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(l=64, r=28, t=32, b=78),
    )
    return fig


def fairness_performance_frontier(results: Sequence[Mapping[str, Any]], gap_key: str = "dp_diff") -> go.Figure:
    rows = []
    for r in results or []:
        fair = r.get("fairness") or {}
        gap = _finite(fair.get(gap_key))
        acc = _finite(r.get("accuracy"))
        if gap is not None and acc is not None:
            rows.append({"Framework": str(r.get("framework")), "Gap": gap, "Accuracy": acc})
    frame = pd.DataFrame(rows)
    fig = go.Figure()
    if frame.empty:
        return fig
    frame["Pareto"] = _pareto_high_y_low_x(frame, "Gap", "Accuracy")
    frontier = frame[frame["Pareto"]].sort_values("Gap")
    if len(frontier) >= 2:
        fig.add_trace(go.Scatter(
            x=frontier["Gap"], y=frontier["Accuracy"], mode="lines",
            line=dict(color="#0f766e", width=3, dash="dash"),
            name="Nondominated frontier", hoverinfo="skip",
        ))
    for _, row in frame.iterrows():
        fw = row["Framework"]
        fig.add_trace(go.Scatter(
            x=[row["Gap"]], y=[row["Accuracy"]], mode="markers+text", text=[fw], textposition="top center",
            marker=dict(
                size=14, symbol="diamond" if row["Pareto"] else "circle",
                color=FRAMEWORK_COLORS.get(fw, "#64748b"),
                line=dict(color="#0f172a" if row["Pareto"] else "white", width=2),
            ),
            showlegend=False,
            hovertemplate=f"<b>{fw}</b><br>DP gap: %{{x:.4f}}<br>Accuracy: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        xaxis=dict(title="Demographic-parity gap ↓", rangemode="tozero", gridcolor="rgba(148,163,184,.20)"),
        yaxis=dict(title="Prequential accuracy ↑", gridcolor="rgba(148,163,184,.20)"),
        margin=dict(l=70, r=32, t=18, b=58), showlegend=False,
    )
    return fig


_RAI_SPECS = [
    ("Accuracy", "Accuracy ↑", True),
    ("DP gap", "Fairness · DP gap ↓", False),
    ("XAI fidelity", "XAI fidelity ↑", True),
    ("Energy (kWh)", "Energy efficiency ↑", False),
    ("CO2 (kg)", "CO₂ efficiency ↑", False),
]


def rai_directional_profile_figure(frame: pd.DataFrame) -> go.Figure:
    """Direction-aligned run-relative profiles with exact raw values in hover."""
    work = frame.copy()
    fig = go.Figure()
    if work.empty:
        return fig

    dimensions = [label for _, label, _ in _RAI_SPECS]
    score_by_col = {}
    raw_by_col = {}
    for col, label, higher in _RAI_SPECS:
        raw = pd.to_numeric(work.get(col), errors="coerce")
        raw_by_col[label] = raw
        score_by_col[label] = _normalise(raw, higher)

    for i, r in work.iterrows():
        fw = str(r.get("Framework"))
        ys = [score_by_col[d].loc[i] for d in dimensions]
        raw = [raw_by_col[d].loc[i] for d in dimensions]
        custom = []
        for d, val in zip(dimensions, raw):
            custom.append([d, "N/A" if pd.isna(val) else f"{float(val):.6g}"])
        fig.add_trace(go.Scatter(
            x=dimensions, y=ys, mode="lines+markers",
            name=fw,
            line=dict(width=2.7, color=FRAMEWORK_COLORS.get(fw, "#64748b")),
            marker=dict(size=9, color=FRAMEWORK_COLORS.get(fw, "#64748b"), line=dict(color="white", width=1.5)),
            customdata=np.asarray(custom, dtype=object),
            hovertemplate=(
                "<b>" + fw + "</b><br>%{customdata[0]}"
                "<br>Exact observed value: %{customdata[1]}"
                "<br>Run-relative desirability: %{y:.3f}<extra></extra>"
            ),
        ))
    fig.add_annotation(
        x=0.0, y=1.075, xref="paper", yref="paper",
        text="All axes direction-aligned for visualization only · higher = more desirable",
        showarrow=False, xanchor="left", font=dict(size=11, color="#475569"),
    )
    fig.update_layout(
        xaxis=dict(title=None),
        yaxis=dict(title="Run-relative desirability ↑", range=[-0.03, 1.06], tickvals=[0, .25, .5, .75, 1], gridcolor="rgba(148,163,184,.18)"),
        legend=dict(orientation="h", y=-0.17, x=0.5, xanchor="center"),
        margin=dict(l=70, r=28, t=46, b=92),
        hovermode="x",
    )
    return fig


def rai_rank_flow_figure(frame: pd.DataFrame) -> go.Figure:
    """Framework rank trajectory across heterogeneous RAI dimensions."""
    work = frame.copy()
    fig = go.Figure()
    if work.empty:
        return fig
    labels = []
    ranks = {}
    for col, label, higher in _RAI_SPECS:
        vals = pd.to_numeric(work.get(col), errors="coerce")
        rank = vals.rank(method="min", ascending=not higher, na_option="bottom")
        labels.append(label)
        ranks[label] = rank
    for i, r in work.iterrows():
        fw = str(r.get("Framework"))
        ys = [float(ranks[label].loc[i]) for label in labels]
        fig.add_trace(go.Scatter(
            x=labels, y=ys, mode="lines+markers+text",
            text=[f"#{int(v)}" for v in ys], textposition="top center",
            name=fw,
            line=dict(width=2.5, color=FRAMEWORK_COLORS.get(fw, "#64748b")),
            marker=dict(size=8, color=FRAMEWORK_COLORS.get(fw, "#64748b")),
            hovertemplate="<b>" + fw + "</b><br>%{x}<br>Rank %{y:.0f}<extra></extra>",
        ))
    n = max(1, len(work))
    fig.update_layout(
        xaxis=dict(title=None),
        yaxis=dict(title="Within-dimension rank ↓", range=[n + .45, .55], dtick=1, gridcolor="rgba(148,163,184,.18)"),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(l=76, r=30, t=24, b=96),
        hovermode="x unified",
    )
    return fig


def rai_dimension_leaders(frame: pd.DataFrame) -> list[dict[str, Any]]:
    leaders = []
    if frame is None or frame.empty:
        return leaders
    for col, label, higher in _RAI_SPECS:
        vals = pd.to_numeric(frame.get(col), errors="coerce")
        valid = vals.dropna()
        if valid.empty:
            leaders.append({"Dimension": label, "Framework": "N/A", "Value": None})
            continue
        idx = valid.idxmax() if higher else valid.idxmin()
        leaders.append({
            "Dimension": label,
            "Framework": str(frame.loc[idx, "Framework"]),
            "Value": float(vals.loc[idx]),
        })
    return leaders


_FAITH_COMPONENTS = [
    ("Grounding", "mean_grounding_validity"),
    ("Decision alignment", "mean_decision_alignment"),
    ("Attribution alignment", "mean_attribution_alignment"),
    ("Counterfactual sensitivity", "mean_counterfactual_sensitivity"),
    ("Irrelevant invariance", "mean_irrelevant_invariance"),
]


def faithfulness_evidence_profile_figure(summary: Mapping[str, Any]) -> go.Figure:
    """Component lollipops with deterministic AEF as a reference line."""
    rows = []
    for label, key in _FAITH_COMPONENTS:
        value = _finite(summary.get(key))
        if value is not None:
            rows.append({"Metric": label, "Score": value})
    data = pd.DataFrame(rows)
    fig = go.Figure()
    if data.empty:
        return fig
    aef = _finite(summary.get("mean_evidence_fidelity_score"))
    colors = ["#38bdf8", "#2563eb", "#6366f1", "#f59e0b", "#10b981"]
    cmap = {label: colors[i] for i, (label, _) in enumerate(_FAITH_COMPONENTS)}
    data = data.iloc[::-1].copy()
    for _, r in data.iterrows():
        fig.add_trace(go.Scatter(
            x=[0, r["Score"]], y=[r["Metric"], r["Metric"]], mode="lines",
            line=dict(color="rgba(100,116,139,.28)", width=4),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=data["Score"], y=data["Metric"], mode="markers+text",
        text=[f"{v:.3f}" for v in data["Score"]], textposition="middle right",
        marker=dict(
            size=15,
            color=[cmap[m] for m in data["Metric"]],
            line=dict(color="white", width=2),
        ),
        hovertemplate="<b>%{y}</b><br>Observed score: %{x:.4f}<extra></extra>",
        showlegend=False,
    ))
    if aef is not None:
        fig.add_vline(x=aef, line=dict(color="#8b5cf6", width=2.2, dash="dash"))
        fig.add_annotation(
            x=aef, y=1.04, yref="paper",
            text=f"Deterministic AEF {aef:.3f}", showarrow=False,
            font=dict(size=11, color="#7c3aed"),
            bgcolor="rgba(255,255,255,.88)",
        )
    fig.update_layout(
        xaxis=dict(title="Observed Phase-8 score ↑", range=[0, 1.04], gridcolor="rgba(148,163,184,.18)"),
        yaxis=dict(title=None),
        margin=dict(l=170, r=44, t=46, b=58),
        showlegend=False,
    )
    return fig


def faithfulness_aef_comparison_figure(deterministic: Mapping[str, Any], ollama: Mapping[str, Any]) -> go.Figure:
    det = _finite(deterministic.get("mean_evidence_fidelity_score"))
    live = _finite(ollama.get("mean_evidence_fidelity_score"))
    fig = go.Figure()
    values = [v for v in [det, live] if v is not None]
    if not values:
        return fig
    if det is not None and live is not None:
        fig.add_trace(go.Scatter(
            x=[det, live], y=[0, 0], mode="lines",
            line=dict(color="#94a3b8", width=7), hoverinfo="skip", showlegend=False,
        ))
    if det is not None:
        fig.add_trace(go.Scatter(
            x=[det], y=[0], mode="markers+text", name="Deterministic",
            text=[f"Deterministic {det:.3f}"], textposition="bottom center",
            marker=dict(size=18, color="#2563eb", line=dict(color="white", width=2)),
            hovertemplate="Deterministic AEF %{x:.4f}<extra></extra>",
        ))
    if live is not None:
        fig.add_trace(go.Scatter(
            x=[live], y=[0], mode="markers+text", name="Live Ollama",
            text=[f"Live Ollama {live:.3f}"], textposition="top center",
            marker=dict(size=18, color="#8b5cf6", symbol="diamond", line=dict(color="white", width=2)),
            hovertemplate="Live Ollama AEF %{x:.4f}<extra></extra>",
        ))
    if det is not None and live is not None:
        delta = live - det
        fig.add_annotation(
            x=(det + live) / 2, y=0.19,
            text=f"Δ {delta:+.3f}", showarrow=False,
            font=dict(size=12, color="#0f172a"), bgcolor="rgba(255,255,255,.9)",
        )
    lo = max(0.0, min(values) - 0.08)
    hi = min(1.0, max(values) + 0.08)
    if hi - lo < 0.12:
        mid = (hi + lo) / 2
        lo, hi = max(0, mid - 0.08), min(1, mid + 0.08)
    fig.update_layout(
        xaxis=dict(title="External evidence-faithfulness score ↑", range=[lo, hi], gridcolor="rgba(148,163,184,.18)"),
        yaxis=dict(visible=False, range=[-0.35, 0.35]),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        margin=dict(l=34, r=34, t=26, b=78),
    )
    return fig


def faithfulness_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    vals = []
    for label, key in _FAITH_COMPONENTS:
        value = _finite(summary.get(key))
        if value is not None:
            vals.append((label, value))
    if not vals:
        return {"lowest_label": "N/A", "lowest_value": None, "spread": None}
    low = min(vals, key=lambda kv: kv[1])
    scores = [v for _, v in vals]
    return {
        "lowest_label": low[0],
        "lowest_value": low[1],
        "spread": float(max(scores) - min(scores)),
    }
