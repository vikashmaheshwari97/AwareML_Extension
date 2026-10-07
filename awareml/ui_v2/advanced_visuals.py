from __future__ import annotations

"""Research-grade Plotly builders for the v4 Decision/Fairness/XAI views.

The functions in this module are intentionally presentation-only.  They never
change measured values, ranking logic, fairness values or XAI diagnostics.
"""

import math
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots import FRAMEWORK_COLORS


OBJECTIVE_LABELS = {
    "accuracy": "Accuracy",
    "runtime": "Runtime",
    "energy": "Energy",
    "co2": "CO₂",
    "fairness": "Fairness",
    "interpretability": "Interpretability",
}


def utility_contribution_map(contrib: pd.DataFrame, ranking: pd.DataFrame) -> go.Figure:
    """Bubble matrix of weighted objective contributions.

    Bubble area and label encode the exact weighted contribution.  This is a
    different visual grammar from the previous stacked bars while preserving
    the same inputs and total utility semantics.
    """
    objectives = [c for c in OBJECTIVE_LABELS if c in contrib.columns]
    if not objectives:
        return go.Figure()
    order = ranking["framework"].astype(str).tolist() if "framework" in ranking else contrib["framework"].astype(str).tolist()
    utility = {
        str(row.get("framework")): row.get("utility")
        for _, row in ranking.iterrows()
    }
    rows = []
    for _, row in contrib.iterrows():
        fw = str(row.get("framework"))
        for obj in objectives:
            value = pd.to_numeric(pd.Series([row.get(obj)]), errors="coerce").iloc[0]
            if pd.notna(value):
                rows.append((fw, obj, float(value)))
    if not rows:
        return go.Figure()
    vmax = max(v for _, _, v in rows) or 1.0
    xs, ys, sizes, texts, hovers, colors = [], [], [], [], [], []
    for fw, obj, value in rows:
        xs.append(OBJECTIVE_LABELS[obj])
        ys.append(fw)
        sizes.append(18.0 + 46.0 * math.sqrt(max(0.0, value) / vmax))
        texts.append("{:.3f}".format(value))
        hovers.append(
            "{}<br>{}: {:.4f}<br>Total utility: {}".format(
                fw,
                OBJECTIVE_LABELS[obj],
                value,
                "N/A" if utility.get(fw) is None else "{:.4f}".format(float(utility[fw])),
            )
        )
        colors.append(FRAMEWORK_COLORS.get(fw, "#64748b"))
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=texts,
            textposition="middle center",
            hovertext=hovers,
            hoverinfo="text",
            marker=dict(size=sizes, color=colors, opacity=0.88, line=dict(color="white", width=2)),
        )
    )
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(order)))
    fig.update_layout(
        title="Utility contribution by objective · contribution map",
        xaxis_title="Objective",
        yaxis_title="Framework",
        height=440,
        margin=dict(l=90, r=28, t=58, b=70),
        showlegend=False,
    )
    return fig


def desirability_radar(norm: pd.DataFrame) -> go.Figure:
    objectives = [c for c in OBJECTIVE_LABELS if c in norm.columns]
    if not objectives:
        return go.Figure()
    labels = [OBJECTIVE_LABELS[x] for x in objectives]
    fig = go.Figure()
    for _, row in norm.iterrows():
        fw = str(row.get("framework"))
        vals = [float(row.get(c)) if pd.notna(row.get(c)) else None for c in objectives]
        # Close the polygon.
        theta = labels + [labels[0]]
        radius = vals + [vals[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=radius,
                theta=theta,
                mode="lines+markers",
                name=fw,
                line=dict(color=FRAMEWORK_COLORS.get(fw, "#64748b"), width=2.5),
                marker=dict(size=6),
                opacity=0.88,
                connectgaps=False,
                hovertemplate=fw + "<br>%{theta}: %{r:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Observed objective desirability · higher is better · radar profiles",
        polar=dict(
            radialaxis=dict(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1.0], gridcolor="rgba(148,163,184,.35)"),
            angularaxis=dict(gridcolor="rgba(148,163,184,.25)"),
        ),
        height=440,
        margin=dict(l=45, r=45, t=60, b=45),
        legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center"),
    )
    return fig


def correlation_network(corr: pd.DataFrame) -> go.Figure:
    """Circular network for objective redundancy / opposition."""
    if corr is None or corr.empty:
        return go.Figure()
    labels = [str(c) for c in corr.columns]
    n = len(labels)
    if n == 0:
        return go.Figure()
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {label: (math.cos(a), math.sin(a)) for label, a in zip(labels, angles)}
    fig = go.Figure()
    # Draw strongest relationships first so weaker edges stay visible.
    edges = []
    for i, a in enumerate(labels):
        for j in range(i + 1, n):
            b = labels[j]
            try:
                rho = float(corr.loc[a, b])
            except Exception:
                continue
            if np.isfinite(rho):
                edges.append((abs(rho), rho, a, b))
    for _, rho, a, b in sorted(edges):
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        strength = abs(rho)
        color = "rgba(220,38,38,{:.3f})".format(0.16 + 0.74 * strength) if rho >= 0 else "rgba(37,99,235,{:.3f})".format(0.16 + 0.74 * strength)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(width=1.0 + 7.0 * strength, color=color),
                hoverinfo="text",
                text=["{} ↔ {}<br>Spearman ρ={:.2f}".format(a, b, rho)] * 2,
                showlegend=False,
            )
        )
        if strength >= 0.55:
            fig.add_annotation(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text="ρ={:.2f}".format(rho),
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,.78)",
                borderpad=2,
            )
    node_x, node_y, node_text = [], [], []
    for label in labels:
        x, y = pos[label]
        node_x.append(x); node_y.append(y)
        pretty = label.replace("runtime_sec", "runtime").replace("energy_kwh", "energy").replace("co2_kg", "CO₂").replace("fairness_score", "fairness").replace("interpretability_score", "interpretability")
        node_text.append(pretty)
    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=node_text,
            textposition="middle center", hoverinfo="text",
            marker=dict(size=72, color="white", line=dict(color="#0f172a", width=2.2)),
            showlegend=False,
        )
    )
    fig.update_xaxes(visible=False, range=[-1.35, 1.35])
    fig.update_yaxes(visible=False, range=[-1.35, 1.35], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title="Objective correlation network · red = positive, blue = inverse",
        height=500,
        margin=dict(l=25, r=25, t=58, b=25),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def fairness_bubble_matrix(fair: pd.DataFrame, metrics: Iterable[str]) -> go.Figure:
    metric_list = [m for m in metrics if m in fair.columns]
    rows = []
    for _, row in fair.iterrows():
        fw = str(row.get("Framework"))
        for metric in metric_list:
            value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            rows.append((fw, metric, None if pd.isna(value) else float(value)))
    finite = [v for _, _, v in rows if v is not None]
    vmax = max(finite) if finite else 1.0
    vmax = max(vmax, 0.05)
    fig = go.Figure()
    valid = [(fw, m, v) for fw, m, v in rows if v is not None]
    if valid:
        fig.add_trace(go.Scatter(
            x=[m for _, m, _ in valid], y=[fw for fw, _, _ in valid], mode="markers+text",
            text=["{:.3f}".format(v) for _, _, v in valid], textposition="middle center",
            marker=dict(
                size=[20 + 54 * math.sqrt(v / vmax) for _, _, v in valid],
                color=[v for _, _, v in valid], colorscale="YlOrRd", cmin=0, cmax=vmax,
                showscale=True, colorbar=dict(title="Gap", thickness=12, len=0.74),
                line=dict(color="white", width=2), opacity=0.92,
            ),
            hovertemplate="%{y}<br>%{x}: %{marker.color:.4f}<extra></extra>",
            showlegend=False,
        ))
    missing = [(fw, m) for fw, m, v in rows if v is None]
    if missing:
        fig.add_trace(go.Scatter(
            x=[m for _, m in missing], y=[fw for fw, _ in missing], mode="markers+text",
            text=["N/A"] * len(missing), textposition="middle center",
            marker=dict(size=36, symbol="x", color="#94a3b8"),
            hovertemplate="%{y}<br>%{x}: unavailable<extra></extra>", showlegend=False,
        ))
    fig.update_layout(
        title="Fairness disparity landscape · larger/darker bubbles = larger gap",
        height=420,
        margin=dict(l=105, r=55, t=58, b=80),
        xaxis_title="Fairness criterion",
        yaxis_title="Framework",
    )
    return fig


def fairness_risk_map(summary: pd.DataFrame) -> go.Figure:
    if summary is None or summary.empty:
        return go.Figure()
    fig = go.Figure()
    max_worst = pd.to_numeric(summary["Worst-window gap"], errors="coerce").max()
    max_worst = float(max_worst) if pd.notna(max_worst) and max_worst > 0 else 1.0
    for _, row in summary.iterrows():
        fw = str(row["Framework"])
        worst = float(row["Worst-window gap"])
        fig.add_trace(go.Scatter(
            x=[float(row["Mean gap"])], y=[float(row["Temporal volatility"])], mode="markers+text",
            text=[fw], textposition="top center", name=fw,
            marker=dict(
                size=24 + 34 * math.sqrt(max(0.0, worst) / max_worst),
                color=FRAMEWORK_COLORS.get(fw, "#64748b"),
                line=dict(color="white", width=2), opacity=0.9,
            ),
            hovertemplate=(
                fw + "<br>Mean gap %{x:.4f}<br>Volatility %{y:.4f}<br>Worst gap " + "{:.4f}".format(worst) + "<extra></extra>"
            ),
            showlegend=False,
        ))
    fig.update_layout(
        title="Temporal fairness risk map · lower-left is more stable and lower-disparity",
        xaxis_title="Mean observed gap ↓",
        yaxis_title="Temporal volatility ↓",
        height=390,
        margin=dict(l=65, r=25, t=58, b=62),
    )
    return fig


def system_xai_constellation(cdf: pd.DataFrame) -> go.Figure:
    cols = ["Fidelity", "Stability", "Consistency", "Sensitivity", "Sparsity"]
    labels = {
        "Fidelity": "Deletion fidelity ↑",
        "Stability": "Stability ↑",
        "Consistency": "Consistency ↑",
        "Sensitivity": "Sensitivity ↓",
        "Sparsity": "Sparsity (descriptive)",
    }
    rows = []
    for _, row in cdf.iterrows():
        fw = str(row.get("Framework"))
        for col in cols:
            value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if pd.notna(value):
                rows.append((fw, labels[col], float(value), col))
    if not rows:
        return go.Figure()
    fig = go.Figure()
    for fw in cdf["Framework"].astype(str).tolist():
        rr = [x for x in rows if x[0] == fw]
        if not rr:
            continue
        fig.add_trace(go.Scatter(
            x=[x[1] for x in rr], y=[fw] * len(rr), mode="markers+text",
            text=["{:.2f}".format(x[2]) for x in rr], textposition="middle center",
            marker=dict(
                size=[24 + 34 * math.sqrt(max(0.0, min(1.0, x[2]))) for x in rr],
                color=FRAMEWORK_COLORS.get(fw, "#64748b"),
                line=dict(color="white", width=2), opacity=0.88,
            ),
            name=fw,
            customdata=[[x[3], x[2]] for x in rr],
            hovertemplate=fw + "<br>%{x}<br>Raw value %{customdata[1]:.4f}<extra></extra>",
        ))
    fig.update_layout(
        title="System-level XAI constellation · raw diagnostics, no composite score",
        height=440,
        margin=dict(l=100, r=28, t=58, b=92),
        xaxis_title="Diagnostic (directions shown in label)",
        yaxis_title="Framework",
        legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center"),
    )
    return fig


def fairness_mean_worst_dumbbell(fair: pd.DataFrame) -> go.Figure:
    if fair is None or fair.empty:
        return go.Figure()
    work = fair[["Framework", "Comparable mean gap", "Worst available gap"]].copy()
    work["Comparable mean gap"] = pd.to_numeric(work["Comparable mean gap"], errors="coerce")
    work["Worst available gap"] = pd.to_numeric(work["Worst available gap"], errors="coerce")
    work = work.dropna(how="all", subset=["Comparable mean gap", "Worst available gap"]).sort_values("Worst available gap")
    fig = go.Figure()
    for _, row in work.iterrows():
        fw = str(row["Framework"])
        mean = row["Comparable mean gap"]
        worst = row["Worst available gap"]
        if pd.notna(mean) and pd.notna(worst):
            fig.add_trace(go.Scatter(
                x=[float(mean), float(worst)], y=[fw, fw], mode="lines",
                line=dict(color="rgba(148,163,184,.65)", width=5),
                hoverinfo="skip", showlegend=False,
            ))
    fig.add_trace(go.Scatter(
        x=work["Comparable mean gap"], y=work["Framework"], mode="markers", name="Comparable mean gap",
        marker=dict(size=14, color="#2563eb", line=dict(color="white", width=1.5)),
        hovertemplate="%{y}<br>Mean gap %{x:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=work["Worst available gap"], y=work["Framework"], mode="markers", name="Worst available gap",
        marker=dict(size=16, color="#ef4444", symbol="diamond", line=dict(color="white", width=1.5)),
        hovertemplate="%{y}<br>Worst gap %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Mean-to-worst disparity span · shorter and further left is better",
        xaxis_title="Observed fairness gap ↓", yaxis_title="Framework",
        height=420, margin=dict(l=100, r=25, t=58, b=75),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    )
    return fig


def system_xai_evidence_map(cdf: pd.DataFrame) -> go.Figure:
    """Joint view of three commonly interpretable XAI diagnostics.

    X=stability, Y=deletion fidelity, bubble area=consistency. Sensitivity and
    sparsity remain hover-only so no artificial composite score is created.
    """
    if cdf is None or cdf.empty:
        return go.Figure()
    work = cdf.copy()
    for col in ["Fidelity", "Stability", "Consistency", "Sensitivity", "Sparsity"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["Fidelity", "Stability", "Consistency"], how="any")
    fig = go.Figure()
    if work.empty:
        return fig
    for _, row in work.iterrows():
        fw = str(row["Framework"])
        consistency = max(0.0, min(1.0, float(row["Consistency"])))
        fig.add_trace(go.Scatter(
            x=[float(row["Stability"])], y=[float(row["Fidelity"])],
            mode="markers+text", text=[fw], textposition="top center", name=fw,
            marker=dict(
                size=26 + 42 * math.sqrt(consistency),
                color=FRAMEWORK_COLORS.get(fw, "#64748b"),
                line=dict(color="white", width=2), opacity=0.9,
            ),
            customdata=[[row.get("Method used"), row.get("Sensitivity"), row.get("Sparsity"), row.get("Consistency")]],
            hovertemplate=(
                fw + "<br>Stability %{x:.4f}<br>Deletion fidelity %{y:.4f}"
                + "<br>Consistency %{customdata[3]:.4f}<br>Sensitivity %{customdata[1]}"
                + "<br>Sparsity %{customdata[2]}<br>Method %{customdata[0]}<extra></extra>"
            ),
            showlegend=False,
        ))
    fig.update_layout(
        title="System XAI evidence map · bubble area = consistency",
        xaxis_title="Stability ↑",
        yaxis_title="Deletion fidelity ↑ (replay accuracy drop)",
        height=430,
        margin=dict(l=70, r=25, t=58, b=65),
    )
    return fig


# ---------------------------------------------------------------------------
# Fairness Lab v4.1 research-grade visualizations
# ---------------------------------------------------------------------------

def _pretty_fairness_metric(metric: str) -> str:
    return str(metric).replace("_", " ").replace("diff", "difference").strip().title()


def fairness_ranked_small_multiples(fair: pd.DataFrame, metrics: Iterable[str]) -> go.Figure:
    """Criterion-wise ranked lollipop small multiples.

    This intentionally replaces the old bubble matrix.  Position on a common
    quantitative axis carries the value; marker area does not encode the gap.
    Each panel ranks frameworks for one fairness definition and prints the
    exact observed value, which makes pairwise comparison auditable.
    """
    metric_list = [m for m in metrics if m in fair.columns]
    if not metric_list:
        return go.Figure()

    cols = 2
    rows = int(math.ceil(len(metric_list) / cols))
    subplot_titles = [_pretty_fairness_metric(m) + " ↓" for m in metric_list]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.16,
        vertical_spacing=0.18,
    )

    # Keep a common x range across panels because all displayed quantities are
    # fairness gaps on the same lower-is-better scale.
    all_values = []
    for metric in metric_list:
        vals = pd.to_numeric(fair[metric], errors="coerce")
        all_values.extend(vals.dropna().astype(float).tolist())
    xmax = max(all_values) if all_values else 1.0
    xmax = max(0.05, xmax * 1.24)

    for i, metric in enumerate(metric_list):
        rr, cc = divmod(i, cols)
        r, c = rr + 1, cc + 1
        work = fair[["Framework", metric]].copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna(subset=[metric]).sort_values(metric, ascending=True).reset_index(drop=True)
        if work.empty:
            fig.add_annotation(
                text="No comparable evidence", x=0.5, y=0.5,
                xref=f"x{'' if i == 0 else i+1} domain",
                yref=f"y{'' if i == 0 else i+1} domain",
                showarrow=False, font=dict(color="#64748b", size=12),
            )
            continue

        y = list(range(len(work)))
        # Neutral reference stems make magnitude readable without area encoding.
        for ypos, (_, row) in zip(y, work.iterrows()):
            fw = str(row["Framework"])
            value = float(row[metric])
            fig.add_trace(
                go.Scatter(
                    x=[0.0, value], y=[ypos, ypos], mode="lines",
                    line=dict(color="#cbd5e1", width=4),
                    hoverinfo="skip", showlegend=False,
                ), row=r, col=c,
            )
            fig.add_trace(
                go.Scatter(
                    x=[value], y=[ypos], mode="markers+text",
                    marker=dict(
                        size=14,
                        color=FRAMEWORK_COLORS.get(fw, "#64748b"),
                        line=dict(color="white", width=2),
                    ),
                    text=[f"{value:.3f}"], textposition="middle right",
                    customdata=[[fw]],
                    hovertemplate="%{customdata[0]}<br>Gap %{x:.4f}<extra></extra>",
                    showlegend=False,
                ), row=r, col=c,
            )

        fig.update_xaxes(
            range=[0, xmax], title_text="Observed gap ↓", zeroline=True,
            zerolinecolor="#94a3b8", gridcolor="#e2e8f0", row=r, col=c,
        )
        fig.update_yaxes(
            tickmode="array", tickvals=y, ticktext=work["Framework"].astype(str).tolist(),
            range=[len(work) - 0.45, -0.55], showgrid=False, row=r, col=c,
        )

    fig.update_layout(
        title="Criterion-wise fairness ranking · exact gaps, lower is better",
        height=640 if rows > 1 else 410,
        margin=dict(l=105, r=45, t=72, b=62),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
    )
    return fig


def fairness_cross_criterion_interval(fair: pd.DataFrame, metrics: Iterable[str]) -> go.Figure:
    """Show best, comparable mean and worst criterion for each framework.

    The interval length exposes sensitivity to the chosen fairness definition;
    short intervals are more criterion-robust, while leftward position means
    lower disparity.  No aggregate 'fairness score' is invented.
    """
    metric_list = [m for m in metrics if m in fair.columns]
    rows = []
    for _, row in fair.iterrows():
        fw = str(row.get("Framework"))
        values = pd.to_numeric(pd.Series([row.get(m) for m in metric_list]), errors="coerce").dropna()
        if values.empty:
            continue
        comparable_mean = pd.to_numeric(pd.Series([row.get("Comparable mean gap")]), errors="coerce").iloc[0]
        mean_value = float(comparable_mean) if pd.notna(comparable_mean) else float(values.mean())
        rows.append({
            "Framework": fw,
            "Best": float(values.min()),
            "Mean": mean_value,
            "Worst": float(values.max()),
            "Range": float(values.max() - values.min()),
        })
    work = pd.DataFrame(rows)
    if work.empty:
        return go.Figure()
    work = work.sort_values(["Worst", "Mean"], ascending=[True, True]).reset_index(drop=True)

    fig = go.Figure()
    for _, row in work.iterrows():
        fw = str(row["Framework"])
        fig.add_trace(go.Scatter(
            x=[row["Best"], row["Worst"]], y=[fw, fw], mode="lines",
            line=dict(color="#cbd5e1", width=7), hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=work["Best"], y=work["Framework"], mode="markers",
        name="Best criterion", marker=dict(size=10, symbol="line-ns-open", color="#64748b", line=dict(width=2)),
        hovertemplate="%{y}<br>Best criterion %{x:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=work["Mean"], y=work["Framework"], mode="markers",
        name="Comparable mean", marker=dict(size=13, symbol="circle", color="#2563eb", line=dict(color="white", width=2)),
        hovertemplate="%{y}<br>Comparable mean %{x:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=work["Worst"], y=work["Framework"], mode="markers+text",
        name="Worst criterion", marker=dict(size=12, symbol="diamond", color="#ef4444", line=dict(color="white", width=1.5)),
        text=[f"{v:.3f}" for v in work["Worst"]], textposition="middle right",
        hovertemplate="%{y}<br>Worst criterion %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Cross-criterion robustness · best → mean → worst",
        xaxis_title="Observed fairness gap ↓",
        yaxis_title="Framework",
        height=430,
        margin=dict(l=100, r=45, t=58, b=90),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(rangemode="tozero", gridcolor="#e2e8f0", zerolinecolor="#94a3b8")
    fig.update_yaxes(showgrid=False)
    return fig


def temporal_fairness_envelope_figure(results, metric_key: str, criterion_label: str) -> go.Figure:
    """Observed framework trajectories plus cross-framework median/IQR envelope.

    This keeps raw framework windows visible while adding a robust distribution
    summary at each stream position.  Predictive drift/refit markers are omitted
    on purpose; fairness-change events belong in the dedicated debt audit below.
    """
    records = []
    for result in results or []:
        fw = str(result.get("framework"))
        for pt in result.get("points") or []:
            sample = pd.to_numeric(pd.Series([pt.get("sample")]), errors="coerce").iloc[0]
            value = pd.to_numeric(pd.Series([pt.get(metric_key)]), errors="coerce").iloc[0]
            if pd.notna(sample) and pd.notna(value):
                records.append({"Framework": fw, "Sample": float(sample), "Gap": float(value)})
    df = pd.DataFrame(records)
    if df.empty:
        return go.Figure()

    stats = (
        df.groupby("Sample")["Gap"]
        .agg(
            Q1=lambda s: float(np.quantile(s, 0.25)),
            Median="median",
            Q3=lambda s: float(np.quantile(s, 0.75)),
        )
        .reset_index()
        .sort_values("Sample")
    )

    fig = go.Figure()
    # IQR envelope first, so framework evidence remains visually dominant.
    fig.add_trace(go.Scatter(
        x=stats["Sample"], y=stats["Q3"], mode="lines",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=stats["Sample"], y=stats["Q1"], mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(100,116,139,0.12)",
        name="Cross-framework IQR",
        hovertemplate="Sample %{x:,.0f}<br>Q1 %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=stats["Sample"], y=stats["Median"], mode="lines",
        line=dict(color="#0f172a", width=3, dash="dash"),
        name="Cross-framework median",
        hovertemplate="Sample %{x:,.0f}<br>Median gap %{y:.4f}<extra></extra>",
    ))

    for fw, part in df.groupby("Framework", sort=False):
        part = part.sort_values("Sample")
        fig.add_trace(go.Scatter(
            x=part["Sample"], y=part["Gap"], mode="lines+markers", name=str(fw),
            line=dict(color=FRAMEWORK_COLORS.get(str(fw), "#64748b"), width=2.4),
            marker=dict(size=6, color=FRAMEWORK_COLORS.get(str(fw), "#64748b"), line=dict(color="white", width=1)),
            hovertemplate=str(fw) + "<br>Sample %{x:,.0f}<br>Gap %{y:.4f}<extra></extra>",
        ))

    # Highlight the single worst observed window, without implying a threshold.
    worst = df.loc[df["Gap"].idxmax()]
    fig.add_trace(go.Scatter(
        x=[worst["Sample"]], y=[worst["Gap"]], mode="markers+text",
        marker=dict(size=15, symbol="x", color="#dc2626", line=dict(width=2)),
        text=["worst observed window"], textposition="top center",
        customdata=[[worst["Framework"]]],
        hovertemplate="%{customdata[0]}<br>Sample %{x:,.0f}<br>Worst gap %{y:.4f}<extra></extra>",
        name="Worst observed window", showlegend=False,
    ))

    fig.update_layout(
        title=f"{criterion_label} dynamics · observed trajectories + median/IQR envelope",
        xaxis_title="Stream sample",
        yaxis_title="Fairness gap ↓",
        height=500,
        margin=dict(l=70, r=30, t=62, b=92),
        legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center"),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(rangemode="tozero", gridcolor="#e2e8f0")
    return fig


def temporal_fairness_robustness_figure(summary: pd.DataFrame) -> go.Figure:
    """Two-panel research summary: mean/P95/worst interval + volatility.

    This replaces the old bubble risk map.  Every visual quantity is read from a
    position on an axis, which is more precise for research comparison.
    """
    if summary is None or summary.empty:
        return go.Figure()
    work = summary.copy()
    for col in ["Mean gap", "P95 gap", "Worst-window gap", "Temporal volatility"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["Mean gap", "P95 gap", "Worst-window gap", "Temporal volatility"])
    if work.empty:
        return go.Figure()
    work = work.sort_values(["Worst-window gap", "Mean gap"], ascending=True).reset_index(drop=True)

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        column_widths=[0.68, 0.32], horizontal_spacing=0.08,
        subplot_titles=("Distribution level across windows", "Temporal volatility"),
    )
    for _, row in work.iterrows():
        fw = str(row["Framework"])
        fig.add_trace(go.Scatter(
            x=[row["Mean gap"], row["Worst-window gap"]], y=[fw, fw], mode="lines",
            line=dict(color="#cbd5e1", width=7), hoverinfo="skip", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[0.0, row["Temporal volatility"]], y=[fw, fw], mode="lines",
            line=dict(color="#e2e8f0", width=6), hoverinfo="skip", showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[row["Temporal volatility"]], y=[fw], mode="markers+text",
            marker=dict(size=12, color=FRAMEWORK_COLORS.get(fw, "#64748b"), line=dict(color="white", width=1.5)),
            text=[f"{row['Temporal volatility']:.3f}"], textposition="middle right",
            customdata=[[fw]],
            hovertemplate="%{customdata[0]}<br>Volatility %{x:.4f}<extra></extra>",
            showlegend=False,
        ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=work["Mean gap"], y=work["Framework"], mode="markers",
        name="Mean", marker=dict(size=12, color="#2563eb", symbol="circle", line=dict(color="white", width=1.5)),
        hovertemplate="%{y}<br>Mean %{x:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=work["P95 gap"], y=work["Framework"], mode="markers",
        name="P95", marker=dict(size=11, color="#f59e0b", symbol="square", line=dict(color="white", width=1.5)),
        hovertemplate="%{y}<br>P95 %{x:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=work["Worst-window gap"], y=work["Framework"], mode="markers+text",
        name="Worst", marker=dict(size=12, color="#ef4444", symbol="diamond", line=dict(color="white", width=1.5)),
        text=[f"{v:.3f}" for v in work["Worst-window gap"]], textposition="middle right",
        hovertemplate="%{y}<br>Worst %{x:.4f}<extra></extra>",
    ), row=1, col=1)

    fig.update_xaxes(title_text="Fairness gap ↓", rangemode="tozero", gridcolor="#e2e8f0", row=1, col=1)
    fig.update_xaxes(title_text="Std. dev. across windows ↓", rangemode="tozero", gridcolor="#e2e8f0", row=1, col=2)
    fig.update_yaxes(title_text="Framework", showgrid=False, row=1, col=1)
    fig.update_layout(
        title="Temporal robustness profile · mean → P95 → worst, with volatility",
        height=430,
        margin=dict(l=100, r=45, t=70, b=92),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
