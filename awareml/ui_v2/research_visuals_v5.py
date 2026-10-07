from __future__ import annotations

"""Research-grade Plotly visualisations for AwareML Extension v5.

The module deliberately avoids bubble-size encodings for quantitative comparison.
Every primary quantity is encoded by position on an axis, with exact values exposed
through labels/hover. No metric is recomputed except transparent run-relative ranks,
normalisations used only for display, or explicitly-labelled Pareto/robustness summaries.
"""

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots import FRAMEWORK_COLORS


OBJECTIVE_COLORS_V5 = {
    "accuracy": "#2563eb",
    "runtime": "#0ea5e9",
    "energy": "#10b981",
    "co2": "#14b8a6",
    "fairness": "#f59e0b",
    "interpretability": "#8b5cf6",
}


def _finite(value) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _framework_order(values: Iterable[object]) -> list[str]:
    found = [str(v) for v in values if v is not None]
    canonical = [name for name in FRAMEWORK_COLORS if name in found]
    extras = [name for name in found if name not in canonical]
    out: list[str] = []
    for name in canonical + extras:
        if name not in out:
            out.append(name)
    return out


def _format_value(value: float) -> str:
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    if abs(value) < 0.1:
        return f"{value:.4f}"
    return f"{value:.3f}"


def utility_decomposition_figure(contrib: pd.DataFrame, ranking: pd.DataFrame) -> go.Figure:
    """Exact weighted-utility decomposition as stacked horizontal bars."""
    work = contrib.copy()
    if "framework" not in work.columns:
        raise ValueError("contrib must include a 'framework' column")
    order = ranking["framework"].astype(str).tolist() if "framework" in ranking else work["framework"].astype(str).tolist()
    work["framework"] = work["framework"].astype(str)
    work = work.set_index("framework").reindex(order).reset_index()
    objectives = [c for c in OBJECTIVE_COLORS_V5 if c in work.columns]

    fig = go.Figure()
    for objective in objectives:
        vals = pd.to_numeric(work[objective], errors="coerce").fillna(0.0)
        fig.add_trace(go.Bar(
            x=vals,
            y=work["framework"],
            orientation="h",
            name=objective.replace("co2", "CO₂").title(),
            marker=dict(color=OBJECTIVE_COLORS_V5[objective], line=dict(width=0)),
            customdata=np.column_stack([vals]),
            hovertemplate=(
                "<b>%{y}</b><br>Objective: " + objective.replace("co2", "CO₂").title()
                + "<br>Weighted contribution: %{x:.4f}<extra></extra>"
            ),
        ))

    totals = work[objectives].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    max_total = float(totals.max()) if len(totals) else 1.0
    for framework, total in zip(work["framework"], totals):
        fig.add_annotation(
            x=float(total) + max(0.005, max_total * 0.012),
            y=framework,
            text=f"Σ {total:.3f}",
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color="#0f172a"),
        )

    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Weighted utility contribution", rangemode="tozero", gridcolor="rgba(148,163,184,.22)"),
        yaxis=dict(title="", autorange="reversed"),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        margin=dict(l=110, r=72, t=18, b=90),
        hovermode="y unified",
    )
    return fig


def desirability_profile_figure(norm: pd.DataFrame) -> go.Figure:
    """Direction-aligned run-relative objective profiles without radar geometry."""
    if "framework" not in norm.columns:
        raise ValueError("norm must include a 'framework' column")
    objectives = [c for c in OBJECTIVE_COLORS_V5 if c in norm.columns]
    labels = [c.replace("co2", "CO₂").title() for c in objectives]
    fig = go.Figure()
    for _, row in norm.iterrows():
        fw = str(row["framework"])
        values = [float(row[c]) if pd.notna(row[c]) else np.nan for c in objectives]
        symbols = ["star" if np.isfinite(v) and v >= np.nanmax(values) - 1e-12 else "circle" for v in values]
        fig.add_trace(go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers",
            name=fw,
            line=dict(width=2.6, color=FRAMEWORK_COLORS.get(fw, "#64748b")),
            marker=dict(size=10, symbol=symbols, color=FRAMEWORK_COLORS.get(fw, "#64748b"), line=dict(color="white", width=1)),
            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.3f}<extra></extra>",
        ))
    fig.add_hrect(y0=0.75, y1=1.0, fillcolor="rgba(16,185,129,.055)", line_width=0, layer="below")
    fig.update_layout(
        xaxis=dict(title="Observed objective", showgrid=False),
        yaxis=dict(title="Run-relative desirability ↑", range=[-0.02, 1.04], dtick=0.25, gridcolor="rgba(148,163,184,.22)"),
        legend=dict(orientation="h", yanchor="top", y=-0.17, xanchor="center", x=0.5),
        margin=dict(l=64, r=24, t=18, b=92),
        hovermode="x unified",
    )
    return fig


def correlation_arc_figure(corr: pd.DataFrame, threshold: float = 0.20) -> go.Figure:
    """Arc diagram for objective dependence. Positive arcs rise, negative arcs fall."""
    labels = [str(c) for c in corr.columns]
    n = len(labels)
    x_pos = {label: i for i, label in enumerate(labels)}
    fig = go.Figure()

    # Stronger relations are drawn later so they remain visible.
    edges: list[tuple[float, float, str, str]] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            rho = _finite(corr.loc[a, b])
            if rho is None or abs(rho) < float(threshold):
                continue
            edges.append((abs(rho), rho, a, b))
    edges.sort()

    for magnitude, rho, a, b in edges:
        xa, xb = float(x_pos[a]), float(x_pos[b])
        t = np.linspace(0.0, 1.0, 60)
        x = xa + (xb - xa) * t
        direction = 1.0 if rho >= 0 else -1.0
        height = direction * (0.33 + 0.14 * abs(xb - xa) + 0.26 * magnitude)
        y = 4.0 * height * t * (1.0 - t)
        color = "rgba(220,38,38,{:.3f})".format(0.28 + 0.58 * magnitude) if rho >= 0 else "rgba(37,99,235,{:.3f})".format(0.28 + 0.58 * magnitude)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color, width=1.2 + 5.2 * magnitude),
            hoverinfo="skip",
            showlegend=False,
        ))
        if magnitude >= 0.65:
            fig.add_annotation(
                x=(xa + xb) / 2,
                y=float(np.max(y) if rho >= 0 else np.min(y)),
                text=f"ρ={rho:+.2f}",
                showarrow=False,
                yshift=10 if rho >= 0 else -10,
                font=dict(size=10, color="#334155"),
                bgcolor="rgba(255,255,255,.82)",
                borderpad=2,
            )

    for label in labels:
        display = label.replace("runtime_sec", "Runtime").replace("energy_kwh", "Energy").replace("co2_kg", "CO₂").replace("fairness_score", "Fairness").replace("interpretability_score", "Interpretability").replace("accuracy", "Accuracy")
        fig.add_trace(go.Scatter(
            x=[x_pos[label]], y=[0], mode="markers+text", text=[display], textposition="bottom center",
            marker=dict(size=26, color="white", line=dict(color="#0f172a", width=2.2)),
            hovertemplate=f"<b>{display}</b><extra></extra>", showlegend=False,
        ))

    # Legend semantics.
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#dc2626", width=4), name="Positive ρ"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#2563eb", width=4), name="Inverse ρ"))
    span = max(1.0, n - 1)
    fig.add_annotation(x=span / 2, y=0.98, text="Positive dependence", showarrow=False, font=dict(size=11, color="#991b1b"))
    fig.add_annotation(x=span / 2, y=-0.98, text="Inverse dependence", showarrow=False, font=dict(size=11, color="#1d4ed8"))
    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.45, max(0.45, n - 0.55)]),
        yaxis=dict(visible=False, range=[-1.12, 1.12]),
        legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5),
        margin=dict(l=18, r=18, t=18, b=54),
        hovermode=False,
    )
    return fig


def correlation_pair_ranking(corr: pd.DataFrame) -> go.Figure:
    rows: list[dict] = []
    labels = list(corr.columns)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            rho = _finite(corr.loc[a, b])
            if rho is None:
                continue
            pretty_a = str(a).replace("runtime_sec", "Runtime").replace("energy_kwh", "Energy").replace("co2_kg", "CO₂").replace("fairness_score", "Fairness").replace("interpretability_score", "Interpretability").replace("accuracy", "Accuracy")
            pretty_b = str(b).replace("runtime_sec", "Runtime").replace("energy_kwh", "Energy").replace("co2_kg", "CO₂").replace("fairness_score", "Fairness").replace("interpretability_score", "Interpretability").replace("accuracy", "Accuracy")
            rows.append({"pair": f"{pretty_a} ↔ {pretty_b}", "rho": rho, "abs": abs(rho)})
    work = pd.DataFrame(rows).sort_values("abs", ascending=True).tail(10) if rows else pd.DataFrame()
    fig = go.Figure()
    if work.empty:
        return fig
    colors = ["#dc2626" if v >= 0 else "#2563eb" for v in work["rho"]]
    fig.add_trace(go.Bar(
        x=work["rho"], y=work["pair"], orientation="h", marker_color=colors,
        text=[f"ρ={v:+.2f}" for v in work["rho"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Spearman ρ=%{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#64748b", line_width=1.2)
    fig.update_layout(
        xaxis=dict(title="Spearman ρ", range=[-1.12, 1.12], dtick=0.5, gridcolor="rgba(148,163,184,.22)"),
        yaxis=dict(title=""),
        margin=dict(l=142, r=58, t=16, b=58),
        showlegend=False,
    )
    return fig


def fairness_window_evidence_figure(results: Sequence[Mapping], metric_map: Mapping[str, str]) -> go.Figure:
    """Paper-style fairness figure: window observations + run aggregate diamonds.

    The FairStream paper uses datasets as rows. AwareML's interactive dashboard is
    scoped to one active dataset, so this version uses frameworks as rows and the
    repeated observations are actual stream-window fairness measurements. This keeps
    the visual grammar while avoiding fabricated multi-dataset evidence.
    """
    metrics = list(metric_map.items())[:4]
    titles = [label + " ↓" for label, _ in metrics]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, horizontal_spacing=0.12, vertical_spacing=0.22)
    frameworks = _framework_order([r.get("framework") for r in results])
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for metric_index, ((label, key), (row_i, col_i)) in enumerate(zip(metrics, positions)):
        metric_values: list[float] = []
        for result in results:
            fw = str(result.get("framework"))
            color = FRAMEWORK_COLORS.get(fw, "#64748b")
            points = []
            samples = []
            for point in result.get("points") or []:
                value = _finite(point.get(key))
                if value is None:
                    continue
                points.append(value)
                samples.append(point.get("sample"))
                metric_values.append(value)
            if points:
                fig.add_trace(go.Scatter(
                    x=points,
                    y=[fw] * len(points),
                    mode="markers",
                    marker=dict(size=6, color=color, opacity=0.28, symbol="circle", line=dict(width=0)),
                    customdata=np.array(samples, dtype=object).reshape(-1, 1),
                    hovertemplate="<b>" + fw + "</b><br>Window gap: %{x:.4f}<br>Sample: %{customdata[0]}<extra></extra>",
                    showlegend=False,
                ), row=row_i, col=col_i)

            aggregate = _finite((result.get("fairness") or {}).get(key))
            if aggregate is None and points:
                aggregate = float(np.mean(points))
            if aggregate is not None:
                metric_values.append(aggregate)
                fig.add_trace(go.Scatter(
                    x=[aggregate], y=[fw], mode="markers",
                    marker=dict(size=13, color=color, symbol="diamond", line=dict(color="#1e293b", width=1.4)),
                    name=fw,
                    legendgroup=fw,
                    showlegend=(metric_index == 0),
                    hovertemplate="<b>" + fw + "</b><br>Recorded run gap: %{x:.4f}<extra></extra>",
                ), row=row_i, col=col_i)

        max_value = max(metric_values) if metric_values else 0.10
        upper = max(0.02, max_value * 1.12)
        fig.update_xaxes(range=[0, upper], title_text="Gap ↓", gridcolor="rgba(148,163,184,.23)", row=row_i, col=col_i)
        fig.update_yaxes(
            categoryorder="array", categoryarray=list(reversed(frameworks)), title_text="" if col_i == 2 else "Framework",
            row=row_i, col=col_i,
        )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
        margin=dict(l=90, r=24, t=42, b=82),
        hovermode="closest",
    )
    return fig


def fairness_rank_bump_figure(fair: pd.DataFrame, metrics: Iterable[str]) -> go.Figure:
    metric_list = [m for m in metrics if m in fair.columns]
    fig = go.Figure()
    if not metric_list:
        return fig
    ranks = fair[metric_list].apply(pd.to_numeric, errors="coerce").rank(axis=0, method="min", ascending=True)
    labels = [m.replace("Demographic parity", "DP").replace("Equal opportunity", "EO").replace("Equalized odds", "EOdds").replace("Error-rate parity", "Error-rate") for m in metric_list]
    n = max(1, len(fair))
    for idx, row in fair.reset_index(drop=True).iterrows():
        fw = str(row["Framework"])
        y = [ranks.iloc[idx][m] for m in metric_list]
        fig.add_trace(go.Scatter(
            x=labels, y=y, mode="lines+markers", name=fw,
            line=dict(width=3, color=FRAMEWORK_COLORS.get(fw, "#64748b")),
            marker=dict(size=10, color=FRAMEWORK_COLORS.get(fw, "#64748b"), line=dict(color="white", width=1)),
            customdata=np.array([[row[m]] for m in metric_list], dtype=object),
            hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Rank: %{y:.0f}<br>Gap: %{customdata[0]:.4f}<extra></extra>",
        ))
    fig.update_layout(
        xaxis=dict(title="Fairness criterion", showgrid=False),
        yaxis=dict(title="Within-criterion rank ↓", autorange="reversed", tickmode="linear", dtick=1, range=[n + 0.35, 0.65], gridcolor="rgba(148,163,184,.2)"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=62, r=22, t=16, b=92),
        hovermode="x unified",
    )
    return fig


def _pareto_lower_lower(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            keep[i] = False
            continue
        for j in range(len(x)):
            if i == j or not np.isfinite(x[j]) or not np.isfinite(y[j]):
                continue
            if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                keep[i] = False
                break
    return keep


def fairness_robustness_frontier(fair: pd.DataFrame, metrics: Iterable[str]) -> go.Figure:
    metric_list = [m for m in metrics if m in fair.columns]
    work = fair[["Framework", *metric_list]].copy()
    vals = work[metric_list].apply(pd.to_numeric, errors="coerce")
    work["Mean gap"] = vals.mean(axis=1, skipna=True)
    work["Worst gap"] = vals.max(axis=1, skipna=True)
    work = work.dropna(subset=["Mean gap", "Worst gap"])
    fig = go.Figure()
    if work.empty:
        return fig
    pareto = _pareto_lower_lower(work["Mean gap"].to_numpy(float), work["Worst gap"].to_numpy(float))
    work["Pareto"] = pareto
    frontier = work[work["Pareto"]].sort_values("Mean gap")
    if len(frontier) >= 2:
        fig.add_trace(go.Scatter(
            x=frontier["Mean gap"], y=frontier["Worst gap"], mode="lines",
            line=dict(color="#0f766e", width=3, dash="dash"), name="Robustness frontier", hoverinfo="skip",
        ))
    for _, row in work.iterrows():
        fw = str(row["Framework"])
        fig.add_trace(go.Scatter(
            x=[row["Mean gap"]], y=[row["Worst gap"]], mode="markers+text", text=[fw], textposition="top center",
            name=fw, showlegend=False,
            marker=dict(size=16 if row["Pareto"] else 12, symbol="diamond" if row["Pareto"] else "circle",
                        color=FRAMEWORK_COLORS.get(fw, "#64748b"), line=dict(color="#0f172a" if row["Pareto"] else "white", width=2)),
            hovertemplate="<b>" + fw + "</b><br>Mean gap: %{x:.4f}<br>Worst criterion: %{y:.4f}<extra></extra>",
        ))
    max_x = max(float(work["Mean gap"].max()), 1e-6)
    max_y = max(float(work["Worst gap"].max()), 1e-6)
    fig.add_annotation(x=0.02 * max_x, y=0.06 * max_y, text="preferred direction ↙", showarrow=False, xanchor="left", font=dict(color="#0f766e", size=11))
    fig.update_layout(
        xaxis=dict(title="Mean observed gap ↓", rangemode="tozero", gridcolor="rgba(148,163,184,.22)"),
        yaxis=dict(title="Worst criterion gap ↓", rangemode="tozero", gridcolor="rgba(148,163,184,.22)"),
        margin=dict(l=66, r=26, t=16, b=58), showlegend=False,
    )
    return fig


def system_xai_ranked_panels(cdf: pd.DataFrame) -> go.Figure:
    specs = [
        ("Fidelity", "↑"), ("Stability", "↑"), ("Consistency", "↑"),
        ("Sensitivity", "↓"), ("Sparsity", "↔"),
    ]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{name} {direction}" for name, direction in specs] + [""], horizontal_spacing=0.11, vertical_spacing=0.23)
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    frameworks = _framework_order(cdf["Framework"].tolist())
    for (metric, direction), (row_i, col_i) in zip(specs, positions):
        work = cdf[["Framework", metric]].copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna(subset=[metric])
        ascending = direction == "↓"
        if direction == "↔":
            work["__order"] = work["Framework"].map({f: i for i, f in enumerate(frameworks)})
            work = work.sort_values("__order")
        else:
            work = work.sort_values(metric, ascending=ascending)
        for _, record in work.iterrows():
            fw = str(record["Framework"])
            value = float(record[metric])
            fig.add_trace(go.Scatter(
                x=[0, value], y=[fw, fw], mode="lines",
                line=dict(color="rgba(148,163,184,.50)", width=3), showlegend=False, hoverinfo="skip",
            ), row=row_i, col=col_i)
            fig.add_trace(go.Scatter(
                x=[value], y=[fw], mode="markers+text", text=[_format_value(value)], textposition="middle right",
                marker=dict(size=11, color=FRAMEWORK_COLORS.get(fw, "#64748b"), line=dict(color="white", width=1)),
                showlegend=False,
                hovertemplate="<b>" + fw + f"</b><br>{metric}: %{{x:.4f}}<extra></extra>",
            ), row=row_i, col=col_i)
        max_val = max(1.0, float(work[metric].max()) * 1.10) if not work.empty else 1.0
        fig.update_xaxes(range=[0, max_val], title_text=metric, gridcolor="rgba(148,163,184,.20)", row=row_i, col=col_i)
        fig.update_yaxes(title_text="", row=row_i, col=col_i)
    fig.update_layout(margin=dict(l=92, r=36, t=42, b=46), showlegend=False)
    return fig


def system_xai_rank_bump(cdf: pd.DataFrame) -> go.Figure:
    specs = [("Fidelity", False), ("Stability", False), ("Consistency", False), ("Sensitivity", True)]
    valid = cdf[["Framework"] + [m for m, _ in specs]].copy()
    for metric, _ in specs:
        valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
    fig = go.Figure()
    n = max(1, len(valid))
    for _, row in valid.iterrows():
        fw = str(row["Framework"])
        ys = []
        raw_values = []
        for metric, ascending in specs:
            s = valid[metric]
            ranks = s.rank(method="min", ascending=ascending)
            ys.append(ranks.loc[row.name])
            raw_values.append(row[metric])
        fig.add_trace(go.Scatter(
            x=[m for m, _ in specs], y=ys, mode="lines+markers", name=fw,
            line=dict(width=3, color=FRAMEWORK_COLORS.get(fw, "#64748b")),
            marker=dict(size=10, color=FRAMEWORK_COLORS.get(fw, "#64748b"), line=dict(color="white", width=1)),
            customdata=np.array(raw_values, dtype=float).reshape(-1, 1),
            hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Rank: %{y:.0f}<br>Raw value: %{customdata[0]:.4f}<extra></extra>",
        ))
    fig.update_layout(
        xaxis=dict(title="Directional XAI diagnostic", showgrid=False),
        yaxis=dict(title="Within-metric rank ↓", autorange="reversed", dtick=1, range=[n + 0.35, 0.65], gridcolor="rgba(148,163,184,.20)"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=62, r=22, t=16, b=92), hovermode="x unified",
    )
    return fig


def sustainability_resource_scoreboard(sdf: pd.DataFrame) -> go.Figure:
    specs = [("Runtime s", "Runtime (s) ↓"), ("Energy kWh", "Energy (kWh) ↓"), ("CO₂ kg", "CO₂ (kg) ↓")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[label for _, label in specs], horizontal_spacing=0.10)
    for col_i, (metric, _) in enumerate(specs, start=1):
        work = sdf[["Framework", metric]].copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna(subset=[metric]).sort_values(metric, ascending=True)
        for _, row in work.iterrows():
            fw = str(row["Framework"])
            value = float(row[metric])
            fig.add_trace(go.Scatter(
                x=[0, value], y=[fw, fw], mode="lines", line=dict(color="rgba(148,163,184,.48)", width=3), hoverinfo="skip", showlegend=False,
            ), row=1, col=col_i)
            fig.add_trace(go.Scatter(
                x=[value], y=[fw], mode="markers+text", text=[_format_value(value)], textposition="middle right",
                marker=dict(size=12, color=FRAMEWORK_COLORS.get(fw, "#64748b"), line=dict(color="white", width=1)), showlegend=False,
                hovertemplate="<b>" + fw + f"</b><br>{metric}: %{{x:.6g}}<extra></extra>",
            ), row=1, col=col_i)
        fig.update_xaxes(title_text=metric, rangemode="tozero", gridcolor="rgba(148,163,184,.20)", row=1, col=col_i)
        fig.update_yaxes(title_text="" if col_i > 1 else "Framework", row=1, col=col_i)
    fig.update_layout(margin=dict(l=96, r=48, t=42, b=58), showlegend=False)
    return fig


def sustainability_rank_bump(sdf: pd.DataFrame) -> go.Figure:
    metrics = ["Runtime s", "Energy kWh", "CO₂ kg"]
    work = sdf[["Framework"] + metrics].copy()
    for metric in metrics:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    fig = go.Figure()
    n = max(1, len(work))
    for _, row in work.iterrows():
        fw = str(row["Framework"])
        ys = [work[m].rank(method="min", ascending=True).loc[row.name] for m in metrics]
        raw = [row[m] for m in metrics]
        fig.add_trace(go.Scatter(
            x=["Runtime", "Energy", "CO₂"], y=ys, mode="lines+markers", name=fw,
            line=dict(width=3, color=FRAMEWORK_COLORS.get(fw, "#64748b")),
            marker=dict(size=10, color=FRAMEWORK_COLORS.get(fw, "#64748b"), line=dict(color="white", width=1)),
            customdata=np.array(raw, dtype=object).reshape(-1, 1),
            hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Efficiency rank: %{y:.0f}<br>Raw value: %{customdata[0]}<extra></extra>",
        ))
    fig.update_layout(
        xaxis=dict(title="Measured resource", showgrid=False),
        yaxis=dict(title="Efficiency rank ↓", autorange="reversed", dtick=1, range=[n + 0.35, 0.65], gridcolor="rgba(148,163,184,.20)"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=62, r=22, t=16, b=92), hovermode="x unified",
    )
    return fig


def sustainability_efficiency_frontier(sdf: pd.DataFrame) -> go.Figure:
    work = sdf[["Framework", "Runtime s", "Energy kWh", "CO₂ kg"]].copy()
    for metric in ["Runtime s", "Energy kWh", "CO₂ kg"]:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=["Runtime s", "Energy kWh"])
    fig = go.Figure()
    if work.empty:
        return fig
    pareto = _pareto_lower_lower(work["Runtime s"].to_numpy(float), work["Energy kWh"].to_numpy(float))
    work["Pareto"] = pareto

    # Closest Pareto point to the run-relative ideal (minimum runtime, minimum energy).
    r = work["Runtime s"]
    e = work["Energy kWh"]
    rn = (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else pd.Series(0.0, index=work.index)
    en = (e - e.min()) / (e.max() - e.min()) if e.max() > e.min() else pd.Series(0.0, index=work.index)
    distances = np.sqrt(rn ** 2 + en ** 2)
    eligible = work.index[work["Pareto"]]
    optimal_idx = distances.loc[eligible].idxmin() if len(eligible) else distances.idxmin()

    frontier = work[work["Pareto"]].sort_values("Runtime s")
    if len(frontier) >= 2:
        fig.add_trace(go.Scatter(x=frontier["Runtime s"], y=frontier["Energy kWh"], mode="lines", line=dict(color="#0f766e", width=3, dash="dash"), name="Resource Pareto frontier", hoverinfo="skip"))
    for idx, row in work.iterrows():
        fw = str(row["Framework"])
        optimal = idx == optimal_idx
        fig.add_trace(go.Scatter(
            x=[row["Runtime s"]], y=[row["Energy kWh"]], mode="markers+text", text=[fw], textposition="top center",
            name=fw, showlegend=False,
            marker=dict(size=20 if optimal else 13, symbol="star" if optimal else ("diamond" if row["Pareto"] else "circle"),
                        color="#f59e0b" if optimal else FRAMEWORK_COLORS.get(fw, "#64748b"), line=dict(color="#0f172a" if row["Pareto"] or optimal else "white", width=2)),
            customdata=[[row["CO₂ kg"]]],
            hovertemplate="<b>" + fw + "</b><br>Runtime: %{x:.3f} s<br>Energy: %{y:.6g} kWh<br>CO₂: %{customdata[0]} kg<extra></extra>",
        ))
    opt = work.loc[optimal_idx]
    fig.add_annotation(x=opt["Runtime s"], y=opt["Energy kWh"], text="2D resource compromise", showarrow=True, arrowhead=2, ax=54, ay=-44, bgcolor="rgba(255,255,255,.9)")
    fig.update_layout(
        xaxis=dict(title="Runtime (s) ↓", rangemode="tozero", gridcolor="rgba(148,163,184,.20)"),
        yaxis=dict(title="Measured energy (kWh) ↓", rangemode="tozero", gridcolor="rgba(148,163,184,.20)"),
        margin=dict(l=72, r=28, t=18, b=60), showlegend=False,
    )
    return fig


def carbon_intensity_consistency_figure(sdf: pd.DataFrame) -> go.Figure:
    work = sdf[["Framework", "Energy kWh", "CO₂ kg", "Carbon intensity gCO2/kWh"]].copy()
    for metric in ["Energy kWh", "CO₂ kg", "Carbon intensity gCO2/kWh"]:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work[(work["Energy kWh"] > 0) & work["CO₂ kg"].notna()].copy()
    work["Observed intensity"] = work["CO₂ kg"] * 1000.0 / work["Energy kWh"]
    fig = go.Figure()
    if work.empty:
        return fig
    for _, row in work.iterrows():
        fw = str(row["Framework"])
        observed = float(row["Observed intensity"])
        reported = _finite(row["Carbon intensity gCO2/kWh"])
        if reported is not None:
            fig.add_trace(go.Scatter(
                x=[reported, observed], y=[fw, fw], mode="lines",
                line=dict(color="rgba(148,163,184,.58)", width=4), showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=[reported], y=[fw], mode="markers", marker=dict(size=10, symbol="square-open", color="#475569", line=dict(width=2)),
                name="Reported intensity", legendgroup="reported", showlegend=(fw == str(work.iloc[0]["Framework"])),
                hovertemplate="<b>" + fw + "</b><br>Reported intensity: %{x:.2f} gCO₂/kWh<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=[observed], y=[fw], mode="markers+text", text=[f"{observed:.1f}"], textposition="middle right",
            marker=dict(size=11, symbol="circle", color=FRAMEWORK_COLORS.get(fw, "#64748b"), line=dict(color="white", width=1)),
            name="Observed CO₂/Energy intensity", legendgroup="observed", showlegend=(fw == str(work.iloc[0]["Framework"])),
            hovertemplate="<b>" + fw + "</b><br>Observed CO₂/Energy intensity: %{x:.2f} gCO₂/kWh<extra></extra>",
        ))
    fig.update_layout(
        xaxis=dict(title="Carbon intensity (gCO₂/kWh)", rangemode="tozero", gridcolor="rgba(148,163,184,.20)"),
        yaxis=dict(title="Framework"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=96, r=60, t=18, b=84),
    )
    return fig
