from __future__ import annotations

"""Research-grade visual builders for Sustainability and Responsible-AI views.

The functions in this module deliberately separate measured values from visual
normalization.  No aggregate sustainability or fairness score is manufactured.
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
    lo = float(finite.min())
    hi = float(finite.max())
    if abs(hi - lo) <= 1e-12:
        return pd.Series(0.5, index=values.index, dtype=float).where(vals.notna())
    score = (vals - lo) / (hi - lo)
    if not higher_is_better:
        score = 1.0 - score
    return score


def _pareto_high_y_low_x(frame: pd.DataFrame, x: str, y: str) -> pd.Series:
    work = frame[[x, y]].apply(pd.to_numeric, errors="coerce")
    out = []
    for i, row in work.iterrows():
        if row.isna().any():
            out.append(False)
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
        out.append(not dominated)
    return pd.Series(out, index=frame.index, dtype=bool)


def sustainability_decision_matrix(sdf: pd.DataFrame) -> go.Figure:
    """Framework × objective matrix using run-relative desirability for color.

    Cell labels remain exact measured values; color is only a within-run visual
    aid so unlike units are never numerically combined.
    """
    work = sdf.copy()
    specs = [
        ("Accuracy", "Accuracy ↑", True),
        ("Runtime s", "Runtime ↓", False),
        ("Energy kWh", "Energy ↓", False),
        ("CO₂ kg", "CO₂ ↓", False),
    ]
    for col, _, _ in specs:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")

    z = []
    text = []
    hover = []
    for _, label, higher in specs:
        col = next(c for c, l, h in specs if l == label)
        scores = _normalise(work[col], higher)
        z.append(scores.to_numpy(dtype=float))
        labels = []
        hovers = []
        for fw, value, score in zip(work["Framework"], work[col], scores):
            if pd.isna(value):
                labels.append("N/A")
                hovers.append(f"<b>{fw}</b><br>{label}: N/A")
            elif col == "Accuracy":
                labels.append(f"{float(value):.3f}")
                hovers.append(f"<b>{fw}</b><br>Accuracy: {float(value):.4f}<br>Run-relative desirability: {float(score):.3f}")
            elif col == "Runtime s":
                labels.append(f"{float(value):.2f}s")
                hovers.append(f"<b>{fw}</b><br>Runtime: {float(value):.3f} s<br>Run-relative desirability: {float(score):.3f}")
            elif col == "Energy kWh":
                labels.append(f"{float(value):.2e}")
                hovers.append(f"<b>{fw}</b><br>Energy: {float(value):.8f} kWh<br>Run-relative desirability: {float(score):.3f}")
            else:
                labels.append(f"{float(value):.2e}")
                hovers.append(f"<b>{fw}</b><br>CO₂: {float(value):.8f} kg<br>Run-relative desirability: {float(score):.3f}")
        text.append(labels)
        hover.append(hovers)

    z_arr = np.asarray(z, dtype=float).T
    text_arr = np.asarray(text, dtype=object).T
    hover_arr = np.asarray(hover, dtype=object).T
    fig = go.Figure(go.Heatmap(
        z=z_arr,
        x=[label for _, label, _ in specs],
        y=work["Framework"].astype(str).tolist(),
        text=text_arr,
        texttemplate="%{text}",
        customdata=hover_arr,
        hovertemplate="%{customdata}<extra></extra>",
        zmin=0,
        zmax=1,
        colorscale=[
            [0.0, "#eef2ff"],
            [0.35, "#bfdbfe"],
            [0.65, "#60a5fa"],
            [1.0, "#1d4ed8"],
        ],
        colorbar=dict(title="Run-relative<br>desirability", thickness=14),
        xgap=5,
        ygap=5,
    ))
    fig.update_layout(
        xaxis=dict(side="top", title=None),
        yaxis=dict(title=None, autorange="reversed"),
        margin=dict(l=112, r=74, t=38, b=20),
    )
    return fig


def accuracy_energy_frontier(sdf: pd.DataFrame) -> go.Figure:
    work = sdf[["Framework", "Accuracy", "Energy kWh", "CO₂ kg", "Runtime s"]].copy()
    for col in ["Accuracy", "Energy kWh", "CO₂ kg", "Runtime s"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["Accuracy", "Energy kWh"])
    fig = go.Figure()
    if work.empty:
        return fig

    work["Pareto"] = _pareto_high_y_low_x(work, "Energy kWh", "Accuracy")
    frontier = work[work["Pareto"]].sort_values("Energy kWh")
    if len(frontier) >= 2:
        fig.add_trace(go.Scatter(
            x=frontier["Energy kWh"], y=frontier["Accuracy"], mode="lines",
            line=dict(color="#0f766e", width=3, dash="dash"),
            name="Accuracy–energy frontier", hoverinfo="skip",
        ))

    # Run-relative compromise among exact Pareto points.
    e = work["Energy kWh"]
    a = work["Accuracy"]
    en = (e - e.min()) / (e.max() - e.min()) if e.max() > e.min() else pd.Series(0.0, index=work.index)
    an = (a.max() - a) / (a.max() - a.min()) if a.max() > a.min() else pd.Series(0.0, index=work.index)
    dist = np.sqrt(en ** 2 + an ** 2)
    eligible = work.index[work["Pareto"]]
    compromise_idx = dist.loc[eligible].idxmin() if len(eligible) else dist.idxmin()

    for idx, row in work.iterrows():
        fw = str(row["Framework"])
        compromise = idx == compromise_idx
        symbol = "star" if compromise else ("diamond" if bool(row["Pareto"]) else "circle")
        size = 19 if compromise else 13
        color = "#f59e0b" if compromise else FRAMEWORK_COLORS.get(fw, "#64748b")
        fig.add_trace(go.Scatter(
            x=[row["Energy kWh"]], y=[row["Accuracy"]], mode="markers+text",
            text=[fw], textposition="top center", showlegend=False,
            marker=dict(size=size, symbol=symbol, color=color,
                        line=dict(color="#0f172a" if row["Pareto"] or compromise else "white", width=2)),
            customdata=[[row["Runtime s"], row["CO₂ kg"]]],
            hovertemplate=(
                "<b>" + fw + "</b><br>Accuracy: %{y:.4f}<br>Energy: %{x:.8f} kWh"
                "<br>Runtime: %{customdata[0]:.3f} s<br>CO₂: %{customdata[1]:.8f} kg<extra></extra>"
            ),
        ))
    comp = work.loc[compromise_idx]
    fig.add_annotation(
        x=comp["Energy kWh"], y=comp["Accuracy"], text="run-relative compromise",
        showarrow=True, arrowhead=2, ax=64, ay=-42, bgcolor="rgba(255,255,255,.92)",
    )
    fig.update_layout(
        xaxis=dict(title="Measured energy (kWh) ↓", rangemode="tozero", gridcolor="rgba(148,163,184,.20)"),
        yaxis=dict(title="Prequential accuracy ↑", gridcolor="rgba(148,163,184,.20)"),
        margin=dict(l=72, r=34, t=18, b=60), showlegend=False,
    )
    return fig


def carbon_contribution_waterfall(sdf: pd.DataFrame) -> go.Figure:
    work = sdf[["Framework", "Energy kWh", "CO₂ kg", "Carbon intensity gCO2/kWh"]].copy()
    for c in ["Energy kWh", "CO₂ kg", "Carbon intensity gCO2/kWh"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["CO₂ kg"]).sort_values("CO₂ kg")
    fig = go.Figure()
    if work.empty:
        return fig
    total = float(work["CO₂ kg"].sum())
    x = work["Framework"].astype(str).tolist() + ["Experiment total"]
    y = work["CO₂ kg"].astype(float).tolist() + [total]
    measure = ["relative"] * len(work) + ["total"]
    custom = []
    for _, row in work.iterrows():
        custom.append([
            row["Energy kWh"],
            row["Carbon intensity gCO2/kWh"],
            (float(row["CO₂ kg"]) / total if total > 0 else np.nan),
        ])
    custom.append([float(work["Energy kWh"].sum(skipna=True)), np.nan, 1.0])
    fig.add_trace(go.Waterfall(
        orientation="v", x=x, y=y, measure=measure,
        text=[f"{v:.2e}" for v in y], textposition="outside",
        customdata=np.asarray(custom, dtype=object),
        connector=dict(line=dict(color="rgba(100,116,139,.45)", width=1.5)),
        increasing=dict(marker=dict(color="#2563eb")),
        totals=dict(marker=dict(color="#0f766e")),
        hovertemplate=(
            "<b>%{x}</b><br>CO₂ contribution: %{y:.8f} kg"
            "<br>Measured energy: %{customdata[0]:.8f} kWh"
            "<br>Carbon intensity: %{customdata[1]} gCO₂/kWh"
            "<br>Share of experiment CO₂: %{customdata[2]:.1%}<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis=dict(title=None), yaxis=dict(title="CO₂ contribution (kg)", rangemode="tozero", gridcolor="rgba(148,163,184,.20)"),
        margin=dict(l=72, r=28, t=18, b=70), showlegend=False,
    )
    return fig


def resource_tradeoff_statement(sdf: pd.DataFrame) -> dict[str, Any]:
    work = sdf[["Framework", "Accuracy", "Energy kWh", "CO₂ kg"]].copy()
    for c in ["Accuracy", "Energy kWh", "CO₂ kg"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["Accuracy", "Energy kWh"])
    if work.empty:
        return {}
    accuracy_leader = work.loc[work["Accuracy"].idxmax()]
    energy_leader = work.loc[work["Energy kWh"].idxmin()]
    delta_pp = float((accuracy_leader["Accuracy"] - energy_leader["Accuracy"]) * 100.0)
    ratio = None
    if float(energy_leader["Energy kWh"]) > 0:
        ratio = float(accuracy_leader["Energy kWh"] / energy_leader["Energy kWh"])
    return {
        "accuracy_framework": str(accuracy_leader["Framework"]),
        "energy_framework": str(energy_leader["Framework"]),
        "accuracy_delta_pp": delta_pp,
        "energy_ratio": ratio,
        "same_framework": str(accuracy_leader["Framework"]) == str(energy_leader["Framework"]),
    }


def fairness_group_frame(fair: Mapping[str, Any]) -> pd.DataFrame:
    perf = fair.get("group_performance") or {}
    rows = []
    for group, values in perf.items():
        row = {"Group": str(group)}
        for label, key in FAIRNESS_GROUP_METRICS.items():
            row[label] = _finite((values or {}).get(key))
        row["N"] = (fair.get("groups") or {}).get(str(group))
        rows.append(row)
    return pd.DataFrame(rows)


def group_metric_matrix_figure(fair: Mapping[str, Any], selected_group: str | None = None, selected_metric: str | None = None) -> go.Figure:
    frame = fairness_group_frame(fair)
    fig = go.Figure()
    if frame.empty:
        return fig
    metrics = list(FAIRNESS_GROUP_METRICS)
    groups = frame["Group"].astype(str).tolist()
    x, y, z, text, outline = [], [], [], [], []
    for _, row in frame.iterrows():
        for metric in metrics:
            value = _finite(row.get(metric))
            if value is None:
                continue
            x.append(metric); y.append(str(row["Group"])); z.append(value); text.append(f"{value:.3f}")
            outline.append(4 if (str(row["Group"]) == str(selected_group) and metric == selected_metric) else 1.5)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text", text=text, textposition="middle center",
        marker=dict(
            symbol="square", size=58, color=z, cmin=0, cmax=1,
            colorscale=[[0, "#eff6ff"], [0.5, "#60a5fa"], [1, "#1d4ed8"]],
            colorbar=dict(title="Raw metric<br>magnitude", thickness=14),
            line=dict(color="#0f172a", width=outline),
        ),
        customdata=np.column_stack([y, x, z]),
        hovertemplate="<b>Group %{customdata[0]}</b><br>%{customdata[1]}: %{customdata[2]:.4f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        xaxis=dict(title=None, side="top", categoryorder="array", categoryarray=metrics),
        yaxis=dict(title="Sensitive group", autorange="reversed", categoryorder="array", categoryarray=groups),
        margin=dict(l=90, r=72, t=45, b=24), dragmode=False,
    )
    return fig


def fairness_gap_dumbbell(fair: Mapping[str, Any], metric_label: str) -> go.Figure:
    frame = fairness_group_frame(fair)
    fig = go.Figure()
    if frame.empty or metric_label not in frame:
        return fig
    vals = pd.to_numeric(frame[metric_label], errors="coerce")
    valid = frame.loc[vals.dropna().index].copy()
    if valid.empty:
        return fig
    valid[metric_label] = pd.to_numeric(valid[metric_label], errors="coerce")
    low = valid.loc[valid[metric_label].idxmin()]
    high = valid.loc[valid[metric_label].idxmax()]
    x0, x1 = float(low[metric_label]), float(high[metric_label])
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[0, 0], mode="lines", line=dict(color="#94a3b8", width=7),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[x0], y=[0], mode="markers+text", text=[str(low["Group"])], textposition="bottom center",
        marker=dict(size=18, color="#2563eb", line=dict(color="white", width=2)),
        name="Lowest group",
        hovertemplate=f"<b>{low['Group']}</b><br>{metric_label}: %{{x:.4f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[x1], y=[0], mode="markers+text", text=[str(high["Group"])], textposition="top center",
        marker=dict(size=18, color="#ef4444", symbol="diamond", line=dict(color="white", width=2)),
        name="Highest group",
        hovertemplate=f"<b>{high['Group']}</b><br>{metric_label}: %{{x:.4f}}<extra></extra>",
    ))
    fig.add_annotation(
        x=(x0+x1)/2, y=0.15, text=f"gap = {x1-x0:.4f}", showarrow=False,
        font=dict(size=13, color="#0f172a"), bgcolor="rgba(255,255,255,.92)",
    )
    fig.update_layout(
        xaxis=dict(title=f"{metric_label} · group range", rangemode="tozero", gridcolor="rgba(148,163,184,.20)"),
        yaxis=dict(visible=False, range=[-0.35, 0.35]),
        margin=dict(l=40, r=34, t=28, b=58),
        legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center"),
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
    frame = pd.DataFrame(rows)
    palette = ["#2563eb", "#16a34a", "#f97316", "#7c3aed"]
    for color, (metric, group) in zip(palette, frame.groupby("Metric", sort=False)):
        fig.add_trace(go.Scatter(
            x=group["Sample"], y=group["Gap"], mode="lines+markers", name=metric,
            line=dict(width=2.5, color=color), marker=dict(size=7),
            hovertemplate="Sample %{x:,.0f}<br>Gap %{y:.4f}<extra>" + metric + "</extra>",
        ))
    for i, sample in enumerate(result.get("drift_events") or []):
        fig.add_vline(x=float(sample), line=dict(color="#dc2626", width=1.5, dash="dot"), opacity=0.45)
        if i == 0:
            fig.add_annotation(x=float(sample), y=1.02, yref="paper", text="predictive-change marker", showarrow=False, font=dict(size=10, color="#b91c1c"))
    fig.update_layout(
        xaxis=dict(title="Stream sample", gridcolor="rgba(148,163,184,.20)"),
        yaxis=dict(title="Observed fairness gap ↓", rangemode="tozero", gridcolor="rgba(148,163,184,.20)"),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(l=64, r=28, t=32, b=78),
    )
    return fig


def group_profile_figure(fair: Mapping[str, Any], selected_group: str) -> go.Figure:
    frame = fairness_group_frame(fair)
    fig = go.Figure()
    if frame.empty:
        return fig
    row = frame[frame["Group"].astype(str).eq(str(selected_group))]
    if row.empty:
        row = frame.iloc[[0]]
    row = row.iloc[0]
    metrics = list(FAIRNESS_GROUP_METRICS)
    values = [_finite(row.get(m)) for m in metrics]
    fig.add_trace(go.Bar(
        x=[v if v is not None else np.nan for v in values], y=metrics, orientation="h",
        marker=dict(color=["#2563eb", "#16a34a", "#ef4444", "#7c3aed", "#0f766e", "#f97316"]),
        text=["N/A" if v is None else f"{v:.3f}" for v in values], textposition="outside",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Observed group metric", range=[0, 1.05], gridcolor="rgba(148,163,184,.20)"),
        yaxis=dict(title=None, autorange="reversed"),
        margin=dict(l=122, r=58, t=18, b=54), showlegend=False,
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
            line=dict(color="#0f766e", width=3, dash="dash"), name="Nondominated frontier", hoverinfo="skip",
        ))
    for _, row in frame.iterrows():
        fw = row["Framework"]
        fig.add_trace(go.Scatter(
            x=[row["Gap"]], y=[row["Accuracy"]], mode="markers+text", text=[fw], textposition="top center",
            marker=dict(size=14, symbol="diamond" if row["Pareto"] else "circle", color=FRAMEWORK_COLORS.get(fw, "#64748b"),
                        line=dict(color="#0f172a" if row["Pareto"] else "white", width=2)),
            showlegend=False,
            hovertemplate=f"<b>{fw}</b><br>DP gap: %{{x:.4f}}<br>Accuracy: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        xaxis=dict(title="Demographic-parity gap ↓", rangemode="tozero", gridcolor="rgba(148,163,184,.20)"),
        yaxis=dict(title="Prequential accuracy ↑", gridcolor="rgba(148,163,184,.20)"),
        margin=dict(l=70, r=32, t=18, b=58), showlegend=False,
    )
    return fig


def rai_cross_dimension_matrix(frame: pd.DataFrame) -> go.Figure:
    """Framework × RAI evidence matrix; exact values in cells, ranks/desirability only for color."""
    work = frame.copy()
    specs = [
        ("Accuracy", "Accuracy ↑", True),
        ("DP gap", "DP gap ↓", False),
        ("XAI fidelity", "XAI fidelity ↑", True),
        ("Energy (kWh)", "Energy ↓", False),
        ("CO2 (kg)", "CO₂ ↓", False),
    ]
    zcols, textcols = [], []
    for col, label, higher in specs:
        vals = pd.to_numeric(work.get(col), errors="coerce")
        zcols.append(_normalise(vals, higher).to_numpy(dtype=float))
        textcols.append(["N/A" if pd.isna(v) else (f"{float(v):.3f}" if col in {"Accuracy", "DP gap", "XAI fidelity"} else f"{float(v):.2e}") for v in vals])
    fig = go.Figure(go.Heatmap(
        z=np.asarray(zcols).T,
        x=[label for _, label, _ in specs],
        y=work["Framework"].astype(str).tolist(),
        text=np.asarray(textcols, dtype=object).T,
        texttemplate="%{text}",
        zmin=0, zmax=1,
        colorscale=[[0, "#f8fafc"], [0.45, "#c7d2fe"], [1, "#4f46e5"]],
        colorbar=dict(title="Run-relative<br>desirability", thickness=14),
        xgap=5, ygap=5,
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(side="top"), yaxis=dict(autorange="reversed"),
        margin=dict(l=110, r=74, t=40, b=22),
    )
    return fig
