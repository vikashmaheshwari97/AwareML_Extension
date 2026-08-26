from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


FRAMEWORK_SYMBOLS = {
    "AutoStreamML": "circle",
    "AutoClass": "diamond",
    "EvoAutoML": "square",
    "OAML": "cross",
    "ChaCha": "x",
}

FRAMEWORK_COLORS = {
    "AutoStreamML": "#2563eb",
    "AutoClass": "#16a34a",
    "EvoAutoML": "#ea580c",
    "OAML": "#ca8a04",
    "ChaCha": "#9333ea",
}

OBJECTIVE_COLORS = {
    "accuracy": "#2563eb",
    "runtime": "#0ea5e9",
    "energy": "#10b981",
    "co2": "#14b8a6",
    "fairness": "#f59e0b",
    "interpretability": "#8b5cf6",
}


def _safe_numeric(series, fallback=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(fallback)


def _theme_mode() -> str:
    if st is None:
        return "Dark"
    try:
        state = st.session_state.get("awareml_state", {})
        mode = state.get("theme_mode", "System")
        if mode != "System":
            return mode
        base = st.get_option("theme.base")
        return "Light" if str(base).lower() == "light" else "Dark"
    except Exception:
        return "Dark"


def _tokens():
    if _theme_mode() == "Light":
        return {
            "panel": "#ffffff",
            "text": "#172033",
            "muted": "#64748b",
            "grid": "rgba(15,23,42,0.10)",
            "zero": "rgba(15,23,42,0.20)",
            "drift": "#dc2626",
            "refit": "#059669",
            "band": "rgba(220,38,38,0.06)",
            "refit_band": "rgba(5,150,105,0.06)",
        }
    return {
        "panel": "#0f172a",
        "text": "#f8fafc",
        "muted": "#cbd5e1",
        "grid": "rgba(226,232,240,0.12)",
        "zero": "rgba(226,232,240,0.24)",
        "drift": "#fb7185",
        "refit": "#34d399",
        "band": "rgba(251,113,133,0.09)",
        "refit_band": "rgba(52,211,153,0.07)",
    }


def _pad_axis(values, ratio=0.06, floor=1e-6):
    vals = [float(v) for v in values if pd.notna(v) and np.isfinite(v)]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    span = hi - lo
    pad = max(span * ratio, floor)
    if span <= floor:
        pad = max(abs(lo) * ratio, 1.0 if abs(lo) > 1 else 0.05)
    return [lo - pad, hi + pad]


def apply_research_layout(fig, *, height=380, legend="bottom", title=None, bottom_margin=None):
    """Shared Plotly layout used by Phase-9.6 charts to prevent legend/edge overlap."""
    t = _tokens()
    margin_bottom = bottom_margin if bottom_margin is not None else (78 if legend == "bottom" else 32)
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t["panel"],
        font={"color": t["text"], "size": 11},
        margin={"l": 48, "r": 30, "t": 52 if title else 26, "b": margin_bottom},
        title={"text": title or "", "font": {"size": 15, "color": t["text"]}, "x": 0.01, "xanchor": "left"},
        hoverlabel={"font": {"size": 11}},
    )
    if legend == "bottom":
        fig.update_layout(
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.18,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 9},
                "bgcolor": "rgba(0,0,0,0)",
                "itemwidth": 46,
            }
        )
    elif legend == "right":
        fig.update_layout(
            legend={
                "orientation": "v",
                "yanchor": "top",
                "y": 1,
                "xanchor": "left",
                "x": 1.02,
                "font": {"size": 9},
                "bgcolor": "rgba(0,0,0,0)",
            },
            margin={"l": 48, "r": 120, "t": 52 if title else 26, "b": 36},
        )
    else:
        fig.update_layout(showlegend=False)
    return fig


def decision_space_3d(ranked, selected_framework=None, normalized=False):
    if ranked is None or ranked.empty:
        return go.Figure()

    t = _tokens()
    frame = ranked.copy()
    for src, dst in [("accuracy", "Accuracy"), ("runtime", "Runtime"), ("energy", "Energy"), ("co2", "CO2"), ("utility", "Utility")]:
        frame[dst] = _safe_numeric(frame[src])

    if normalized:
        # All three axes now have the same interpretation: higher is better.
        acc = frame["Accuracy"]
        run = frame["Runtime"]
        ene = frame["Energy"]

        def desirability(s, direction):
            lo, hi = float(s.min()), float(s.max())
            if hi <= lo:
                return pd.Series([0.5] * len(s), index=s.index)
            u = (s - lo) / (hi - lo)
            return u if direction == "max" else 1.0 - u

        frame["X"] = desirability(acc, "max")
        frame["Y"] = desirability(run, "min")
        frame["Z"] = desirability(ene, "min")
        axis_titles = ("Accuracy desirability ↑", "Runtime efficiency ↑", "Energy efficiency ↑")
        x_range = y_range = z_range = [-0.05, 1.05]
    else:
        frame["X"] = frame["Accuracy"]
        frame["Y"] = frame["Runtime"]
        frame["Z"] = frame["Energy"]
        axis_titles = ("Predicted accuracy ↑", "Predicted runtime ↓", "Predicted energy ↓")
        x_range = _pad_axis(frame["X"], 0.10, 0.01)
        y_range = _pad_axis(frame["Y"], 0.12, 0.1)
        z_range = _pad_axis(frame["Z"], 0.12, 0.0001)

    co2 = frame["CO2"].to_numpy(dtype=float)
    lo, hi = float(np.nanmin(co2)), float(np.nanmax(co2))
    sizes = np.full(len(co2), 15.0) if hi - lo <= 1e-12 else 10.0 + 14.0 * (co2 - lo) / (hi - lo)

    fig = go.Figure()
    for pos, (_, row) in enumerate(frame.iterrows()):
        fw = str(row["framework"])
        is_selected = fw == selected_framework
        pareto = bool(row.get("pareto_efficient", False))
        color = FRAMEWORK_COLORS.get(fw, "#64748b")
        fig.add_trace(
            go.Scatter3d(
                x=[float(row["X"])], y=[float(row["Y"])], z=[float(row["Z"])],
                mode="markers+text",
                text=[fw], textposition="top center",
                textfont={"size": 11, "color": t["text"]},
                name=fw,
                marker={
                    "size": float(sizes[pos]) + (5 if is_selected else 0),
                    "symbol": FRAMEWORK_SYMBOLS.get(fw, "circle"),
                    "opacity": 0.94,
                    "color": color,
                    "line": {
                        "width": 5 if is_selected else 3 if pareto else 1.2,
                        "color": "#ffffff" if is_selected else "#22c55e" if pareto else color,
                    },
                },
                customdata=[[
                    int(row.get("rank", 0)), float(row["Utility"]), float(row["Accuracy"]),
                    float(row["Runtime"]), float(row["Energy"]), float(row["CO2"]), pareto,
                ]],
                hovertemplate=(
                    "<b>%{text}</b><br>Rank #%{customdata[0]}<br>Utility %{customdata[1]:.4f}<br>"
                    "Predicted accuracy %{customdata[2]:.4f}<br>Predicted runtime %{customdata[3]:.4g} s<br>"
                    "Predicted energy %{customdata[4]:.4g} kWh<br>Predicted CO₂ %{customdata[5]:.4g} kg<br>"
                    "Pareto efficient %{customdata[6]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=640,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=t["panel"],
        font={"color": t["text"], "size": 11}, showlegend=False,
        scene={
            "bgcolor": t["panel"],
            "aspectmode": "manual",
            "aspectratio": {"x": 1.12, "y": 1.0, "z": 0.90},
            "xaxis": {"title": axis_titles[0], "gridcolor": t["grid"], "backgroundcolor": t["panel"], "showbackground": True, "range": x_range, "color": t["text"], "showspikes": False},
            "yaxis": {"title": axis_titles[1], "gridcolor": t["grid"], "backgroundcolor": t["panel"], "showbackground": True, "range": y_range, "color": t["text"], "showspikes": False},
            "zaxis": {"title": axis_titles[2], "gridcolor": t["grid"], "backgroundcolor": t["panel"], "showbackground": True, "range": z_range, "color": t["text"], "showspikes": False},
            "camera": {"eye": {"x": 1.48, "y": 1.42, "z": 1.18}},
        },
    )
    return fig



def decision_space_3d_normalized(ranked, selected_framework=None):
    return decision_space_3d(ranked, selected_framework=selected_framework, normalized=True)

def ranking_bar(ranked):
    t = _tokens()
    frame = ranked.sort_values("utility", ascending=True).copy()
    fig = go.Figure(go.Bar(
        y=frame["framework"], x=frame["utility"], orientation="h",
        text=["#{:d}".format(int(v)) for v in frame["rank"]], textposition="inside",
        marker={"color": [FRAMEWORK_COLORS.get(str(fw), "#64748b") for fw in frame["framework"]]},
        hovertemplate="%{y}<br>Utility %{x:.4f}<extra></extra>",
    ))
    apply_research_layout(fig, height=335, legend="none", bottom_margin=50)
    fig.update_layout(
        xaxis={"title": "Preference utility", "gridcolor": t["grid"], "range": [0, max(1.0, float(frame["utility"].max()) * 1.05)]},
        yaxis={"title": None},
        margin={"l": 92, "r": 20, "t": 18, "b": 50},
    )
    return fig


def _extract_events(results):
    drift_positions = set()
    refit_positions = set()
    recovery_positions = set()
    for result in results:
        for event in result.get("drift_events") or []:
            try:
                if isinstance(event, dict):
                    event = event.get("sample") or event.get("index") or event.get("position")
                if event is not None:
                    drift_positions.add(int(event))
            except Exception:
                pass

        candidates = list(result.get("refit_events") or [])
        params = result.get("parameters") or {}
        candidates.extend(params.get("refit_events") or [])
        for episode in (result.get("drift_summary") or {}).get("episodes") or []:
            if isinstance(episode, dict):
                event = episode.get("refit_sample") or episode.get("retrain_sample") or episode.get("adaptation_sample")
                if event is not None:
                    candidates.append(event)
                recovery = (
                    episode.get("recovery_sample")
                    or episode.get("recovered_at_sample")
                    or episode.get("recovery_end_sample")
                    or episode.get("recovery_sample_index")
                )
                if recovery is not None:
                    try:
                        recovery_positions.add(int(recovery))
                    except Exception:
                        pass
        for event in candidates:
            try:
                if isinstance(event, dict):
                    event = event.get("sample") or event.get("index") or event.get("position")
                if event is not None:
                    refit_positions.add(int(event))
            except Exception:
                pass
    return sorted(drift_positions), sorted(refit_positions), sorted(recovery_positions)


def temporal_metric_figure(results, metric, title, y_title):
    """Research-grade temporal chart with a separate event strip to stop drift labels overlapping data."""
    t = _tokens()
    drift_positions, refit_positions, recovery_positions = _extract_events(results)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.16, 0.84],
    )

    all_x, all_y = [], []
    for result in results:
        fw = str(result.get("framework", "framework"))
        xs, ys = [], []
        for point in result.get("points") or []:
            if not isinstance(point, dict):
                continue
            x, y = point.get("sample"), point.get(metric)
            if x is not None and y is not None and pd.notna(y):
                try:
                    xs.append(float(x)); ys.append(float(y))
                except Exception:
                    pass
        if xs:
            all_x.extend(xs); all_y.extend(ys)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines+markers", name=fw,
                    line={"width": 2.8, "color": FRAMEWORK_COLORS.get(fw, "#64748b")},
                    marker={"size": 5.5, "color": FRAMEWORK_COLORS.get(fw, "#64748b")},
                    hovertemplate=fw + "<br>Sample %{x}<br>" + y_title + " %{y:.4f}<extra></extra>",
                ), row=2, col=1,
            )

    # Event strip: D1/D2... and R1/R2... no verbose boxes over the data area.
    if drift_positions:
        fig.add_trace(
            go.Scatter(
                x=drift_positions, y=[1.0] * len(drift_positions),
                mode="markers+text", name="Drift",
                marker={"symbol": "triangle-down", "size": 11, "color": t["drift"]},
                text=[f"D{i+1}" for i in range(len(drift_positions))],
                textposition="top center", textfont={"size": 9, "color": t["drift"]},
                hovertemplate="Drift %{text}<br>Sample %{x}<extra></extra>",
            ), row=1, col=1,
        )
    if refit_positions:
        fig.add_trace(
            go.Scatter(
                x=refit_positions, y=[0.55] * len(refit_positions),
                mode="markers+text", name="Refit / retrain",
                marker={"symbol": "diamond", "size": 10, "color": t["refit"]},
                text=[f"R{i+1}" for i in range(len(refit_positions))],
                textposition="top center", textfont={"size": 9, "color": t["refit"]},
                hovertemplate="Recorded refit/retrain %{text}<br>Sample %{x}<extra></extra>",
            ), row=1, col=1,
        )
    if recovery_positions:
        fig.add_trace(
            go.Scatter(
                x=recovery_positions, y=[0.08] * len(recovery_positions),
                mode="markers+text", name="Recovery",
                marker={"symbol": "circle", "size": 9, "color": "#14b8a6"},
                text=[f"REC{i+1}" for i in range(len(recovery_positions))],
                textposition="bottom center", textfont={"size": 9, "color": "#0f766e" if _theme_mode() == "Light" else "#5eead4"},
                hovertemplate="Observed recovery %{text}<br>Sample %{x}<extra></extra>",
            ), row=1, col=1,
        )

    if all_x:
        x_unique = sorted(set(all_x))
        step = x_unique[1] - x_unique[0] if len(x_unique) > 1 else max(1, x_unique[0] * 0.02)
        x_pad = max(step, (max(x_unique) - min(x_unique)) * 0.025)
        x_range = [min(x_unique) - x_pad, max(x_unique) + x_pad]
        for event in drift_positions:
            fig.add_vline(x=event, line_width=2.0, line_dash="dot", line_color=t["drift"], row=2, col=1)
            fig.add_vrect(x0=event, x1=min(x_range[1], event + max(step, (x_range[1]-x_range[0])*0.025)), fillcolor=t["band"], line_width=0, row=2, col=1)
        for event in refit_positions:
            fig.add_vline(x=event, line_width=2.0, line_dash="dash", line_color=t["refit"], row=2, col=1)
        for event in recovery_positions:
            fig.add_vline(x=event, line_width=1.6, line_dash="dashdot", line_color="#14b8a6", row=2, col=1)
        fig.update_xaxes(range=x_range, row=2, col=1)

    if all_y:
        fig.update_yaxes(range=_pad_axis(all_y, 0.08, 0.01), row=2, col=1)

    fig.update_yaxes(visible=False, range=[-0.35, 1.45], fixedrange=True, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="Stream sample", gridcolor=t["grid"], row=2, col=1)
    fig.update_yaxes(title_text=y_title, gridcolor=t["grid"], zerolinecolor=t["zero"], row=2, col=1)

    apply_research_layout(fig, height=455, legend="bottom", title=title, bottom_margin=92)
    fig.update_layout(
        margin={"l": 58, "r": 24, "t": 54, "b": 92},
        legend={"orientation": "h", "yanchor": "top", "y": -0.17, "xanchor": "center", "x": 0.5, "font": {"size": 9}},
    )

    if drift_positions and not refit_positions:
        fig.add_annotation(
            x=1.0, y=1.105, xref="paper", yref="paper",
            text="No explicit refit/retrain logged" + (" · recovery markers shown where available" if recovery_positions else ""),
            showarrow=False, xanchor="right",
            font={"size": 9, "color": t["muted"]},
        )
    return fig


def rai_metric_bar(frame, value_column, title, lower_is_better=False):
    """Horizontal ranking bars avoid crowded x-axis labels and text overlap."""
    t = _tokens()
    data = frame[["Framework", value_column]].dropna().copy()
    if data.empty:
        return go.Figure()
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.dropna().sort_values(value_column, ascending=not lower_is_better)
    colors = [FRAMEWORK_COLORS.get(str(fw), "#64748b") for fw in data["Framework"]]
    fig = go.Figure(go.Bar(
        x=data[value_column], y=data["Framework"], orientation="h",
        marker={"color": colors},
        text=[f"{v:.4g}" for v in data[value_column]],
        textposition="inside",
        hovertemplate="%{y}<br>%{x:.6g}<extra></extra>",
    ))
    apply_research_layout(fig, height=330, legend="none", title=title, bottom_margin=46)
    fig.update_layout(
        margin={"l": 96, "r": 24, "t": 50, "b": 46},
        xaxis={"title": "Lower is better" if lower_is_better else "Higher is better", "gridcolor": t["grid"]},
        yaxis={"title": None},
    )
    return fig


def faithfulness_components(summary):
    t = _tokens()
    names = ["Grounding", "Decision alignment", "Attribution alignment", "Counterfactual sensitivity", "Irrelevant invariance", "AEF"]
    keys = [
        "mean_grounding_validity", "mean_decision_alignment", "mean_attribution_alignment",
        "mean_counterfactual_sensitivity", "mean_irrelevant_invariance", "mean_evidence_fidelity_score",
    ]
    values = [float(summary.get(k, 0.0)) for k in keys]
    colors = ["#38bdf8", "#2563eb", "#6366f1", "#f59e0b", "#10b981", "#8b5cf6"]
    fig = go.Figure(go.Bar(
        x=names, y=values, marker={"color": colors},
        text=["{:.3f}".format(v) for v in values], textposition="inside",
        hovertemplate="%{x}<br>Score %{y:.3f}<extra></extra>",
    ))
    apply_research_layout(fig, height=370, legend="none", bottom_margin=68)
    fig.update_layout(
        margin={"l": 54, "r": 20, "t": 24, "b": 88},
        yaxis={"range": [0, 1.05], "title": "Score", "gridcolor": t["grid"]},
        xaxis={"title": None, "tickangle": -18},
    )
    return fig
