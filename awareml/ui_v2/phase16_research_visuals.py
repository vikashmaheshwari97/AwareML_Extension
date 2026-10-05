from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# Research-paper-oriented palette: high contrast, color-blind-friendlier than
# the previous single-blue charts, and consistent across the researcher views.
COLORS = {
    "correct": "#0F766E",
    "incorrect": "#E76F51",
    "accept": "#0E9F6E",
    "override": "#D97706",
    "reject": "#D14343",
    "blue": "#2563EB",
    "sky": "#38BDF8",
    "violet": "#7C3AED",
    "slate": "#64748B",
    "grid": "#E2E8F0",
    "muted": "#94A3B8",
    "text": "#0F172A",
    "paper": "rgba(0,0,0,0)",
    "panel": "#FFFFFF",
}

SOURCE_STAGE_LABELS = {
    "B": "Framework recommendation",
    "E": "Fairness explanation",
    "F_CHAT": "Conversational explanation",
    "F_XAI": "Feature-attribution explanation",
}

SOURCE_ORDER = [
    "Framework recommendation",
    "Fairness explanation",
    "Conversational explanation",
    "Feature-attribution explanation",
]

CONSTRUCT_LABELS = {
    "human_agency": "Human agency",
    "calibrated_reliance_support": "Calibration support",
    "explainability": "Explainability",
    "usability": "Usability",
    "global_reliance": "Global reliance",
}

CONSTRUCT_COLORS = {
    "Human agency": "#7C3AED",
    "Calibration support": "#2563EB",
    "Explainability": "#0284C7",
    "Usability": "#0F766E",
    "Global reliance": "#D97706",
}

ITEM_LABELS = {
    "critical_evaluation": ("Critical evaluation", "Human agency"),
    "freedom_to_override": ("Freedom to override / reject", "Human agency"),
    "final_control": ("Final decision control", "Human agency"),
    "rely_or_question": ("Know when to rely / question", "Calibration support"),
    "evidence_checking": ("Evidence checking", "Calibration support"),
    "explanation_usefulness": ("Explanation usefulness", "Explainability"),
    "interface_usability": ("Interface usability", "Usability"),
    "real_world_support_trust": ("Real-world support trust", "Global reliance"),
}

LIKERT_COLORS = {
    1: "#B91C1C",
    2: "#DC2626",
    3: "#F97316",
    4: "#CBD5E1",
    5: "#93C5FD",
    6: "#3B82F6",
    7: "#1D4ED8",
}


def _num(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _pct(value: Any) -> Optional[float]:
    value = _num(value)
    return None if value is None else 100.0 * value


def _as_frame(rows: Any) -> pd.DataFrame:
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    try:
        return pd.DataFrame(list(rows))
    except Exception:
        return pd.DataFrame()


def _mode_from_rows(rows: Any) -> Optional[str]:
    frame = _as_frame(rows)
    if frame.empty or "collection_mode" not in frame.columns:
        return None
    modes = frame["collection_mode"].dropna().astype(str).str.lower().unique().tolist()
    return modes[0] if len(modes) == 1 else None


def _small_n(n: int, rows: Any = None, label: str = "participants") -> None:
    mode = _mode_from_rows(rows)
    if mode == "final":
        prefix = "Main Study descriptive view"
    elif mode == "pilot":
        prefix = "Pilot visualization only"
    else:
        prefix = "Descriptive visualization"
    if int(n or 0) < 10:
        st.caption(
            f"{prefix} · {int(n or 0)} {label}. "
            "Interpret descriptively; small samples can move these estimates substantially."
        )
    else:
        st.caption(f"{prefix} · {int(n or 0)} {label}.")


def _style_figure(
    fig: go.Figure,
    *,
    height: int,
    left: int = 90,
    right: int = 35,
    top: int = 24,
    bottom: int = 55,
    showlegend: bool = False,
) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=left, r=right, t=top, b=bottom),
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["panel"],
        font=dict(family="Arial, sans-serif", size=12, color=COLORS["text"]),
        showlegend=showlegend,
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=12, font_family="Arial"),
    )
    fig.update_xaxes(
        gridcolor=COLORS["grid"],
        zeroline=False,
        linecolor="#CBD5E1",
        tickfont=dict(color="#64748B"),
        title_font=dict(color="#64748B"),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0)",
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color="#475569"),
        title_font=dict(color="#64748B"),
        automargin=True,
    )
    return fig


def _show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
            "scrollZoom": False,
        },
        key=key,
    )


def _condition_lollipop(
    values: Sequence[tuple[str, float]],
    *,
    title: str,
    x_title: str,
    domain: tuple[float, float],
    neutral: Optional[float] = None,
    suffix: str = "",
    key: str,
) -> None:
    st.markdown(f"**{title}**")
    fig = go.Figure()
    rows = [name for name, value in values if value is not None]
    valid = [(name, value) for name, value in values if value is not None]

    if neutral is not None:
        fig.add_vline(
            x=neutral,
            line_width=1.5,
            line_dash="dash",
            line_color="#94A3B8",
        )

    for name, value in valid:
        color = COLORS["correct"] if name.lower().startswith("correct") else COLORS["incorrect"]
        fig.add_trace(
            go.Scatter(
                x=[domain[0], value],
                y=[name, name],
                mode="lines",
                line=dict(color=color, width=6),
                opacity=0.18,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[name],
                mode="markers+text",
                marker=dict(size=18, color=color, line=dict(width=2, color="#FFFFFF")),
                text=[f"{value:.2f}{suffix}" if suffix != "%" else f"{value:.1f}%"],
                textposition="middle right",
                textfont=dict(size=13, color=color),
                hovertemplate=f"<b>{name}</b><br>{x_title}: %{{x:.2f}}{suffix}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_xaxes(range=[domain[0], domain[1]], title=x_title)
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=rows,
        autorange="reversed",
        title=None,
        tickfont=dict(size=13, color="#334155"),
    )
    _style_figure(fig, height=285, left=95, right=80, bottom=55)
    _show(fig, key)


def _reliance_stacked(responses: pd.DataFrame) -> None:
    if responses.empty or not {"correctness_condition", "decision_action"}.issubset(responses.columns):
        return

    actions = (
        responses.assign(
            Condition=responses["correctness_condition"].astype(str).str.title(),
            Action=responses["decision_action"].astype(str),
        )
        .groupby(["Condition", "Action"])
        .size()
        .reset_index(name="Trials")
    )

    all_rows = []
    for condition in ["Correct", "Incorrect"]:
        subtotal = int(actions.loc[actions["Condition"].eq(condition), "Trials"].sum())
        for action in ["Accept", "Override", "Reject"]:
            match = actions[
                actions["Condition"].eq(condition) & actions["Action"].eq(action)
            ]
            trials = int(match["Trials"].iloc[0]) if not match.empty else 0
            all_rows.append(
                {
                    "Condition": condition,
                    "Action": action,
                    "Trials": trials,
                    "Percent": (100.0 * trials / subtotal) if subtotal else 0.0,
                }
            )
    frame = pd.DataFrame(all_rows)

    st.markdown("**Accept / Override / Reject behavior**")
    st.caption(
        "Each row is a separate correctness condition. Segment width is the percentage of decisions in that condition."
    )

    fig = go.Figure()
    action_colors = {
        "Accept": COLORS["accept"],
        "Override": COLORS["override"],
        "Reject": COLORS["reject"],
    }
    for action in ["Accept", "Override", "Reject"]:
        part = frame[frame["Action"].eq(action)]
        fig.add_trace(
            go.Bar(
                x=part["Percent"],
                y=part["Condition"],
                name=action,
                orientation="h",
                marker=dict(color=action_colors[action], line=dict(color="#FFFFFF", width=1)),
                text=[
                    f"{value:.1f}%" if value >= 7 else ""
                    for value in part["Percent"].tolist()
                ],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + action
                    + ": %{x:.1f}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        bargap=0.48,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0,
            title=None,
        ),
    )
    fig.update_xaxes(range=[0, 100], dtick=10, title="Share of decisions (%)")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=["Correct", "Incorrect"],
        autorange="reversed",
        title=None,
        tickfont=dict(size=13),
    )
    _style_figure(fig, height=320, left=105, right=30, top=55, bottom=55, showlegend=True)
    _show(fig, "p16_v8_reliance_actions")


def _reliance_profile(result: Dict[str, Any]) -> None:
    rel = result.get("reliance") or {}
    over = result.get("overtrust") or {}
    under = result.get("undertrust") or {}
    rows = [
        ("Appropriate reliance", _pct(rel.get("appropriate_reliance_rate")), COLORS["accept"]),
        ("Over-trust", _pct(over.get("overtrust_rate")), COLORS["override"]),
        ("Under-trust", _pct(under.get("undertrust_rate")), COLORS["reject"]),
    ]
    rows = [(name, value, color) for name, value, color in rows if value is not None]
    if not rows:
        return

    st.markdown("**Reliance profile**")
    st.caption(
        "Appropriate reliance is desirable; over-trust and under-trust are error modes."
    )

    fig = go.Figure()
    for name, value, color in rows:
        fig.add_trace(
            go.Scatter(
                x=[0, value],
                y=[name, name],
                mode="lines",
                line=dict(color=color, width=7),
                opacity=0.17,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[name],
                mode="markers+text",
                marker=dict(size=17, color=color, line=dict(width=2, color="#FFFFFF")),
                text=[f"{value:.1f}%"],
                textposition="middle right",
                textfont=dict(color=color, size=13),
                hovertemplate=f"<b>{name}</b><br>%{{x:.1f}}%<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_xaxes(range=[0, 103], dtick=10, title="Rate (%)")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=[r[0] for r in rows],
        autorange="reversed",
        title=None,
    )
    _style_figure(fig, height=280, left=160, right=70)
    _show(fig, "p16_v8_reliance_profile")


def _source_dumbbell(result: Dict[str, Any]) -> None:
    source_rows = []
    for stage, values in (result.get("source_stages") or {}).items():
        values = values or {}
        correct = _num(values.get("mean_trust_correct"))
        incorrect = _num(values.get("mean_trust_incorrect"))
        overtrust = _pct(values.get("incorrect_acceptance_rate"))
        if correct is None or incorrect is None:
            continue
        source_rows.append(
            {
                "Source": SOURCE_STAGE_LABELS.get(str(stage), str(stage)),
                "Correct": correct,
                "Incorrect": incorrect,
                "Gap": correct - incorrect,
                "Incorrect acceptance": overtrust,
            }
        )

    frame = pd.DataFrame(source_rows)
    if frame.empty:
        return
    order = [x for x in SOURCE_ORDER if x in set(frame["Source"])]
    extra = [x for x in frame["Source"].tolist() if x not in order]
    order += extra
    frame["Source"] = pd.Categorical(frame["Source"], categories=order, ordered=True)
    frame = frame.sort_values("Source")

    st.markdown("**Trust discrimination by explanation source**")
    st.caption(
        "Each row is one explanation source. The line connects incorrect and correct mean trust; "
        "the Δ label is the correct-minus-incorrect trust gap."
    )

    fig = go.Figure()
    for _, row in frame.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Incorrect"], row["Correct"]],
                y=[row["Source"], row["Source"]],
                mode="lines",
                line=dict(color="#CBD5E1", width=7),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=frame["Incorrect"],
            y=frame["Source"],
            mode="markers",
            name="Incorrect",
            marker=dict(size=16, color=COLORS["incorrect"], line=dict(width=2, color="#FFFFFF")),
            customdata=frame[["Gap"]],
            hovertemplate="<b>%{y}</b><br>Incorrect trust: %{x:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["Correct"],
            y=frame["Source"],
            mode="markers",
            name="Correct",
            marker=dict(size=16, color=COLORS["correct"], line=dict(width=2, color="#FFFFFF")),
            hovertemplate="<b>%{y}</b><br>Correct trust: %{x:.2f}<extra></extra>",
        )
    )
    for _, row in frame.iterrows():
        fig.add_annotation(
            x=7.03,
            y=row["Source"],
            text=f"Δ {row['Gap']:+.2f}",
            showarrow=False,
            xanchor="left",
            font=dict(size=12, color="#334155"),
        )
    fig.add_vline(x=4, line_width=1.3, line_dash="dash", line_color="#94A3B8")
    fig.update_xaxes(range=[1, 7.55], dtick=0.5, title="Mean trust (1–7)")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=order,
        autorange="reversed",
        title=None,
        tickfont=dict(size=13),
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            title=None,
        )
    )
    _style_figure(
        fig,
        height=max(350, 70 * len(frame) + 90),
        left=205,
        right=95,
        top=55,
        showlegend=True,
    )
    _show(fig, "p16_v8_source_dumbbell")

    acceptance = frame.dropna(subset=["Incorrect acceptance"]).copy()
    if not acceptance.empty:
        st.markdown("**Incorrect-explanation acceptance by source**")
        st.caption("Lower values indicate less over-reliance on incorrect explanations.")
        fig2 = go.Figure(
            go.Bar(
                x=acceptance["Incorrect acceptance"],
                y=acceptance["Source"],
                orientation="h",
                marker=dict(color=COLORS["incorrect"]),
                text=[f"{v:.1f}%" for v in acceptance["Incorrect acceptance"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Accepted incorrect: %{x:.1f}%<extra></extra>",
            )
        )
        fig2.update_xaxes(range=[0, 105], dtick=10, title="Accepted incorrect explanations (%)")
        fig2.update_yaxes(
            categoryorder="array",
            categoryarray=order,
            autorange="reversed",
            title=None,
        )
        _style_figure(fig2, height=max(300, 60 * len(acceptance) + 80), left=205, right=70)
        _show(fig2, "p16_v8_source_overtrust")


def _expertise_views(result: Dict[str, Any]) -> None:
    rows = []
    for group, values in (result.get("expertise_groups") or {}).items():
        values = values or {}
        rows.append(
            {
                "Expertise group": str(group),
                "Participants": int(values.get("participants") or 0),
                "Trust gap": _num(values.get("mean_trust_gap")),
                "Over-trust (%)": _pct(values.get("overtrust_rate")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return

    st.markdown("**Calibration by expertise group**")
    st.caption(
        "Subgroup estimates are descriptive. Participant counts are shown in the labels so small groups are easy to identify."
    )

    c1, c2 = st.columns(2)
    with c1:
        d = frame.dropna(subset=["Trust gap"]).copy()
        if not d.empty:
            labels = [
                f"{g}  (n={n})"
                for g, n in zip(d["Expertise group"], d["Participants"])
            ]
            fig = go.Figure(
                go.Scatter(
                    x=d["Trust gap"],
                    y=labels,
                    mode="markers+text",
                    marker=dict(size=18, color=COLORS["blue"], line=dict(width=2, color="#FFFFFF")),
                    text=[f"{v:.2f}" for v in d["Trust gap"]],
                    textposition="middle right",
                    hovertemplate="<b>%{y}</b><br>Trust gap: %{x:.2f}<extra></extra>",
                )
            )
            fig.add_vline(x=0, line_dash="dash", line_color="#94A3B8")
            fig.update_xaxes(title="Correct − incorrect mean trust")
            fig.update_yaxes(title=None, automargin=True)
            _style_figure(fig, height=max(300, 70 * len(d) + 90), left=180, right=70)
            _show(fig, "p16_v8_expertise_gap")
    with c2:
        d = frame.dropna(subset=["Over-trust (%)"]).copy()
        if not d.empty:
            labels = [
                f"{g}  (n={n})"
                for g, n in zip(d["Expertise group"], d["Participants"])
            ]
            fig = go.Figure(
                go.Scatter(
                    x=d["Over-trust (%)"],
                    y=labels,
                    mode="markers+text",
                    marker=dict(size=18, color=COLORS["override"], line=dict(width=2, color="#FFFFFF")),
                    text=[f"{v:.1f}%" for v in d["Over-trust (%)"]],
                    textposition="middle right",
                    hovertemplate="<b>%{y}</b><br>Over-trust: %{x:.1f}%<extra></extra>",
                )
            )
            fig.update_xaxes(range=[0, 105], dtick=10, title="Accepted incorrect explanations (%)")
            fig.update_yaxes(title=None, automargin=True)
            _style_figure(fig, height=max(300, 70 * len(d) + 90), left=180, right=70)
            _show(fig, "p16_v8_expertise_overtrust")


def _style_and_timing(result: Dict[str, Any], responses: pd.DataFrame) -> None:
    style = result.get("style_vs_correctness") or {}
    regression = style.get("participant_centered_regression") or {}
    betas = regression.get("betas") or {}
    names = {
        "correctness_binary": "Actual correctness",
        "fluency_rating": "Perceived fluency",
        "perceived_confidence": "Confident-sounding language",
        "order_index": "Trial order",
    }
    rows = [
        {"Predictor": names.get(key, key), "Coefficient": _num(value)}
        for key, value in betas.items()
        if _num(value) is not None
    ]
    frame = pd.DataFrame(rows)

    if not frame.empty:
        st.markdown("**What is associated with trust?**")
        st.caption(
            "Participant-centered standardized coefficients. Values farther from zero indicate a stronger conditional association with trust."
        )
        fig = go.Figure()
        for _, row in frame.iterrows():
            value = row["Coefficient"]
            color = COLORS["violet"] if value >= 0 else COLORS["slate"]
            fig.add_trace(
                go.Scatter(
                    x=[0, value],
                    y=[row["Predictor"], row["Predictor"]],
                    mode="lines",
                    line=dict(color=color, width=7),
                    opacity=0.2,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[value],
                    y=[row["Predictor"]],
                    mode="markers+text",
                    marker=dict(size=17, color=color, line=dict(width=2, color="#FFFFFF")),
                    text=[f"{value:+.3f}"],
                    textposition="middle right" if value >= 0 else "middle left",
                    textfont=dict(color=color),
                    hovertemplate="<b>%{y}</b><br>Coefficient: %{x:+.3f}<extra></extra>",
                    showlegend=False,
                )
            )
        fig.add_vline(x=0, line_width=1.4, line_color="#64748B")
        pad = max(0.08, abs(frame["Coefficient"]).max() * 0.2)
        fig.update_xaxes(
            range=[min(-pad, frame["Coefficient"].min() - pad), max(pad, frame["Coefficient"].max() + pad)],
            title="Participant-centered standardized coefficient",
        )
        fig.update_yaxes(title=None, autorange="reversed")
        _style_figure(fig, height=max(300, 62 * len(frame) + 80), left=210, right=80)
        _show(fig, "p16_v8_style_coefficients")

    st.caption(
        "Supportive analysis only. The primary result remains the within-participant correct-minus-incorrect trust gap."
    )

    if responses.empty or not {"correctness_condition", "response_time_sec"}.issubset(responses.columns):
        return
    rt = responses[["correctness_condition", "response_time_sec"]].copy()
    rt["Condition"] = rt["correctness_condition"].astype(str).str.title()
    rt["Response time (s)"] = pd.to_numeric(rt["response_time_sec"], errors="coerce")
    rt = rt[rt["Response time (s)"] > 0].dropna()
    if rt.empty:
        return

    st.markdown("**Response-time distribution**")
    max_rt = float(rt["Response time (s)"].max())
    median_rt = float(rt["Response time (s)"].median())
    use_log = median_rt > 0 and max_rt / median_rt >= 12
    if use_log:
        st.caption(
            "A logarithmic x-axis is used because a small number of long pauses would otherwise compress the typical response times."
        )
    else:
        st.caption("Box-and-point view preserves the individual trial times while showing the distribution.")

    fig = go.Figure()
    for condition, color in [("Correct", COLORS["correct"]), ("Incorrect", COLORS["incorrect"])]:
        values = rt.loc[rt["Condition"].eq(condition), "Response time (s)"]
        if values.empty:
            continue
        fig.add_trace(
            go.Box(
                x=values,
                y=[condition] * len(values),
                name=condition,
                orientation="h",
                boxpoints="all",
                jitter=0.32,
                pointpos=0,
                marker=dict(size=6, color=color, opacity=0.55),
                line=dict(color=color, width=2),
                fillcolor="rgba(255,255,255,0)",
                hovertemplate=f"<b>{condition}</b><br>%{{x:.1f}} s<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_xaxes(
        type="log" if use_log else "linear",
        title="Response time (seconds{})".format(", log scale" if use_log else ""),
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=["Correct", "Incorrect"],
        autorange="reversed",
        title=None,
    )
    _style_figure(fig, height=330, left=105, right=35)
    _show(fig, "p16_v8_response_time")


def render_trust_calibration_visuals(
    result: Dict[str, Any],
    responses: Optional[pd.DataFrame] = None,
    *_args: Any,
    **_kwargs: Any,
) -> None:
    if str(result.get("status") or "") != "ok":
        return

    response_frame = _as_frame(responses)
    n = int(result.get("participants_completed_valid") or 0)

    st.markdown("#### Visual calibration dashboard")
    st.caption(
        "Researcher-only views designed for fast interpretation of calibration, reliance, subgroup patterns and timing. "
        "The underlying analysis and estimands are unchanged."
    )
    _small_n(n, response_frame)

    condition = result.get("condition_summary") or {}
    correct = condition.get("correct") or {}
    incorrect = condition.get("incorrect") or {}

    overview_tab, reliance_tab, source_tab, style_tab = st.tabs(
        ["Calibration overview", "Reliance behavior", "Expertise & sources", "Style & timing"]
    )

    with overview_tab:
        st.markdown(
            "**Core question. Do participants assign more trust to correct explanations and reduce reliance on incorrect explanations?**"
        )
        c1, c2 = st.columns(2)
        with c1:
            _condition_lollipop(
                [
                    ("Correct", _num(correct.get("mean_trust"))),
                    ("Incorrect", _num(incorrect.get("mean_trust"))),
                ],
                title="Trust by explanation correctness",
                x_title="Mean trust (1–7)",
                domain=(1, 7.25),
                neutral=4,
                key="p16_v8_correctness_trust",
            )
            st.caption("Dashed line = neutral midpoint (4). A larger correct–incorrect separation indicates stronger calibration.")
        with c2:
            _condition_lollipop(
                [
                    ("Correct", _pct(correct.get("acceptance_rate"))),
                    ("Incorrect", _pct(incorrect.get("acceptance_rate"))),
                ],
                title="Acceptance by explanation correctness",
                x_title="Acceptance rate (%)",
                domain=(0, 105),
                suffix="%",
                key="p16_v8_correctness_acceptance",
            )
            st.caption("For incorrect explanations, lower acceptance indicates less over-reliance.")

        paired = ((result.get("primary_calibration") or {}).get("paired_gap") or {})
        gap = _num(paired.get("mean_gap"))
        ci = paired.get("ci95") or []
        if gap is not None:
            text = f"**Within-participant trust gap:** {gap:.2f}"
            if len(ci) >= 2 and _num(ci[0]) is not None and _num(ci[1]) is not None:
                text += f" · bootstrap 95% CI [{float(ci[0]):.2f}, {float(ci[1]):.2f}]"
            st.info(text)

    with reliance_tab:
        _reliance_stacked(response_frame)
        _reliance_profile(result)

    with source_tab:
        _expertise_views(result)
        _source_dumbbell(result)

    with style_tab:
        _style_and_timing(result, response_frame)


def _construct_lollipop(construct_frame: pd.DataFrame) -> None:
    if construct_frame.empty:
        return
    st.markdown("**Construct profile**")
    st.caption(
        "Each marker is the mean construct score. Stems begin at the neutral midpoint (4), so direction and distance are immediately visible."
    )

    frame = construct_frame.copy().sort_values("Mean response", ascending=False)
    fig = go.Figure()
    for _, row in frame.iterrows():
        name = str(row["Construct"])
        value = float(row["Mean response"])
        color = CONSTRUCT_COLORS.get(name, COLORS["blue"])
        fig.add_trace(
            go.Scatter(
                x=[4, value],
                y=[name, name],
                mode="lines",
                line=dict(color=color, width=7),
                opacity=0.20,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[name],
                mode="markers+text",
                marker=dict(size=18, color=color, line=dict(width=2, color="#FFFFFF")),
                text=[f"{value:.2f}"],
                textposition="middle right" if value >= 4 else "middle left",
                textfont=dict(color=color, size=13),
                hovertemplate=f"<b>{name}</b><br>Mean: %{{x:.2f}}<extra></extra>",
                showlegend=False,
            )
        )
    fig.add_vline(x=4, line_width=1.4, line_dash="dash", line_color="#64748B")
    fig.update_xaxes(range=[1, 7.28], dtick=0.5, title="Mean response (1–7)")
    fig.update_yaxes(autorange="reversed", title=None, tickfont=dict(size=13))
    _style_figure(fig, height=max(330, 58 * len(frame) + 80), left=180, right=75)
    _show(fig, "p16_v8_secondary_construct")


def _item_lollipop(item_frame: pd.DataFrame) -> None:
    if item_frame.empty:
        return
    st.markdown("**Item-level profile (1–7)**")
    st.caption(
        "The Y-axis names each questionnaire item; the marker is its mean response. "
        "Stems originate at the neutral midpoint (4), and color identifies the construct."
    )

    frame = item_frame.copy().sort_values(["Construct", "Mean response"], ascending=[True, False])
    fig = go.Figure()
    for _, row in frame.iterrows():
        item = str(row["Item"])
        construct = str(row["Construct"])
        value = float(row["Mean response"])
        color = CONSTRUCT_COLORS.get(construct, COLORS["blue"])
        fig.add_trace(
            go.Scatter(
                x=[4, value],
                y=[item, item],
                mode="lines",
                line=dict(color=color, width=6),
                opacity=0.18,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[item],
                mode="markers+text",
                marker=dict(size=16, color=color, line=dict(width=2, color="#FFFFFF")),
                text=[f"{value:.2f}"],
                textposition="middle right" if value >= 4 else "middle left",
                textfont=dict(color=color),
                customdata=[[construct]],
                hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Mean: %{x:.2f}<extra></extra>",
                showlegend=False,
            )
        )
    fig.add_vline(x=4, line_width=1.3, line_dash="dash", line_color="#64748B")
    fig.update_xaxes(range=[1, 7.3], dtick=0.5, title="Mean response (1–7)")
    fig.update_yaxes(autorange="reversed", title=None, tickfont=dict(size=12))
    _style_figure(fig, height=max(430, 58 * len(frame) + 70), left=235, right=75)
    _show(fig, "p16_v8_secondary_items")

    legend_cols = st.columns(len(CONSTRUCT_COLORS))
    for (name, color), col in zip(CONSTRUCT_COLORS.items(), legend_cols):
        with col:
            st.markdown(
                f'<div style="font-size:.78rem;color:#475569;"><span style="display:inline-block;width:9px;height:9px;'
                f'border-radius:50%;background:{color};margin-right:6px;"></span>{name}</div>',
                unsafe_allow_html=True,
            )


def _likert_diverging(rows: Any) -> None:
    frame = _as_frame(rows)
    if frame.empty:
        return

    distribution_rows = []
    for key, (label, _construct) in ITEM_LABELS.items():
        if key not in frame.columns:
            continue
        values = pd.to_numeric(frame[key], errors="coerce").dropna().astype(int)
        total = len(values)
        if not total:
            continue
        counts = values.value_counts().reindex(range(1, 8), fill_value=0)
        for rating, count in counts.items():
            distribution_rows.append(
                {
                    "Item": label,
                    "Rating": int(rating),
                    "Percent": 100.0 * int(count) / total,
                }
            )

    distribution = pd.DataFrame(distribution_rows)
    if distribution.empty:
        return

    st.markdown("**Likert response distribution**")
    st.caption(
        "Diverging 100% Likert profile: disagreement (1–3) extends left, neutral (4) is centered, and agreement (5–7) extends right."
    )

    item_order = [
        ITEM_LABELS[key][0]
        for key in ITEM_LABELS
        if ITEM_LABELS[key][0] in set(distribution["Item"])
    ]

    fig = go.Figure()
    # Negative side: 3 nearest center, 1 farthest left.
    for rating in [3, 2, 1]:
        part = distribution[distribution["Rating"].eq(rating)].set_index("Item").reindex(item_order)
        values = -part["Percent"].fillna(0).to_numpy()
        fig.add_trace(
            go.Bar(
                x=values,
                y=item_order,
                orientation="h",
                name=str(rating),
                marker=dict(color=LIKERT_COLORS[rating], line=dict(color="#FFFFFF", width=0.7)),
                customdata=abs(values),
                hovertemplate="<b>%{y}</b><br>Rating "
                + str(rating)
                + ": %{customdata:.1f}%<extra></extra>",
            )
        )

    neutral = distribution[distribution["Rating"].eq(4)].set_index("Item").reindex(item_order)["Percent"].fillna(0).to_numpy()
    fig.add_trace(
        go.Bar(
            x=-neutral / 2.0,
            y=item_order,
            orientation="h",
            name="4",
            marker=dict(color=LIKERT_COLORS[4], line=dict(color="#FFFFFF", width=0.7)),
            customdata=neutral,
            hovertemplate="<b>%{y}</b><br>Rating 4: %{customdata:.1f}%<extra></extra>",
            legendgroup="4",
        )
    )
    fig.add_trace(
        go.Bar(
            x=neutral / 2.0,
            y=item_order,
            orientation="h",
            name="4",
            marker=dict(color=LIKERT_COLORS[4], line=dict(color="#FFFFFF", width=0.7)),
            customdata=neutral,
            hovertemplate="<b>%{y}</b><br>Rating 4: %{customdata:.1f}%<extra></extra>",
            legendgroup="4",
            showlegend=False,
        )
    )

    for rating in [5, 6, 7]:
        part = distribution[distribution["Rating"].eq(rating)].set_index("Item").reindex(item_order)
        values = part["Percent"].fillna(0).to_numpy()
        fig.add_trace(
            go.Bar(
                x=values,
                y=item_order,
                orientation="h",
                name=str(rating),
                marker=dict(color=LIKERT_COLORS[rating], line=dict(color="#FFFFFF", width=0.7)),
                customdata=values,
                hovertemplate="<b>%{y}</b><br>Rating "
                + str(rating)
                + ": %{customdata:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="relative",
        bargap=0.28,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            title="Likert rating",
        ),
    )
    fig.add_vline(x=0, line_width=1.2, line_color="#64748B")
    fig.update_xaxes(
        range=[-100, 100],
        tickvals=[-100, -75, -50, -25, 0, 25, 50, 75, 100],
        ticktext=["100%", "75%", "50%", "25%", "0", "25%", "50%", "75%", "100%"],
        title="← Disagreement     Share of responses     Agreement →",
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=item_order,
        autorange="reversed",
        title=None,
        tickfont=dict(size=12),
    )
    _style_figure(
        fig,
        height=max(470, 52 * len(item_order) + 120),
        left=235,
        right=35,
        top=70,
        showlegend=True,
    )
    _show(fig, "p16_v8_secondary_likert")


def _expertise_matrix(summary: Dict[str, Any]) -> None:
    by_expertise = pd.DataFrame(summary.get("by_expertise") or [])
    construct_keys = [key for key in CONSTRUCT_LABELS if key in by_expertise.columns]
    if by_expertise.empty or not construct_keys or "expertise_group" not in by_expertise.columns:
        return

    groups = by_expertise["expertise_group"].astype(str).tolist()
    participants = (
        by_expertise["participants"].fillna(0).astype(int).tolist()
        if "participants" in by_expertise.columns
        else [0] * len(groups)
    )
    xlabels = [f"{group}<br>n={n}" for group, n in zip(groups, participants)]
    ylabels = [CONSTRUCT_LABELS[key] for key in construct_keys]

    matrix = []
    for key in construct_keys:
        matrix.append(
            pd.to_numeric(by_expertise[key], errors="coerce").tolist()
        )

    st.markdown("**Novice / practitioner / expert comparison**")
    st.caption(
        "Annotated construct-by-expertise matrix. Color and printed values both encode the 1–7 mean; participant counts appear under each group name."
    )

    colorscale = [
        [0.00, "#FEE2E2"],
        [0.20, "#FDBA74"],
        [0.50, "#F1F5F9"],
        [0.75, "#93C5FD"],
        [1.00, "#0F766E"],
    ]
    text = [
        ["" if value is None or pd.isna(value) else f"{float(value):.2f}" for value in row]
        for row in matrix
    ]

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=xlabels,
            y=ylabels,
            zmin=1,
            zmax=7,
            colorscale=colorscale,
            colorbar=dict(
                title="Mean<br>(1–7)",
                tickvals=[1, 2, 3, 4, 5, 6, 7],
                thickness=16,
                len=0.78,
            ),
            text=text,
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=13),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Mean: %{z:.2f}<extra></extra>",
            xgap=4,
            ygap=4,
        )
    )
    fig.update_xaxes(side="bottom", title="Expertise group", tickfont=dict(size=12))
    fig.update_yaxes(autorange="reversed", title=None, tickfont=dict(size=12))
    _style_figure(
        fig,
        height=max(360, 58 * len(ylabels) + 120),
        left=170,
        right=75,
        bottom=80,
    )
    _show(fig, "p16_v8_secondary_expertise")
    st.caption(
        "Expertise comparisons remain secondary/descriptive unless the Main Study is separately powered for subgroup interactions."
    )


def render_secondary_visuals(
    summary: Dict[str, Any],
    rows: Any,
    *_args: Any,
    **_kwargs: Any,
) -> None:
    response_rows = list(rows or [])
    n = int(summary.get("responses") or len(response_rows))

    st.markdown("#### Visual study-experience profile")
    st.caption(
        "Researcher-only descriptive summary of human agency, calibration support, explainability, usability and global reliance."
    )
    _small_n(n, response_rows, "completed questionnaires")

    construct_frame = pd.DataFrame(
        [
            {
                "Construct": CONSTRUCT_LABELS.get(key, str(key).replace("_", " ").title()),
                "Mean response": _num(value),
            }
            for key, value in (summary.get("construct_means") or {}).items()
            if _num(value) is not None
        ]
    )
    _construct_lollipop(construct_frame)

    item_frame = pd.DataFrame(
        [
            {
                "Item": ITEM_LABELS.get(key, (str(key).replace("_", " ").title(), "Other"))[0],
                "Construct": ITEM_LABELS.get(key, (str(key), "Other"))[1],
                "Mean response": _num(value),
            }
            for key, value in (summary.get("item_means") or {}).items()
            if _num(value) is not None
        ]
    )
    _item_lollipop(item_frame)
    _likert_diverging(response_rows)
    _expertise_matrix(summary)


# Compatibility aliases for earlier patches / call sites.
render_secondary_questionnaire_visuals = render_secondary_visuals
render_study_experience_visuals = render_secondary_visuals
