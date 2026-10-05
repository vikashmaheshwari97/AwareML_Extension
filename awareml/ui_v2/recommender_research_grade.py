from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from awareml.recommender.v2_ranking import rank_candidates
from .state import ROOT


OBJECTIVES = ("accuracy", "runtime", "energy", "co2")
OBJECTIVE_LABELS = {
    "accuracy": "Accuracy ↑",
    "runtime": "Runtime ↓",
    "energy": "Energy ↓",
    "co2": "CO₂ ↓",
}
OBJECTIVE_COLORS = {
    "accuracy": "#2563eb",
    "runtime": "#7c3aed",
    "energy": "#059669",
    "co2": "#0f766e",
}
FRAMEWORK_COLORS = {
    "AutoStreamML": "#2563eb",
    "AutoClass": "#16a34a",
    "EvoAutoML": "#ea580c",
    "OAML": "#ca8a04",
    "ChaCha": "#9333ea",
}
PREFERENCE_PRESETS = {
    "Balanced": {"accuracy": 25, "runtime": 25, "energy": 25, "co2": 25},
    "Accuracy": {"accuracy": 70, "runtime": 10, "energy": 10, "co2": 10},
    "Runtime": {"accuracy": 10, "runtime": 70, "energy": 10, "co2": 10},
    "Energy": {"accuracy": 10, "runtime": 10, "energy": 70, "co2": 10},
    "CO₂": {"accuracy": 10, "runtime": 10, "energy": 10, "co2": 70},
    "Sustainability": {"accuracy": 20, "runtime": 10, "energy": 35, "co2": 35},
}


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    try:
        return dict(value)
    except Exception:
        return {}


def _normalize_weights(weights: Mapping[str, Any]) -> Dict[str, float]:
    clean = {key: max(0.0, float(weights.get(key, 0.0) or 0.0)) for key in OBJECTIVES}
    total = float(sum(clean.values()))
    if total <= 0:
        return {"accuracy": 0.25, "runtime": 0.25, "energy": 0.25, "co2": 0.25}
    return {key: value / total for key, value in clean.items()}


def _desirability(values: pd.Series, maximize: bool) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    if s.isna().all():
        return pd.Series([np.nan] * len(s), index=s.index)
    lo, hi = float(s.min()), float(s.max())
    if abs(hi - lo) <= 1e-12:
        return pd.Series([0.5] * len(s), index=s.index, dtype=float)
    out = (s - lo) / (hi - lo)
    if not maximize:
        out = 1.0 - out
    return out.clip(0.0, 1.0)


def _contribution_frame(ranked: pd.DataFrame, weights: Mapping[str, Any]) -> pd.DataFrame:
    if ranked is None or ranked.empty:
        return pd.DataFrame()
    w = _normalize_weights(weights)
    work = ranked.copy()
    for obj in OBJECTIVES:
        score_col = obj + "_score"
        if score_col not in work.columns:
            work[score_col] = _desirability(work[obj], maximize=(obj == "accuracy"))
    rows = []
    for _, row in work.iterrows():
        for obj in OBJECTIVES:
            score = float(row.get(obj + "_score", np.nan))
            if np.isfinite(score):
                rows.append({
                    "Framework": str(row["framework"]),
                    "Objective": OBJECTIVE_LABELS[obj],
                    "Objective key": obj,
                    "Weight": float(w[obj]),
                    "Desirability": score,
                    "Contribution": float(w[obj]) * score,
                    "Utility": float(row.get("utility", np.nan)),
                    "Rank": int(row.get("rank", 0) or 0),
                })
    return pd.DataFrame(rows)


def _weights_text(weights: Mapping[str, Any]) -> str:
    w = _normalize_weights(weights)
    return " · ".join("{} {:.0%}".format(OBJECTIVE_LABELS[k].replace(" ↑", "").replace(" ↓", ""), w[k]) for k in OBJECTIVES)


def _render_contribution_chart(ranked: pd.DataFrame, weights: Mapping[str, Any], key: str) -> None:
    contributions = _contribution_frame(ranked, weights)
    if contributions.empty:
        return
    order = (
        contributions[["Framework", "Rank"]]
        .drop_duplicates()
        .sort_values("Rank")["Framework"]
        .tolist()
    )
    fig = px.bar(
        contributions,
        x="Contribution",
        y="Framework",
        color="Objective",
        orientation="h",
        category_orders={"Framework": list(reversed(order))},
        color_discrete_map={OBJECTIVE_LABELS[k]: OBJECTIVE_COLORS[k] for k in OBJECTIVES},
        hover_data={"Weight": ":.0%", "Desirability": ":.3f", "Contribution": ":.3f", "Rank": True},
        title="How each objective contributes to the preference utility",
    )
    fig.update_layout(
        height=360,
        barmode="stack",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(l=95, r=25, t=55, b=75),
        xaxis_title="Weighted utility contribution",
        yaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.caption(
        "Each bar is decomposed into weight × normalized desirability. This makes the ranking auditable: utility is a relative preference score, not a confidence probability."
    )


def _render_uncertainty_chart(ranked: pd.DataFrame, key: str) -> None:
    if ranked is None or ranked.empty:
        return
    specs = [
        ("accuracy", "Predicted accuracy", True, 4),
        ("runtime", "Predicted runtime (s)", False, 3),
        ("energy", "Predicted energy (kWh)", False, 6),
        ("co2", "Predicted CO₂ (kg)", False, 6),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[s[1] for s in specs], horizontal_spacing=0.13, vertical_spacing=0.20)
    for idx, (metric, label, maximize, precision) in enumerate(specs):
        r, c = divmod(idx, 2)
        r += 1; c += 1
        vals = pd.to_numeric(ranked[metric], errors="coerce")
        lower = pd.to_numeric(ranked.get(metric + "_lower"), errors="coerce") if metric + "_lower" in ranked.columns else vals
        upper = pd.to_numeric(ranked.get(metric + "_upper"), errors="coerce") if metric + "_upper" in ranked.columns else vals
        plus = (upper - vals).clip(lower=0)
        minus = (vals - lower).clip(lower=0)
        colors = [FRAMEWORK_COLORS.get(str(fw), "#64748b") for fw in ranked["framework"]]
        fig.add_trace(
            go.Bar(
                x=ranked["framework"], y=vals,
                marker_color=colors,
                error_y=dict(type="data", symmetric=False, array=plus, arrayminus=minus, thickness=1.5, width=5),
                text=[("{:.%df}" % precision).format(v) if pd.notna(v) else "N/A" for v in vals],
                textposition="auto",
                hovertemplate="%{x}<br>point=%{y}<extra></extra>",
                showlegend=False,
            ), row=r, col=c,
        )
        fig.update_xaxes(tickangle=-18, row=r, col=c)
        fig.update_yaxes(title_text=("Higher is better" if maximize else "Lower is better"), row=r, col=c)
    fig.update_layout(height=660, margin=dict(l=45, r=25, t=65, b=70), title="Predicted objective evidence with empirical 90% bounds")
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.caption(
        "The intervals are predictive evidence inherited from empirical residual calibration. They describe uncertainty around predicted outcomes; they are not the probability that a framework will win."
    )


def _preference_stability(ranked: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ranked is None or ranked.empty:
        return pd.DataFrame(), pd.DataFrame()
    candidates = ranked[[c for c in ["framework", "accuracy", "runtime", "energy", "co2"] if c in ranked.columns]].copy()
    if len(candidates.columns) < 5:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for profile, weights in PREFERENCE_PRESETS.items():
        ranked_profile, _ = rank_candidates(candidates, weights=weights, mode="point")
        for _, row in ranked_profile.iterrows():
            rows.append({
                "Preference profile": profile,
                "Framework": str(row["framework"]),
                "Rank": int(row["rank"]),
                "Utility": float(row["utility"]),
            })
    long = pd.DataFrame(rows)
    winners = long[long["Rank"] == 1].groupby("Framework").size().rename("Profiles won").reset_index()
    return long, winners


def _render_stability_chart(ranked: pd.DataFrame, key: str) -> None:
    long, winners = _preference_stability(ranked)
    if long.empty:
        return
    matrix = long.pivot(index="Framework", columns="Preference profile", values="Rank")
    profile_order = list(PREFERENCE_PRESETS)
    fw_order = (
        long.groupby("Framework")["Rank"].mean().sort_values().index.tolist()
    )
    matrix = matrix.reindex(index=fw_order, columns=profile_order)
    fig = px.imshow(
        matrix,
        text_auto=".0f",
        zmin=1,
        zmax=5,
        color_continuous_scale=[(0, "#d1fae5"), (0.25, "#93c5fd"), (0.55, "#fde68a"), (1, "#fecaca")],
        labels=dict(x="Preference profile", y="Framework", color="Predicted rank"),
        title="Preference-sensitivity map · rank under six predefined profiles",
        aspect="auto",
    )
    fig.update_layout(height=350, margin=dict(l=90, r=25, t=55, b=65))
    st.plotly_chart(fig, use_container_width=True, key=key)
    if not winners.empty:
        winner_text = " · ".join("{}: {}".format(r["Framework"], int(r["Profiles won"])) for _, r in winners.sort_values("Profiles won", ascending=False).iterrows())
        st.caption("Predefined-profile wins: {}. This is a sensitivity diagnostic over the same predicted outcomes; no framework is rerun.".format(winner_text))


def render_decision_space_research_layer(ranked: pd.DataFrame, weights: Mapping[str, Any], meta: Optional[Mapping[str, Any]] = None) -> None:
    if ranked is None or ranked.empty:
        return
    st.markdown("## Research-grade decision diagnostics")
    st.caption(
        "The 3D scene is the trade-off view; the panels below make the exact utility calculation, uncertainty, and preference sensitivity inspectable."
    )
    t1, t2, t3 = st.tabs(["Utility decomposition", "Prediction uncertainty", "Preference sensitivity"])
    with t1:
        _render_contribution_chart(ranked, weights, "rg_decision_contrib")
        sorted_ranked = ranked.sort_values("rank") if "rank" in ranked.columns else ranked.sort_values("utility", ascending=False)
        if len(sorted_ranked) >= 2:
            top, runner = sorted_ranked.iloc[0], sorted_ranked.iloc[1]
            margin = float(top["utility"]) - float(runner["utility"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted #1", str(top["framework"]))
            c2.metric("Runner-up", str(runner["framework"]))
            c3.metric("Utility margin", "{:.4f}".format(margin))
            st.info(
                "The recommendation is based on relative utility under {}. A small #1–#2 margin indicates that the decision is preference-sensitive even if the rank is currently unambiguous.".format(_weights_text(weights))
            )
    with t2:
        _render_uncertainty_chart(ranked, "rg_decision_uncertainty")
    with t3:
        _render_stability_chart(ranked, "rg_decision_stability")


def _append_dataset_review(state: Mapping[str, Any], ranked: pd.DataFrame, weights: Mapping[str, Any], decision: str, note: str) -> Dict[str, Any]:
    path = Path(ROOT) / "artifacts" / "copilot" / "dataset_aware_recommender_reviews.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    top = ranked.sort_values("rank").iloc[0] if "rank" in ranked.columns else ranked.sort_values("utility", ascending=False).iloc[0]
    record = {
        "record_type": "dataset_aware_recommender_human_review",
        "review_id": "DAR-{}".format(uuid.uuid4().hex[:10].upper()),
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_name": state.get("dataset_name"),
        "target": state.get("target"),
        "recommended_framework": str(top["framework"]),
        "ranking_utility": float(top["utility"]),
        "weights": _normalize_weights(weights),
        "decision": decision,
        "note": str(note or "").strip() or None,
        "recommender_retrained": False,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    return record


def render_dataset_aware_research_layer(state: Dict[str, Any], ranked: pd.DataFrame, weights: Mapping[str, Any], meta: Optional[Mapping[str, Any]] = None) -> None:
    if ranked is None or ranked.empty:
        return
    st.markdown("### Recommendation diagnostics and human review")
    tabs = st.tabs(["Why this ranking?", "Uncertainty", "Sensitivity", "Human decision"])
    with tabs[0]:
        _render_contribution_chart(ranked, weights, "rg_dataset_contrib")
        order = ranked.sort_values("rank") if "rank" in ranked.columns else ranked.sort_values("utility", ascending=False)
        top = order.iloc[0]
        runner = order.iloc[1] if len(order) > 1 else None
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted #1", str(top["framework"]))
        c2.metric("Utility", "{:.4f}".format(float(top["utility"])))
        c3.metric("Runner-up", str(runner["framework"]) if runner is not None else "N/A")
        c4.metric("#1–#2 margin", "{:.4f}".format(float(top["utility"] - runner["utility"])) if runner is not None else "N/A")
    with tabs[1]:
        _render_uncertainty_chart(ranked, "rg_dataset_uncertainty")
    with tabs[2]:
        _render_stability_chart(ranked, "rg_dataset_stability")
    with tabs[3]:
        st.markdown("**Human review of the dataset-aware recommendation**")
        st.caption(
            "The frozen recommender remains unchanged. This review records whether you would use the recommendation, keep it as an alternative, or reject it; it does not retrain the model or alter the held-out evidence."
        )
        decision = st.segmented_control(
            "How do you want to treat this recommendation?",
            ["Use as candidate plan", "Keep as alternative", "Reject recommendation"],
            default="Use as candidate plan",
            key="rg_dataset_human_decision",
        ) or "Use as candidate plan"
        note = st.text_input(
            "Decision note (optional)",
            key="rg_dataset_human_note",
            placeholder="Example: Keep the recommendation, but validate latency and energy after execution.",
        )
        if st.button("Save dataset-aware review", key="rg_dataset_save_review", use_container_width=True):
            rec = _append_dataset_review(state, ranked, weights, decision, note)
            state["dataset_aware_human_review"] = rec
            st.success("Human review saved as {}.".format(rec["review_id"]))
        saved = state.get("dataset_aware_human_review")
        if isinstance(saved, dict):
            st.caption("Saved review: {} · {}".format(saved.get("review_id"), saved.get("decision")))


def render_historical_prior_research_layer(ranking: pd.DataFrame, weights: Mapping[str, Any]) -> None:
    if ranking is None or ranking.empty:
        return
    st.markdown("### Cross-dataset evidence diagnostics")
    st.caption(
        "This prior is a transparent aggregation over development datasets. It does not use the current dataset; its value is as a global baseline and comparison point for the learned dataset-aware recommender."
    )
    left, right = st.columns([1.1, 0.9])
    with left:
        work = ranking.copy()
        if {"historical_utility", "win_rate", "framework"}.issubset(work.columns):
            if "utility_iqr" not in work.columns:
                work["utility_iqr"] = np.nan
            if "top3_rate" not in work.columns:
                work["top3_rate"] = np.nan
            fig = px.scatter(
                work,
                x="historical_utility",
                y="win_rate",
                text="framework",
                size="top3_rate" if work["top3_rate"].notna().any() else None,
                color="framework",
                color_discrete_map=FRAMEWORK_COLORS,
                labels={
                    "historical_utility": "Cross-dataset preference score",
                    "win_rate": "Cross-dataset win rate",
                    "top3_rate": "Top-3 rate",
                },
                title="Global prior strength and cross-dataset consistency",
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=390, showlegend=False, margin=dict(l=45, r=25, t=55, b=55))
            st.plotly_chart(fig, use_container_width=True, key="rg_historical_scatter")
    with right:
        cols = [c for c in ["framework", "historical_utility", "utility_iqr", "win_rate", "top3_rate"] if c in ranking.columns]
        robust = ranking[cols].copy().rename(columns={
            "framework": "Framework",
            "historical_utility": "Preference score",
            "utility_iqr": "Score IQR",
            "win_rate": "Win rate",
            "top3_rate": "Top-3 rate",
        })
        st.markdown("**Robustness summary**")
        st.dataframe(robust, use_container_width=True, hide_index=True)
        st.info(
            "Why keep this prior? It provides an interpretable dataset-free baseline. Agreement with the dataset-aware model is corroborating evidence; disagreement is a useful signal that this dataset may differ from the global development pattern."
        )


def render_goal_recommendation_research_layer(ranked: pd.DataFrame, weights: Mapping[str, Any], framework: str) -> None:
    if ranked is None or ranked.empty:
        return
    st.markdown("#### Decision trace · why the recommendation is #1")
    t1, t2 = st.tabs(["Objective contributions", "Top alternatives"])
    with t1:
        _render_contribution_chart(ranked, weights, "rg_goal_contrib")
    with t2:
        order = ranked.sort_values("rank") if "rank" in ranked.columns else ranked.sort_values("utility", ascending=False)
        show = [c for c in ["rank", "framework", "utility", "accuracy", "runtime", "energy", "co2", "near_pareto"] if c in order.columns]
        table = order[show].copy().rename(columns={
            "rank": "Rank", "framework": "Framework", "utility": "Utility",
            "accuracy": "Accuracy", "runtime": "Runtime (s)", "energy": "Energy (kWh)",
            "co2": "CO₂ (kg)", "near_pareto": "ε-Pareto",
        })
        st.dataframe(table, use_container_width=True, hide_index=True)
        if len(order) >= 2:
            top, runner = order.iloc[0], order.iloc[1]
            st.info(
                "{} is currently #1 with utility {:.4f}; {} is #2 with {:.4f}. The margin is {:.4f}. The ranking can change if the human changes the objective weights because the predicted outcome profiles remain fixed while utility is recomputed.".format(
                    top["framework"], float(top["utility"]), runner["framework"], float(runner["utility"]), float(top["utility"] - runner["utility"])
                )
            )


def render_priority_review_preview(selected: Sequence[str], corrected: Optional[Sequence[str]]) -> None:
    if corrected is None:
        return
    before = list(selected or [])
    after = list(corrected or [])
    names = ["Accuracy", "Runtime", "Energy", "CO2"]
    rows = []
    for phase, values in [("Current interpretation", before), ("Proposed correction", after)]:
        share = 1.0 / len(values) if values else 0.0
        for name in names:
            rows.append({
                "State": phase,
                "Objective": "CO₂" if name == "CO2" else name,
                "Weight": share if name in values else 0.0,
            })
    frame = pd.DataFrame(rows)
    fig = px.bar(
        frame,
        x="Objective", y="Weight", color="State", barmode="group",
        range_y=[0, 1],
        color_discrete_map={"Current interpretation": "#94a3b8", "Proposed correction": "#2563eb"},
        title="Preview of the objective-weight change",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=300, legend=dict(orientation="h", y=-0.2), margin=dict(l=35, r=20, t=50, b=70))
    st.plotly_chart(fig, use_container_width=True, key="rg_priority_preview")
    st.caption(
        "Saving the correction reranks the same dataset meta-profile. It does not retrain the frozen recommender and does not execute a framework."
    )


AUTOSTREAMML_LEGACY_SEARCH_SPACE = {
    "Perceptron": {"l2": "0.000–0.010, step 0.001"},
    "LogisticRegression": {"l2": "0.000–0.010, step 0.001"},
    "GaussianNB": {},
    "HoeffdingTreeClassifier": {
        "max_depth": "10–100, step 10", "grace_period": "50–500, step 50",
        "split_confidence": "1e-9, 1e-7, 1e-4, 1e-2", "tie_threshold": "0.02–0.08, step 0.01",
        "nb_threshold": "0–50, step 10", "split_criterion": "info_gain | gini | hellinger",
        "leaf_prediction": "mc | nb | nba",
    },
    "AdaptiveRandomForestClassifier": {
        "n_models": "3–9", "max_depth": "10–100, step 10", "grace_period": "50–500, step 50",
        "lambda_value": "2–10", "split_confidence": "0.01–0.10, step 0.01",
        "tie_threshold": "0.02–0.08, step 0.01", "nb_threshold": "0–50, step 10",
        "split_criterion": "info_gain | gini | hellinger", "leaf_prediction": "mc | nb | nba",
    },
    "KNNClassifier": {
        "n_neighbors": "3–9", "window_size": "100–5100, step 200", "weighted": "True | False", "p": "1–5",
    },
    "StandardScaler": {"with_std": "True | False"},
    "PoissonInclusion": {"p": "0.1–1.0, step 0.1"},
    "VarianceThreshold": {"threshold": "0.0–1.0, step 0.1", "min_samples": "1–10"},
    "SelectKBest": {"k": "1–25", "similarity": "PearsonCorr | Cov"},
}


def render_framework_parameter_reference(framework: str) -> None:
    st.caption(
        "Framework-parameter edits are configuration metadata unless the selected framework adapter explicitly exposes those parameters to the execution runner. They must not be assumed to apply across frameworks."
    )
    if str(framework) != "AutoStreamML":
        st.info(
            "The currently proposed framework is {}. The legacy AutoStreamML search space is therefore not applied to this plan. A framework-specific parameter registry is required before parameters can be safely staged for execution.".format(framework or "N/A")
        )
    if st.checkbox("Show legacy AutoStreamML search-space reference", key="rg_show_autostream_legacy"):
        rows = []
        for component, params in AUTOSTREAMML_LEGACY_SEARCH_SPACE.items():
            if not params:
                rows.append({"Component": component, "Parameter": "—", "Search space": "No explicit parameters"})
            for parameter, values in params.items():
                rows.append({"Component": component, "Parameter": parameter, "Search space": values})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Reference source: the previous AutoML_Stream/AutoStreamML.py search-space definition. This table is informational until the current AwareML adapter validates and wires each parameter."
        )


def _config_diff_frame(saved: Mapping[str, Any]) -> pd.DataFrame:
    diff = saved.get("config_diff") or []
    rows = []
    for item in diff:
        data = _as_dict(item)
        rows.append({"Setting": data.get("path"), "Before": data.get("before"), "After": data.get("after")})
    return pd.DataFrame(rows)


def render_final_review_research_layer(state: Dict[str, Any], saved: Mapping[str, Any]) -> None:
    saved = _as_dict(saved)
    if str(saved.get("decision")) not in {"approved", "approved_with_edits"}:
        return
    st.markdown("### Approved plan handoff")
    diff = _config_diff_frame(saved)
    if not diff.empty:
        st.markdown("**Recorded configuration changes**")
        st.dataframe(diff, use_container_width=True, hide_index=True)
    final_config = _as_dict(saved.get("final_config"))
    st.caption(
        "Approval is intentionally separate from execution. You can stage the supported settings for Run Studio, inspect the handoff there, and still decide when to start the benchmark."
    )
    if st.button("Stage approved settings for Run Studio", key="rg_stage_approved_plan", use_container_width=True):
        state["copilot_run_studio_handoff"] = {
            "handoff_id": "RUN-{}".format(uuid.uuid4().hex[:10].upper()),
            "staged_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "proposal_id": saved.get("proposal_id"),
            "decision": saved.get("decision"),
            "final_config": final_config,
            "applied_to_widgets": False,
        }
        st.success("Approved plan staged. Open Run Studio to review and apply supported settings; no benchmark has been started.")


def _handoff_mapping(final_config: Mapping[str, Any]) -> Tuple[Dict[str, Any], list]:
    cfg = _as_dict(final_config)
    supported: Dict[str, Any] = {}
    retained = []

    framework = cfg.get("framework")
    if framework in {"AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha"}:
        supported["run_frameworks"] = [str(framework)]

    window = cfg.get("window_size")
    try:
        window_i = int(window)
        if 100 <= window_i <= 5000:
            supported["run_window"] = window_i
        elif window is not None:
            retained.append("Window size {} is outside the current Run Studio range 100–5000.".format(window))
    except Exception:
        if window is not None:
            retained.append("Window size could not be mapped to Run Studio.")

    budget = cfg.get("time_budget_sec")
    try:
        budget_i = int(round(float(budget)))
        if 5 <= budget_i <= 600:
            supported["run_budget"] = budget_i
        elif budget is not None:
            retained.append("Time budget {} s is outside the current Run Studio range 5–600 s.".format(budget))
    except Exception:
        if budget is not None:
            retained.append("Time budget could not be mapped to Run Studio.")

    explain = _as_dict(cfg.get("explainability"))
    method = str(explain.get("method") or "").lower()
    if method in {"auto", "shap", "lime", "permutation"}:
        supported["run_xai_method"] = method
    if explain.get("level") not in {None, ""}:
        retained.append("Explainability level '{}' is retained in the audit record; Run Studio currently exposes the method, not a separate level control.".format(explain.get("level")))

    sustain = _as_dict(cfg.get("sustainability"))
    if sustain:
        supported["run_codecarbon"] = bool(sustain.get("track_energy") or sustain.get("track_co2"))
        if bool(sustain.get("track_energy")) != bool(sustain.get("track_co2")):
            retained.append("Run Studio currently uses one combined CodeCarbon toggle, so separate Energy/CO₂ tracking choices cannot be mapped independently.")

    drift = _as_dict(cfg.get("drift"))
    if drift:
        retained.append("Drift detector/sensitivity remains recorded in the approved plan; the current Run Studio does not expose an equivalent sensitivity widget.")
    fairness = _as_dict(cfg.get("fairness"))
    if fairness.get("requested"):
        retained.append("Fairness audit request remains recorded. Select/confirm the sensitive attribute in Run Studio; fairness evidence is evaluated after execution.")
    if _as_dict(cfg.get("framework_parameters")):
        retained.append("Framework hyperparameters remain recorded but are not auto-applied because the current runner has no framework-specific validated parameter handoff schema.")
    return supported, retained


def render_run_studio_handoff(state: Dict[str, Any]) -> None:
    handoff = state.get("copilot_run_studio_handoff")
    if not isinstance(handoff, dict):
        return
    cfg = _as_dict(handoff.get("final_config"))
    supported, retained = _handoff_mapping(cfg)
    st.markdown("## Approved Copilot plan available")
    st.info(
        "A human-approved Copilot plan has been staged for this workspace. Nothing runs automatically. Review which settings can be mapped to the current Run Studio before applying them."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Handoff", str(handoff.get("handoff_id") or "N/A"))
    c2.metric("Approved framework", str(cfg.get("framework") or "N/A"))
    c3.metric("Mapped Run Studio fields", len(supported))

    rows = []
    labels = {
        "run_frameworks": "Framework selection",
        "run_window": "Window size",
        "run_budget": "Per-framework time budget",
        "run_xai_method": "Model-level XAI method",
        "run_codecarbon": "CodeCarbon energy/CO₂ measurement",
    }
    for key, value in supported.items():
        rows.append({"Run Studio setting": labels.get(key, key), "Staged value": value, "Status": "Can apply"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if retained:
        with st.expander("Approved settings retained for audit/manual configuration", expanded=False):
            for item in retained:
                st.markdown("- {}".format(item))

    if not handoff.get("applied_to_widgets"):
        if st.button("Apply supported settings to Run Studio", key="rg_apply_handoff", type="primary", use_container_width=True):
            for key, value in supported.items():
                st.session_state[key] = value
            handoff["applied_to_widgets"] = True
            handoff["applied_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            state["copilot_run_studio_handoff"] = handoff
            st.success("Supported settings applied to the Run Studio controls. Review them below, then start the benchmark only when you choose to.")
            st.rerun()
    else:
        st.success("Supported settings have been applied to the Run Studio controls. The benchmark is still waiting for your explicit Run benchmark action.")
        if st.button("Clear staged Copilot handoff", key="rg_clear_handoff"):
            state.pop("copilot_run_studio_handoff", None)
            st.rerun()
