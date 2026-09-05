from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from awareml.recommender.historical_preference import (
    HistoricalPreferenceRecommender,
    normalize_preference_weights,
)
from awareml.recommender.meta_logs_v2_audit import (
    audit_meta_logs_v2,
    historical_winner_sensitivity,
    load_v2_manifest_quality,
)
from awareml.recommender.v2_ranking import normalize_weights

from .data import load_v2_recommender
from .pre14_usability import copilot_weights_from_state
from .state import ROOT, dataset_signature, ensure_research_state


HIST_PRESETS = {
    "Accuracy first": {"accuracy": 55, "runtime": 15, "energy": 15, "co2": 15},
    "Balanced": {"accuracy": 25, "runtime": 25, "energy": 25, "co2": 25},
    "Fast response": {"accuracy": 25, "runtime": 50, "energy": 15, "co2": 10},
    "Low energy": {"accuracy": 25, "runtime": 15, "energy": 50, "co2": 10},
    "Low CO2": {"accuracy": 25, "runtime": 15, "energy": 10, "co2": 50},
}

V2_PRESETS = {
    "Balanced": {"accuracy": 25, "runtime": 25, "energy": 25, "co2": 25},
    "Accuracy focused": {"accuracy": 70, "runtime": 10, "energy": 10, "co2": 10},
    "Runtime focused": {"accuracy": 10, "runtime": 70, "energy": 10, "co2": 10},
    "Energy focused": {"accuracy": 10, "runtime": 10, "energy": 70, "co2": 10},
    "CO2 focused": {"accuracy": 10, "runtime": 10, "energy": 10, "co2": 70},
    "Sustainability": {"accuracy": 20, "runtime": 10, "energy": 35, "co2": 35},
}


def _weights_text(weights: Mapping[str, float]) -> str:
    return " · ".join([
        "Accuracy {:.0%}".format(float(weights["accuracy"])),
        "Runtime {:.0%}".format(float(weights["runtime"])),
        "Energy {:.0%}".format(float(weights["energy"])),
        "CO2 {:.0%}".format(float(weights["co2"])),
    ])


def _set_hist_weights(values: Mapping[str, float]) -> None:
    values = normalize_preference_weights(values)
    for key in ("accuracy", "runtime", "energy", "co2"):
        st.session_state["three_hist_{}".format(key)] = int(round(100 * values[key]))


def _set_v2_weights(values: Mapping[str, float]) -> None:
    values = normalize_weights(values)
    for key in ("accuracy", "runtime", "energy", "co2"):
        st.session_state["three_v2_{}".format(key)] = int(round(100 * values[key]))


@st.cache_data(show_spinner=False)
def _audit():
    return audit_meta_logs_v2(ROOT)


@st.cache_data(show_spinner=False)
def _sensitivity():
    return historical_winner_sensitivity()


@st.cache_data(show_spinner=False)
def _model_quality():
    return load_v2_manifest_quality(ROOT)


def render_historical_preference_prior_tab() -> None:
    state = ensure_research_state()
    st.markdown("# Historical Preference Prior")
    st.markdown(
        "Use this when **no dataset is available yet**. It summarizes what generally worked across the frozen "
        "47-dataset / 705-run development evidence."
    )
    st.info(
        "**Historical aggregation · not machine learning.** "
        "This tab gives a global starting point, not a dataset-specific prediction."
    )

    cards = st.columns(4)
    cards[0].metric("Historical runs", "705")
    cards[1].metric("Development datasets", "47")
    cards[2].metric("Frameworks", "5")
    cards[3].metric("Seeds per dataset/framework", "3")

    st.markdown("## 1 · Choose what matters")
    latest = copilot_weights_from_state(state)
    if latest is not None and st.button("Use latest Goal Copilot priorities", key="three_hist_goal"):
        _set_hist_weights(latest)
        st.rerun()

    cols = st.columns(len(HIST_PRESETS))
    for col, (label, values) in zip(cols, HIST_PRESETS.items()):
        with col:
            if st.button(label, key="three_hist_preset_{}".format(label), use_container_width=True):
                _set_hist_weights(values)
                st.rerun()

    d = HIST_PRESETS["Accuracy first"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        a = st.slider("Accuracy ↑", 0, 100, int(st.session_state.get("three_hist_accuracy", d["accuracy"])), key="three_hist_accuracy")
    with c2:
        r = st.slider("Runtime ↓", 0, 100, int(st.session_state.get("three_hist_runtime", d["runtime"])), key="three_hist_runtime")
    with c3:
        e = st.slider("Low energy ↓", 0, 100, int(st.session_state.get("three_hist_energy", d["energy"])), key="three_hist_energy")
    with c4:
        co = st.slider("Low CO2 ↓", 0, 100, int(st.session_state.get("three_hist_co2", d["co2"])), key="three_hist_co2")

    weights = normalize_preference_weights({"accuracy": a, "runtime": r, "energy": e, "co2": co})
    st.info("Normalized priorities: {}".format(_weights_text(weights)))

    seed_mode_label = st.segmented_control(
        "Seed evidence",
        ["Stable 3-seed aggregate", "Best observed single seed · exploratory"],
        default="Stable 3-seed aggregate",
        key="three_hist_seed",
    ) or "Stable 3-seed aggregate"
    seed_mode = (
        HistoricalPreferenceRecommender.STABLE
        if seed_mode_label.startswith("Stable")
        else HistoricalPreferenceRecommender.BEST_SEED
    )
    if seed_mode == HistoricalPreferenceRecommender.STABLE:
        st.success("Primary research view: each dataset/framework is represented by the mean of seeds 42/43/44.")
    else:
        st.warning("Exploratory upper-bound view. Use the stable 3-seed aggregate for primary reporting.")

    signature = json.dumps({"weights": weights, "seed_mode": seed_mode}, sort_keys=True)
    if state.get("three_hist_signature") not in {None, signature}:
        state["three_hist_result"] = None
        state["three_hist_ranking"] = None

    st.markdown("## 2 · Compare the historical evidence")
    if st.button("Compare frameworks across 705 historical runs", key="three_hist_run", use_container_width=True):
        result = HistoricalPreferenceRecommender().recommend(weights=weights, seed_mode=seed_mode)
        state["three_hist_result"] = result.as_dict()
        state["three_hist_ranking"] = result.ranking.copy()
        state["three_hist_signature"] = signature
        # Preserve compatibility with the previous historical tab.
        state["historical_meta_result"] = result.as_dict()
        state["historical_meta_ranking"] = result.ranking.copy()

    result = state.get("three_hist_result")
    ranking = state.get("three_hist_ranking")
    if isinstance(result, dict) and isinstance(ranking, pd.DataFrame) and not ranking.empty and state.get("three_hist_signature") == signature:
        top = ranking.iloc[0]
        winner = str(top["framework"])
        st.markdown("## Historical Framework Starting Point")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Historical starting point", winner)
        c2.metric("Validated default algorithm", str(result.get("algorithm") or "N/A"))
        c3.metric("Historical preference score", "{:.3f}".format(float(top["historical_utility"])))
        c4.metric("Cross-dataset wins", "{} / {}".format(int(top["win_count"]), int(top["support_datasets"])))
        st.caption(
            "The score is a normalized historical ranking score, not a probability or confidence score."
        )
        st.write(
            "**Why this starting point?** Across the 47 development datasets, {} has the highest aggregate score "
            "under {}. Use it as a global prior only.".format(winner, _weights_text(weights))
        )

        show_cols = [
            "rank", "framework", "historical_utility", "win_rate", "top3_rate",
            "accuracy_median", "runtime_median_sec", "energy_median_kwh", "co2_median_kg",
        ]
        st.dataframe(ranking[[c for c in show_cols if c in ranking.columns]], use_container_width=True, hide_index=True)

        chart = ranking.sort_values("historical_utility", ascending=True)
        fig = px.bar(
            chart, x="historical_utility", y="framework", orientation="h",
            text="historical_utility", title="Historical preference ranking",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="inside")
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="three_hist_chart")

    st.markdown("### Why can the same framework keep winning?")
    sensitivity = _sensitivity()
    st.dataframe(sensitivity, use_container_width=True, hide_index=True)
    winners = sorted(set(sensitivity["Winner"].astype(str))) if not sensitivity.empty else []
    if len(winners) == 1:
        st.info(
            "{} wins all predeclared sensitivity profiles shown here. The sliders are still changing the scores and "
            "cross-dataset win counts; the global winner simply remains the same.".format(winners[0])
        )
    else:
        st.success(
            "The historical prior is preference-sensitive: the sensitivity profiles produce {} different winners.".format(len(winners))
        )

    with st.expander("Research details · is meta_logs_v2 structurally sound?", expanded=False):
        audit = _audit()
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Rows", audit["rows"])
        a2.metric("Duplicate run keys", audit["duplicate_dataset_framework_seed_rows"])
        a3.metric("Primary metric nulls", sum(audit["primary_metric_nulls"].values()))
        a4.metric("Core integrity", "PASS" if audit["core_integrity_pass"] else "CHECK")
        if audit.get("runtime_near_budget_fraction_95") is not None:
            st.metric("Runs within 95% of budget ceiling", "{:.1%}".format(audit["runtime_near_budget_fraction_95"]))
        if audit.get("energy_co2_spearman") is not None:
            st.metric("Energy–CO2 Spearman rho", "{:.3f}".format(audit["energy_co2_spearman"]))
        for warning in audit.get("warnings") or []:
            st.warning(warning)

    st.success(
        "When a dataset is available, move to **Dataset-aware ML Recommender V2**. "
        "That path uses learned models and can legitimately choose a different framework."
    )


def _target_guard(df: pd.DataFrame, target: str) -> Optional[str]:
    if target not in df.columns:
        return "Selected target is not in the current dataset."
    series = df[target]
    if pd.api.types.is_numeric_dtype(series) and int(series.nunique(dropna=True)) >= 20:
        return (
            "High-cardinality numeric target detected ({} unique values). V2 was developed for streaming classification; "
            "confirm that this is truly a classification target."
        ).format(int(series.nunique(dropna=True)))
    return None


def _render_model_quality() -> None:
    quality = _model_quality()
    rows = []
    for objective in ("accuracy", "runtime", "energy", "co2"):
        q = quality.get(objective) or {}
        rows.append({
            "Objective": objective.title(),
            "Frozen learner": q.get("model_name"),
            "LODO Top-1": q.get("top1_accuracy"),
            "LODO Top-3": q.get("top3_accuracy"),
            "Normalized regret": q.get("normalized_regret"),
            "Spearman": q.get("spearman"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Development/LODO diagnostics only; not the final 23-dataset held-out result.")
    rq = quality.get("runtime") or {}
    try:
        if float(rq.get("top1_accuracy")) < 0.50:
            st.warning(
                "Runtime is the weakest frozen objective model by development Top-1 ({:.3f}). "
                "Treat its predictions more cautiously.".format(float(rq["top1_accuracy"]))
            )
    except Exception:
        pass


def render_dataset_aware_v2_tab() -> None:
    state = ensure_research_state()
    st.markdown("# Dataset-aware ML Recommender V2")
    st.markdown(
        "Use this when a **dataset and target are available**. This is the actual learned meta-recommender."
    )
    st.success(
        "**Learned models · dataset-specific pre-run prediction.** "
        "The LLM does not choose the framework in this tab."
    )

    cards = st.columns(4)
    cards[0].metric("Development datasets", "47")
    cards[1].metric("Training profiles", "235")
    cards[2].metric("Underlying runs", "705")
    cards[3].metric("Reserved final-test datasets", "23")

    df = state.get("dataset")
    target = state.get("target")
    if df is None or not target or target not in getattr(df, "columns", []):
        st.warning(
            "Load a dataset and choose its target in **Run Studio**. "
            "No framework execution is required before getting this prediction."
        )
        st.markdown("### Frozen learned models")
        _render_model_quality()
        st.info(
            "The 23 reserved datasets remain untouched for the final external evaluation. "
            "Do not use them for tuning before the Phase-14 protocol is frozen."
        )
        return

    warning = _target_guard(df, str(target))
    if warning:
        st.warning(warning)

    st.markdown("## 1 · Dataset context")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Dataset", str(state.get("dataset_name") or "Loaded dataset"))
    d2.metric("Rows", "{:,}".format(len(df)))
    d3.metric("Predictor columns", max(0, len(df.columns) - 1))
    d4.metric("Target", str(target))
    st.caption(
        "V2 uses a compact dataset meta-profile. The frozen development protocol used window size 1000 and time budget 60 s."
    )

    st.markdown("## 2 · Choose your preferences")
    goal_weights = copilot_weights_from_state(state)
    if goal_weights is not None and st.button("Use latest Goal Copilot priorities", key="three_v2_goal"):
        _set_v2_weights(goal_weights)
        st.rerun()

    cols = st.columns(len(V2_PRESETS))
    for col, (label, values) in zip(cols, V2_PRESETS.items()):
        with col:
            if st.button(label, key="three_v2_preset_{}".format(label), use_container_width=True):
                _set_v2_weights(values)
                st.rerun()

    d = V2_PRESETS["Balanced"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        a = st.slider("Accuracy ↑", 0, 100, int(st.session_state.get("three_v2_accuracy", d["accuracy"])), key="three_v2_accuracy")
    with c2:
        r = st.slider("Runtime ↓", 0, 100, int(st.session_state.get("three_v2_runtime", d["runtime"])), key="three_v2_runtime")
    with c3:
        e = st.slider("Energy ↓", 0, 100, int(st.session_state.get("three_v2_energy", d["energy"])), key="three_v2_energy")
    with c4:
        co = st.slider("CO2 ↓", 0, 100, int(st.session_state.get("three_v2_co2", d["co2"])), key="three_v2_co2")

    weights = normalize_weights({"accuracy": a, "runtime": r, "energy": e, "co2": co})
    st.info("Normalized preferences: {}".format(_weights_text(weights)))

    mode_label = st.segmented_control(
        "Prediction evidence",
        ["Point prediction", "Conservative 90% bounds"],
        default="Point prediction",
        key="three_v2_mode",
    ) or "Point prediction"
    mode = "point" if mode_label == "Point prediction" else "conservative"

    sig = "{}|{}|{}|{}".format(
        dataset_signature(state), json.dumps(weights, sort_keys=True), mode, str(target)
    )
    if state.get("three_v2_signature") not in {None, sig}:
        state["three_v2_ranking"] = None
        state["three_v2_meta"] = None

    st.markdown("## 3 · Predict the five-framework ranking")
    if st.button("Run frozen ML Recommender V2", key="three_v2_run", use_container_width=True):
        ranked, meta = load_v2_recommender().recommend_dataframe(
            df,
            target=str(target),
            weights=weights,
            window_size=1000,
            time_budget_sec=60.0,
            dataset_family="unknown",
            source_type="ui_active_dataset",
            drift_type="unknown",
            ranking_mode=mode,
            coverage=0.90,
        )
        state["three_v2_ranking"] = ranked.copy()
        state["three_v2_meta"] = dict(meta)
        state["three_v2_signature"] = sig
        state["preference_weights"] = dict(weights)

    ranked = state.get("three_v2_ranking")
    meta = state.get("three_v2_meta")
    if not isinstance(ranked, pd.DataFrame) or ranked.empty or state.get("three_v2_signature") != sig:
        st.markdown("### Frozen learned models")
        _render_model_quality()
        return

    top = ranked.iloc[0]
    winner = str(top["framework"])
    st.markdown("## Dataset-specific pre-run recommendation")
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Predicted framework", winner)
    x2.metric("Predicted rank", "#1 of {}".format(len(ranked)))
    x3.metric("Normalized preference utility", "{:.4f}".format(float(top["utility"])))
    x4.metric("ε-Pareto candidate", "Yes" if bool(top.get("near_pareto")) else "No")
    st.caption(
        "Utility is a relative ranking score, not a probability, correctness score or model confidence."
    )

    st.info(
        "V2 predicts **{}** for this dataset under **{}**. "
        "These are learned pre-run predictions, not observed benchmark outcomes.".format(winner, _weights_text(weights))
    )
    for warning_text in (meta or {}).get("warnings") or []:
        st.warning(warning_text)

    cols = [
        "rank", "framework", "utility", "near_pareto",
        "accuracy", "accuracy_lower", "accuracy_upper",
        "runtime", "runtime_lower", "runtime_upper",
        "energy", "energy_lower", "energy_upper",
        "co2", "co2_lower", "co2_upper",
    ]
    table = ranked[[c for c in cols if c in ranked.columns]].copy()
    st.dataframe(table, use_container_width=True, hide_index=True)

    hist = state.get("historical_meta_result") or state.get("three_hist_result") or {}
    compare = st.columns(3)
    compare[0].metric("Goal Copilot", "Priorities ready" if goal_weights is not None else "Not generated")
    compare[1].metric("Historical prior", str(hist.get("winner") or "Not generated"))
    compare[2].metric("Dataset-aware V2", winner)
    st.caption(
        "The historical prior and dataset-aware V2 may disagree. That is expected: the prior is global, while V2 conditions on this dataset."
    )

    st.markdown("### Frozen learned-model evidence")
    _render_model_quality()

    st.info(
        "**Phase-14 boundary:** freeze the model bundle, ranking rule, run protocol, preference profiles and metrics "
        "before opening the reserved 23-dataset outcomes."
    )
