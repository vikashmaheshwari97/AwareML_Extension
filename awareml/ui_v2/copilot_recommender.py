"""The current historical and dataset-aware evidence panels in Copilot."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from .recommender_bridge import (
    OBJECTIVES, bundle_status, evaluation_evidence, historical_prior,
    load_current_recommender, normalize_weights,
)

LABELS = {"accuracy": "Accuracy ↑", "runtime": "Runtime ↓", "energy": "Energy ↓", "co2": "CO₂ ↓"}


def render_pending(status):
    if status["state"] == "invalid_bundle":
        st.error(status["message"])
    else:
        st.info("Model training pending. Goal interpretation is available now. Framework rankings will become available after the revised development results have been validated and the prefix-only models trained.")
    st.caption("No historical or learned ranking is substituted from the previous experiment.")


def render_training_cards(status):
    columns = st.columns(4)
    if not status["ready"]:
        values = [("Model training", "Pending"), ("Observed prefix cap", "1,000"),
                  ("Development validation", "Pending"), ("External evaluation", "Pending")]
    else:
        values = [("Development datasets", status["development_datasets"]),
                  ("Dataset–framework training rows", status["training_rows"]),
                  ("Underlying development runs", status["underlying_runs"]),
                  ("Seeds per dataset/framework", len(status["seeds"]))]
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)


def render_protocol(status):
    if not status["ready"]:
        return
    manifest = status["manifest"]
    cfg = manifest["protocol"]
    cap = cfg.get("max_samples")
    st.caption("Portfolio: {} frameworks · Seeds: {} · Observed prefix: up to {:,} labeled observations.".format(
        status["framework_count"], ", ".join(map(str, status["seeds"])), int(manifest["profile_size"])))
    st.caption("Prediction scope: {} · No scientific wall-clock cutoff · Safety watchdog: {:g} hours.".format(
        "full available streams" if cap is None else "up to {:,} observations, or stream end if shorter".format(int(cap)),
        float(cfg["safety_timeout_sec"]) / 3600))
    if "ChaCha" in manifest["frameworks"]:
        st.info("This extended portfolio includes ChaCha-OVR, an AwareML classification adaptation evaluated separately from the primary classification portfolio.")
    st.caption("Energy and CO₂ predictions reflect the training hardware and its recorded accounting scope. Under a fixed carbon factor, these objectives represent the same underlying signal.")


def render_validation(status):
    if not status["ready"]:
        return
    with st.expander("Development validation and external evaluation", expanded=False):
        for kind, title in (("development", "Nested LODO on development datasets"),
                            ("external", "External classification evaluation")):
            st.markdown("**{}**".format(title))
            try:
                evidence = evaluation_evidence(kind, status["directory"])
            except Exception as exc:
                st.warning(str(exc))
                continue
            if evidence is None:
                st.caption("Evaluation report pending. No performance result is claimed here.")
                continue
            meta = evidence["metadata"]
            st.caption("{} datasets · {} preference cases per method · Statistical unit: dataset.".format(
                meta["n_datasets"], meta["n_preference_cases_per_method"]))
            st.dataframe(pd.read_csv(Path(evidence["directory"]) / "baseline_summary.csv"),
                         use_container_width=True, hide_index=True)
            if kind == "external":
                st.caption("External outcomes are evaluation evidence and are excluded from training and model selection. Reused evaluation datasets are not a newly untouched test set.")


def _weight_controls(prefix):
    values = {}
    for column, key in zip(st.columns(4), OBJECTIVES):
        values[key] = column.slider(LABELS[key], 0, 100, 25, key=prefix + key)
    if sum(values.values()) <= 0:
        st.warning("Select at least one positive priority.")
        return None
    return normalize_weights(values)


def render_historical_preference_prior_tab():
    st.title("Historical Preference Prior")
    st.write("Use this when no dataset is available. It summarizes the validated development outcomes used by the current recommender, aggregated across seeds before comparing frameworks across datasets.")
    st.info("Historical aggregation · not machine learning. This is a global starting point, not a prediction for your dataset.")
    status = bundle_status()
    render_training_cards(status)
    if not status["ready"]:
        render_pending(status)
        return
    render_protocol(status)
    weights = _weight_controls("current_prior_")
    if weights is None:
        return
    try:
        ranked, _ = historical_prior(weights, status["directory"])
    except Exception as exc:
        st.error(str(exc))
        return
    st.metric("Historical framework starting point", ranked.iloc[0].framework_label)
    table = ranked[["rank", "framework_label", "historical_utility", "win_count", "support_datasets"]].rename(columns={
        "rank": "Rank", "framework_label": "Framework", "historical_utility": "Cross-dataset preference score",
        "win_count": "Dataset wins", "support_datasets": "Development datasets"})
    st.dataframe(table,
                 use_container_width=True, hide_index=True)
    st.caption("Preference score is mean within-dataset utility across development datasets. Each framework/dataset outcome is first averaged over all declared seeds. Scores are not probabilities. Tied dataset winners are all counted.")
    render_validation(status)


def render_dataset_aware_tab():
    from .state import ensure_research_state
    state = ensure_research_state()
    st.title("Dataset-aware ML Recommender")
    st.write("The trained objective models use up to the first 1,000 labeled observations to predict framework outcomes before framework execution. You choose the Accuracy, Runtime, Energy and CO₂ priorities.")
    st.info("Dataset-aware model inference. Learned models produce the ranking; the LLM may explain the evidence afterwards but does not choose the framework.")
    status = bundle_status()
    render_training_cards(status)
    if not status["ready"]:
        render_pending(status)
        return
    render_protocol(status)
    frame, target = state.get("dataset"), state.get("target")
    if frame is None or not target or target not in frame.columns:
        st.info("Load a labeled classification dataset and choose its target in Run Studio. The same dataset context is reused here.")
        render_validation(status)
        return
    st.caption("Dataset: {} · Target: {}".format(state.get("dataset_name") or "Loaded dataset", target))
    weights = _weight_controls("current_dataset_")
    if weights is None:
        return
    try:
        ranked, meta = load_current_recommender(status["directory"]).recommend_dataframe(frame, target, weights)
    except Exception as exc:
        st.error("Dataset-aware inference unavailable: " + str(exc))
        return
    st.metric("Recommended framework", ranked.iloc[0].framework_label)
    st.caption("Predicted outcomes under the frozen training protocol. Utility is a relative preference score, not confidence or observed performance.")
    table = ranked[["rank", "framework_label", "utility", "accuracy", "runtime", "energy", "co2"]].rename(columns={
        "rank": "Rank", "framework_label": "Framework", "utility": "Preference utility",
        "accuracy": "Predicted accuracy", "runtime": "Predicted runtime (s)",
        "energy": "Predicted energy (kWh)", "co2": "Predicted CO₂ (kg)"})
    st.dataframe(table,
                 use_container_width=True, hide_index=True)
    st.bar_chart(ranked.set_index("framework_label").utility)
    with st.expander("Observed prefix and prediction intervals", expanded=False):
        st.json(meta)
        st.caption("Intervals use empirical nested residuals; external coverage must be assessed. No full-stream length, family, source or drift-type input is required.")
    from .copilot_three_path import _render_grounded_recommendation_explainer
    signature = status["evidence_id"] + json.dumps(meta, sort_keys=True, default=str) + ranked.to_json()
    _render_grounded_recommendation_explainer(state, ranked, weights,
        str(state.get("dataset_name") or "Loaded dataset"), target, signature)
    render_validation(status)


def render_current_goal_guidance(state, interpretation, proposal=None):
    st.markdown("## 2 · Framework guidance from the evidence available now")
    status = bundle_status()
    if not status["ready"]:
        render_pending(status)
        return
    data = interpretation.model_dump() if hasattr(interpretation, "model_dump") else dict(interpretation or {})
    weights = state.get("copilot_human_corrected_weights") or data.get("primary_weights") or {}
    if hasattr(weights, "normalized_dict"):
        weights = weights.normalized_dict()
    try:
        ranked, _ = historical_prior(weights, status["directory"])
    except Exception as exc:
        st.warning(str(exc))
        return
    prior = ranked.iloc[0]
    st.info("Historical starting point · no dataset required. Seed-aggregated development evidence provides global framework guidance under the active priorities.")
    st.metric("Historical framework starting point", prior.framework_label)
    if proposal is not None:
        data = proposal.model_dump() if hasattr(proposal, "model_dump") else dict(proposal)
        st.markdown("### Compare the available evidence paths")
        left, right = st.columns(2)
        left.metric("Global historical prior", prior.framework_label)
        right.metric("Dataset-aware ML Recommender", data.get("ml_recommender_framework", "Unavailable"))
        st.caption("The global prior averages development utilities; the learned ranking conditions on the observed prefix. Their score magnitudes are not directly comparable.")
    with st.expander("Compare the historical alternatives", expanded=False):
        st.dataframe(ranked, use_container_width=True, hide_index=True)
