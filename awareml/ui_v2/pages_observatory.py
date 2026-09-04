from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .components import empty_state, hero, section
from .data import load_phase8_report
from .plots import (
    faithfulness_components,
    rai_metric_bar,
    temporal_metric_figure,
)
from .state import result_dicts
from .page_utils import phase_pills, plot, results_frame, fmt
from .pre14_usability import prepare_drift_display



def _safe_mean(values):
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    return float(np.mean(vals)) if vals else None



def streaming_observatory_page():
    hero(
        "TEMPORAL ANALYTICS",
        "Streaming Observatory",
        (
            "Inspect observed stream dynamics across frameworks. Drift and refit "
            "events are synchronized across temporal plots so performance, latency "
            "and recovery can be interpreted in context."
        ),
        pills=phase_pills(),
    )

    results = result_dicts()
    if not results:
        empty_state(
            "No observed run results",
            "Execute an experiment in Run Studio. This page only visualizes measured run evidence.",
        )
        return

    frame = results_frame()

    best_idx = frame["Accuracy"].astype(float).idxmax() if not frame.empty else None
    best_framework = frame.loc[best_idx, "Framework"] if best_idx is not None else "N/A"
    best_accuracy = frame.loc[best_idx, "Accuracy"] if best_idx is not None else None
    all_drifts = sum(len(result.get("drift_events") or []) for result in results)
    mean_recovery = _safe_mean((result.get("drift_summary") or {}).get("recovery_rate") for result in results)
    latency = _safe_mean(result.get("mean_prediction_latency_ms") for result in results)

    cols = st.columns(4)
    with cols[0]:
        st.metric("Best final accuracy", f"{best_framework} · {fmt(best_accuracy, 3)}")
    with cols[1]:
        st.metric("Framework-level drift alerts", str(int(all_drifts)))
    with cols[2]:
        st.metric("Mean drift recovery", fmt(mean_recovery, 2))
    with cols[3]:
        st.metric("Median p95 latency", fmt(latency, 2, " ms"))

    section("Run overview", "Observed framework outcomes from the current active experiment.")
    st.dataframe(frame, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="r9-callout">
          <b>Reading the temporal plots:</b> red triangle/dotted markers show <b>drift detections</b>.
          Green diamond/dashed markers show <b>explicitly recorded refit/retrain events</b> when the backend exposes them.
          Teal recovery markers show an <b>observed recovery point</b> when the drift episode records one.
          A continuously updating online learner can adapt after drift without logging a discrete refit, so the UI never fabricates a refit marker.
        </div>
        """,
        unsafe_allow_html=True,
    )

    plot_results = prepare_drift_display(results)

    section(
        "Temporal performance",
        "Research-grade temporal views with synchronized event markers and extra plot padding to avoid clipping.",
    )
    c1, c2 = st.columns(2)
    with c1:
        plot(temporal_metric_figure(plot_results, "accuracy", "Prequential accuracy", "Accuracy"), "obs_accuracy")
    with c2:
        plot(temporal_metric_figure(plot_results, "f1_macro", "Macro-F1", "Macro-F1"), "obs_f1")

    c3, c4 = st.columns(2)
    with c3:
        plot(
            temporal_metric_figure(plot_results, "rolling_accuracy", "Rolling accuracy", "Rolling accuracy"),
            "obs_rolling_accuracy",
        )
    with c4:
        plot(
            temporal_metric_figure(
                plot_results, "mean_prediction_latency_ms", "Prediction latency", "Milliseconds"
            ),
            "obs_latency",
        )

    section(
        "Drift recovery evidence",
        "Recovery is displayed only when the runner recorded a scientifically applicable recovery assessment.",
    )
    rows = []
    for result in results:
        summary = result.get("drift_summary") or {}
        rows.append({
            "Framework": result.get("framework"),
            "Drift events": len(result.get("drift_events") or []),
            "Recovery-applicable": summary.get("n_recovery_applicable"),
            "Recovered": summary.get("n_recovered"),
            "Recovery rate": summary.get("recovery_rate"),
            "Median recovery samples": summary.get("median_recovery_samples"),
            "Max accuracy drop": summary.get("max_accuracy_drop"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)



def responsible_ai_page():
    hero(
        "RESPONSIBLE AI OBSERVATORY",
        "Fairness · Explainability · Sustainability · Faithfulness",
        (
            "A single evidence surface for the four responsible-AI dimensions. "
            "Current-run fairness/XAI/sustainability are kept distinct from "
            "Phase-8 development faithfulness evidence."
        ),
        pills=phase_pills(),
    )

    results = result_dicts()
    frame = results_frame()
    faith = load_phase8_report()

    if results:
        section(
            "Current-run responsible-AI snapshot",
            "These values come from the active observed benchmark run.",
        )
        c1, c2 = st.columns(2)
        with c1:
            plot(
                rai_metric_bar(frame, "DP gap", "Demographic parity gap", lower_is_better=True),
                "rai_dp",
            )
        with c2:
            plot(
                rai_metric_bar(frame, "XAI fidelity", "Explanation fidelity", lower_is_better=False),
                "rai_xai_fid",
            )
        c3, c4 = st.columns(2)
        with c3:
            plot(
                rai_metric_bar(frame, "Energy (kWh)", "Energy consumption", lower_is_better=True),
                "rai_energy",
            )
        with c4:
            plot(
                rai_metric_bar(frame, "CO2 (kg)", "CO₂ emissions", lower_is_better=True),
                "rai_co2",
            )
        st.dataframe(frame, use_container_width=True, hide_index=True)
    else:
        empty_state(
            "Current-run RAI metrics unavailable",
            (
                "Run an experiment to populate fairness, explainability and sustainability evidence. "
                "Phase-8 faithfulness evidence is still available below."
            ),
        )

    section(
        "Faithfulness development evidence",
        (
            "Phase 8 evaluates whether recommendation rationales react to counterfactual evidence. "
            "It is not a current-run fairness/XAI metric."
        ),
    )

    if not faith:
        empty_state("Phase-8 report not found", "Run and freeze Phase 8 before using this panel.")
        return

    deterministic = faith.get("deterministic") or {}
    ollama = faith.get("ollama") or {}

    cols = st.columns(4)
    with cols[0]:
        st.metric("Deterministic AEF", fmt(deterministic.get("mean_evidence_fidelity_score"), 3))
    with cols[1]:
        st.metric("Counterfactual sensitivity", fmt(deterministic.get("mean_counterfactual_sensitivity"), 3))
    with cols[2]:
        st.metric("Live Ollama AEF", fmt(ollama.get("mean_evidence_fidelity_score"), 3))
    with cols[3]:
        st.metric("Development datasets", str(deterministic.get("n_datasets", 0)))

    plot(faithfulness_components(deterministic), "rai_faith_components")

    st.markdown(
        """
        <div class="r9-callout">
          <b>Scientific boundary:</b> AEF is the AwareML project-defined
          external evidence-faithfulness composite. Phase 8 does not claim
          internal Ollama PE-LRP or attention attribution.
        </div>
        """,
        unsafe_allow_html=True,
    )
