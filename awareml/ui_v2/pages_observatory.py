from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .components import empty_state, hero, section
from .data import load_phase8_report
from .plots import faithfulness_components, rai_metric_bar, temporal_metric_figure
from .state import result_dicts, ensure_research_state
from .page_utils import phase_pills, plot, results_frame, fmt
from .pre14_usability import prepare_drift_display
from .temporal_research import (
    adaptation_frame,
    detector_health_frame,
    render_drift_episode_figures,
    render_window_accuracy_heatmap,
)
from .stream_shift_audit import window_distribution_shift, distribution_shift_figure


def _safe_mean(values):
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    return float(np.mean(vals)) if vals else None


def streaming_observatory_page():
    hero(
        "TEMPORAL ANALYTICS",
        "Streaming Observatory",
        (
            "Inspect observed stream dynamics, drift evidence, adaptation actions and recovery across frameworks. "
            "This workspace now contains the former Drift & Temporal Lab deep-dive so temporal analysis is not duplicated."
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
    all_adaptations = sum(
        len(result.get("refit_events") or (result.get("drift_summary") or {}).get("refit_events") or [])
        for result in results
    )
    mean_recovery = _safe_mean((result.get("drift_summary") or {}).get("recovery_rate") for result in results)

    cols = st.columns(4)
    cols[0].metric("Best final accuracy", f"{best_framework} · {fmt(best_accuracy, 3)}")
    cols[1].metric("Confirmed performance-change alerts", str(int(all_drifts)))
    cols[2].metric("Explicit adaptation actions", str(int(all_adaptations)))
    cols[3].metric("Mean drift recovery", fmt(mean_recovery, 2))

    section("Run overview", "Observed framework outcomes from the current active experiment.")
    st.dataframe(frame, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="r9-callout">
          <b>Event semantics:</b> red temporal markers are measured detector alerts. Green adaptation markers are shown only when the runner records an explicit action.
          Teal recovery markers require a measured recovery episode. A continuously updating online learner can adapt without a discrete refit, so AwareML never invents refit markers.
          The main accuracy remains <b>prequential test-then-train</b>; this page does not inflate it by replacing it with a post-fit score.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section(
        "Detector health & adaptation",
        "Compare the active detector mode, confirmed alerts, explicit reset/replay actions and scientifically applicable recovery evidence.",
    )
    health = detector_health_frame(results)
    st.dataframe(health, use_container_width=True, hide_index=True)
    adaptations = adaptation_frame(results)
    if not adaptations.empty:
        with st.expander("Recorded adaptation actions", expanded=True):
            st.dataframe(adaptations, use_container_width=True, hide_index=True)
            st.caption(
                "awareml_reset_and_replay is an AwareML wrapper action: the framework is reset and a bounded recent labelled buffer is replayed. "
                "It is not described as an upstream-native refit unless the backend explicitly exposes one."
            )
    else:
        st.info(
            "No explicit adaptation action was recorded. For historical/frozen runs this usually means monitor-only drift handling; "
            "new interactive runs can enable Hybrid detection + Adaptive reset/replay in Run Studio."
        )

    section(
        "Why can a stream show zero confirmed predictive drift?",
        "AwareML separates predictive-performance change, raw data-composition shift, and fairness dynamics. They are related but not interchangeable signals.",
    )
    state = ensure_research_state()
    processed_samples = 0
    for result in results:
        try:
            processed_samples = max(processed_samples, int(result.get("samples") or 0))
        except Exception:
            pass
        for point in result.get("points") or []:
            try:
                processed_samples = max(processed_samples, int(point.get("sample") or 0))
            except Exception:
                pass
    dataset = state.get("dataset")
    dataset_rows = len(dataset) if isinstance(dataset, pd.DataFrame) else None
    coverage = (processed_samples / dataset_rows) if dataset_rows and processed_samples else None
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Dataset rows available", "N/A" if dataset_rows is None else f"{dataset_rows:,}")
    d2.metric("Rows processed in this run", f"{processed_samples:,}" if processed_samples else "N/A")
    d3.metric("Stream coverage", "N/A" if coverage is None else "{:.1%}".format(min(1.0, coverage)))
    d4.metric("Confirmed performance changes", str(int(all_drifts)))
    if all_drifts == 0:
        st.info(
            "Zero confirmed predictive drift is scientifically possible: ADWIN/Page-Hinkley operate on prediction error/performance, not directly on the fairness gap or every feature distribution. "
            "A stable error process can therefore coexist with fairness violations or smaller distribution changes. In v4 Hybrid mode, a sufficiently sustained material performance drop can also become a confirmed performance-change episode instead of remaining as thousands of warning steps."
        )
    if isinstance(dataset, pd.DataFrame) and dataset_rows:
        try:
            window_size = int(state.get("window_size") or 500)
        except Exception:
            window_size = 500
        audit = window_distribution_shift(
            dataset,
            state.get("target"),
            window_size=window_size,
            processed_samples=processed_samples or None,
        )
        if not audit.empty:
            fig = distribution_shift_figure(audit)
            plot(fig, "r100_distribution_shift_audit")
            max_jsd = pd.to_numeric(audit["Max feature JSD"], errors="coerce").max()
            if pd.notna(max_jsd):
                st.caption(
                    "Largest observed feature-distribution JSD from the first processed window: {:.4f}. JSD is a descriptive composition-shift diagnostic here, not a replacement for the configured predictive drift detector.".format(float(max_jsd))
                )
            with st.expander("Window-level data-composition audit", expanded=False):
                st.dataframe(audit, use_container_width=True, hide_index=True)
    st.caption(
        "For fairness-specific temporal change and accumulated fairness debt, use Advanced Labs → Fairness Lab. Those signals are intentionally not counted as predictive drift events here."
    )

    plot_results = prepare_drift_display(results)
    section(
        "Temporal performance",
        "The former Drift & Temporal Lab views are consolidated here with synchronized event markers.",
    )
    c1, c2 = st.columns(2)
    with c1:
        plot(temporal_metric_figure(plot_results, "accuracy", "Prequential accuracy", "Accuracy"), "obs_accuracy")
    with c2:
        # MetricPoint stores f1_macro; older runs may expose rolling_f1_macro as well.
        plot(temporal_metric_figure(plot_results, "f1_macro", "Macro-F1", "Macro-F1"), "obs_f1")
    c3, c4 = st.columns(2)
    with c3:
        plot(temporal_metric_figure(plot_results, "rolling_accuracy", "Rolling accuracy", "Rolling accuracy"), "obs_rolling_accuracy")
    with c4:
        plot(temporal_metric_figure(plot_results, "mean_prediction_latency_ms", "Prediction latency", "Milliseconds"), "obs_latency")
    c5, c6 = st.columns(2)
    with c5:
        plot(temporal_metric_figure(plot_results, "rolling_f1_macro", "Rolling Macro-F1", "Macro-F1"), "obs_rolling_f1")
    with c6:
        plot(temporal_metric_figure(plot_results, "throughput_samples_sec", "Throughput", "Samples / second"), "obs_throughput")

    section(
        "Post-drift accuracy & drift-loss recovery",
        "Single-dataset episode views derived only from the current uploaded dataset/run; no multi-dataset reference values are copied into the dashboard.",
    )
    render_drift_episode_figures(results)

    section("Window-level comparison", "A compact cross-framework view makes sustained degradation easier to see than isolated alerts.")
    render_window_accuracy_heatmap(results)

    section(
        "Recovery summary",
        "Recovery is reported only when a material post-drift degradation was observed and enough post-drift samples were available.",
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
            "Mean accuracy drop": summary.get("mean_accuracy_drop"),
            "Max accuracy drop": summary.get("max_accuracy_drop"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(
        "Detector note: Hybrid mode combines ADWIN, Page-Hinkley and a fast-vs-slow rolling-accuracy degradation check. v4 also confirms a bounded sustained performance-drop episode when statistical detectors stay silent, while retaining warm-up/minimum-separation protection. "
        "Such an event is evidence of a sustained performance change, not automatic proof of a changed data-generating concept. Legacy ADWIN mode remains available for protocol comparability."
    )


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
            plot(rai_metric_bar(frame, "DP gap", "Demographic parity gap", lower_is_better=True), "rai_dp")
        with c2:
            plot(rai_metric_bar(frame, "XAI fidelity", "Explanation fidelity", lower_is_better=False), "rai_xai_fid")
        c3, c4 = st.columns(2)
        with c3:
            plot(rai_metric_bar(frame, "Energy (kWh)", "Energy consumption", lower_is_better=True), "rai_energy")
        with c4:
            plot(rai_metric_bar(frame, "CO2 (kg)", "CO₂ emissions", lower_is_better=True), "rai_co2")
        st.dataframe(frame, use_container_width=True, hide_index=True)
    else:
        empty_state(
            "Current-run RAI metrics unavailable",
            "Run an experiment to populate fairness, explainability and sustainability evidence. Phase-8 faithfulness evidence is still available below.",
        )

    section(
        "Faithfulness development evidence",
        "Phase 8 evaluates whether recommendation rationales react to counterfactual evidence. It is not a current-run fairness/XAI metric.",
    )
    if not faith:
        empty_state("Phase-8 report not found", "Run and freeze Phase 8 before using this panel.")
        return

    deterministic = faith.get("deterministic") or {}
    ollama = faith.get("ollama") or {}
    cols = st.columns(4)
    cols[0].metric("Deterministic AEF", fmt(deterministic.get("mean_evidence_fidelity_score"), 3))
    cols[1].metric("Counterfactual sensitivity", fmt(deterministic.get("mean_counterfactual_sensitivity"), 3))
    cols[2].metric("Live Ollama AEF", fmt(ollama.get("mean_evidence_fidelity_score"), 3))
    cols[3].metric("Development datasets", str(deterministic.get("n_datasets", 0)))
    plot(faithfulness_components(deterministic), "rai_faith_components")
    st.markdown(
        """
        <div class="r9-callout"><b>Scientific boundary:</b> AEF is the AwareML project-defined external evidence-faithfulness composite. Phase 8 does not claim internal Ollama PE-LRP or attention attribution.</div>
        """,
        unsafe_allow_html=True,
    )
