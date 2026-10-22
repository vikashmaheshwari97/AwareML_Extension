from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .components import empty_state, hero, section
from .data import load_phase8_report
from .plots import temporal_metric_figure, apply_research_layout
from .state import result_dicts, ensure_research_state
from .page_utils import phase_pills, plot, results_frame, fmt
from .pre14_usability import prepare_drift_display
from .temporal_research import (
    adaptation_frame,
    detector_health_frame,
    render_drift_episode_figures,
)
from .temporal_visuals_v6 import render_window_research_comparison
from .stream_shift_audit import (
    window_distribution_shift,
    window_feature_distribution_shift,
    distribution_shift_figure,
    jsd_density_figure,
    jsd_feature_ranking_figure,
)
from .responsible_ai_visuals_v6_4 import (
    FAIRNESS_GROUP_METRICS,
    fairness_group_frame,
    group_metric_profile_figure,
    fairness_disparity_structure_figure,
    fairness_tpr_fpr_plane,
    fairness_over_time_figure,
    fairness_performance_frontier,
    rai_directional_profile_figure,
    rai_rank_flow_figure,
    rai_dimension_leaders,
    faithfulness_evidence_profile_figure,
    faithfulness_aef_comparison_figure,
    faithfulness_summary,
)


def _safe_mean(values):
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    return float(np.mean(vals)) if vals else None


def streaming_observatory_page():
    hero(
        "TEMPORAL ANALYTICS",
        "Streaming Observatory",
        (
            "Inspect predictive-performance change, distribution shift, explicit adaptation and recovery without collapsing distinct temporal signals into one drift label."
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
          <b>Three temporal signals are kept separate.</b><br>
          <b>Predictive-performance change</b> is detected from prequential prediction error/performance by the configured ADWIN/Hybrid controller.<br>
          <b>Dataset-composition shift</b> is a descriptive Jensen–Shannon divergence audit of feature/target distributions.<br>
          <b>Fairness dynamics</b> are evaluated separately in Fairness Lab. A change in one signal does not automatically prove a change in the others.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section(
        "Detector health & adaptation",
        "The current Hybrid detector preserves the useful ADWIN lineage from the earlier implementation, then adds Page-Hinkley, sustained-degradation confirmation, warm-up and minimum-separation protection.",
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
            "No explicit adaptation action was recorded. Online learners may still adapt continuously. AwareML does not invent a refit marker when the backend did not record one."
        )

    section(
        "Predictive drift vs dataset-composition shift",
        "A stream can show zero confirmed predictive drift while its feature distributions move. The panels below explain that distinction rather than forcing a drift event.",
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
            "Zero confirmed predictive drift is scientifically possible. ADWIN/Page-Hinkley receive prediction-error/performance evidence; they do not test every feature distribution. "
            "A stable error process can therefore coexist with smaller composition changes or fairness variation."
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
        feature_audit = window_feature_distribution_shift(
            dataset,
            state.get("target"),
            window_size=window_size,
            processed_samples=processed_samples or None,
        )
        if not audit.empty and not feature_audit.empty:
            st.markdown("#### Jensen–Shannon distribution-shift audit")
            st.markdown(
                """
                **What JSD means.** For empirical distributions *P* and *Q*, AwareML uses
                **JSD(P,Q) = ½ KL(P‖M) + ½ KL(Q‖M)** where **M = ½(P+Q)**. With the base-2 logarithm used here, JSD is bounded from **0 to 1**: 0 means the two binned/categorical empirical distributions are identical; larger values mean stronger separation.

                Every processed window is compared with the **first processed window**. Continuous features use reference-anchored quantile bins so the comparison is stable across windows. This is a **descriptive composition-shift diagnostic**, not a hypothesis test and not a substitute for predictive-error drift detection.
                """
            )
            left, right = st.columns([1.15, 0.85])
            with left:
                fig = jsd_density_figure(feature_audit)
                plot(fig, "r120_jsd_density")
                st.caption(
                    "Each curve is the empirical distribution of feature-level JSD values in one stream window. The final observed window is filled. "
                    "The dotted Observed P95 line summarizes the run's own feature-shift distribution; it is not a universal drift threshold."
                )
            with right:
                fig = jsd_feature_ranking_figure(feature_audit)
                plot(fig, "r120_jsd_feature_rank")
                st.caption(
                    "For each feature, the interval runs from its median to its maximum observed JSD across non-reference windows. This reveals whether shift is broad or concentrated in a few variables."
                )

            max_jsd = pd.to_numeric(audit["Max feature JSD"], errors="coerce").max()
            p95_jsd = pd.to_numeric(audit.get("P95 feature JSD"), errors="coerce").max()
            m1, m2, m3 = st.columns(3)
            m1.metric("Largest feature JSD", "N/A" if pd.isna(max_jsd) else "{:.4f}".format(float(max_jsd)))
            m2.metric("Largest window P95 JSD", "N/A" if pd.isna(p95_jsd) else "{:.4f}".format(float(p95_jsd)))
            target_max = pd.to_numeric(audit.get("Target JSD"), errors="coerce").max()
            m3.metric("Largest target JSD", "N/A" if pd.isna(target_max) else "{:.4f}".format(float(target_max)))

            with st.expander("JSD trend and exact window-level composition audit", expanded=False):
                plot(distribution_shift_figure(audit), "r120_distribution_shift_trend")
                st.dataframe(audit, use_container_width=True, hide_index=True)
                st.dataframe(feature_audit, use_container_width=True, hide_index=True)

    st.caption(
        "Fairness-specific temporal change and accumulated fairness debt remain in Advanced Labs → Fairness Lab; they are intentionally not counted as predictive drift events."
    )

    plot_results = prepare_drift_display(results)
    section(
        "Temporal performance",
        "Prequential and rolling performance are shown with synchronized measured event markers. No synthetic drift markers are generated.",
    )
    c1, c2 = st.columns(2)
    with c1:
        plot(temporal_metric_figure(plot_results, "accuracy", "Prequential accuracy", "Accuracy"), "obs_accuracy")
    with c2:
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
        "Post-change accuracy & recovery",
        "Episode views are derived only from the current uploaded dataset/run. A detector alert, an explicit adaptation and a measured recovery are separate recorded events.",
    )
    render_drift_episode_figures(results)

    section(
        "Window-level comparison",
        "The former color heatmap is replaced by a quantitative deviation-and-rank view: it shows both the magnitude of within-framework degradation and cross-framework ordering over time.",
    )
    render_window_research_comparison(results)

    section(
        "Recovery summary",
        "Recovery is reported only when a material post-change degradation was observed and enough post-event samples were available.",
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
        "Detector note: Hybrid mode combines ADWIN, Page-Hinkley and a fast-vs-slow rolling-accuracy degradation check. A bounded sustained performance drop can be confirmed when statistical detectors remain silent, with warm-up/minimum-separation protection. "
        "This is evidence of a sustained predictive-performance change, not automatic proof of a changed data-generating concept. ADWIN compatibility mode remains available for protocol comparisons."
    )


def _selection_points(event):
    """Return Plotly selection points across Streamlit's dict/object variants."""
    if event is None:
        return []
    try:
        return list(event.selection.points or [])
    except Exception:
        pass
    try:
        selection = event.get("selection", {})
        if hasattr(selection, "get"):
            return list(selection.get("points") or [])
    except Exception:
        pass
    return []


def responsible_ai_page():
    hero(
        "RESPONSIBLE AI OBSERVATORY",
        "Fairness · Explainability · Sustainability · Faithfulness",
        (
            "Inspect disaggregated fairness evidence first, then connect it to current-run explanation and resource evidence without collapsing the dimensions into one Responsible-AI score."
        ),
        pills=phase_pills(),
    )

    results = result_dicts()
    frame = results_frame()
    faith = load_phase8_report()
    state = ensure_research_state()

    fairness_tab, overview_tab, faith_tab = st.tabs([
        "Fairness Observatory",
        "Cross-dimension evidence",
        "Faithfulness development evidence",
    ])

    with fairness_tab:
        if not results:
            empty_state(
                "Current-run fairness evidence unavailable",
                "Run an experiment with a sensitive attribute to populate group-level fairness evidence.",
            )
        else:
            eligible = [
                r for r in results
                if isinstance((r.get("fairness") or {}).get("group_performance"), dict)
                and (r.get("fairness") or {}).get("group_performance")
            ]
            if not eligible:
                empty_state(
                    "Group-level fairness evidence unavailable",
                    "The active run does not contain at least two supported sensitive groups. Re-run with a sensitive attribute and sufficient group support.",
                )
            else:
                framework_names = [str(r.get("framework")) for r in eligible]
                remembered_fw = state.get("rai_fairness_framework")
                default_i = framework_names.index(remembered_fw) if remembered_fw in framework_names else 0
                selected_fw = st.selectbox(
                    "Framework for disaggregated fairness audit",
                    framework_names,
                    index=default_i,
                    key="rai_v64_framework",
                )
                state["rai_fairness_framework"] = selected_fw
                selected_result = next(r for r in eligible if str(r.get("framework")) == selected_fw)
                fair = selected_result.get("fairness") or {}
                group_frame = fairness_group_frame(fair)
                groups = group_frame["Group"].astype(str).tolist() if not group_frame.empty else []

                st.markdown(
                    """
                    <div class="r9-callout">
                      <b>Disaggregated fairness first.</b> Exact group values are shown by position, not box area or a single fairness score. Benefit metrics and error metrics remain separate so reviewers can see where disparity comes from.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                section(
                    "Group × metric fairness profile",
                    "Six aligned small multiples replace the previous square matrix. Each dot is one observed group value on the same 0–1 axis; the shaded span is the observed between-group range for that metric.",
                )
                fig = group_metric_profile_figure(fair)
                apply_research_layout(
                    fig, height=620, legend="bottom",
                    title=f"{selected_fw} · disaggregated group evidence",
                    bottom_margin=104,
                )
                plot(fig, "rai_v64_group_metric_profile")
                st.caption(
                    "Position carries the quantitative value. Group support is kept in hover and the exact table rather than occupying a separate dashboard control. Selection rate is descriptive (↔); it is not automatically higher- or lower-is-better."
                )

                left, right = st.columns([1.05, 0.95])
                with left:
                    fig = fairness_disparity_structure_figure(fair)
                    apply_research_layout(
                        fig, height=410, legend="none",
                        title="Disparity structure across group metrics",
                        bottom_margin=66,
                    )
                    plot(fig, "rai_v64_disparity_structure")
                    st.caption(
                        "Each endpoint is the observed max–min group range. Smaller gaps mean closer parity, but parity alone does not establish good predictive performance."
                    )
                with right:
                    fig = fairness_tpr_fpr_plane(fair)
                    apply_research_layout(
                        fig, height=410, legend="none",
                        title="Group operating points · TPR vs FPR",
                        bottom_margin=66,
                    )
                    plot(fig, "rai_v64_tpr_fpr")
                    st.caption(
                        "This is one observed operating point per supported group, not a threshold-sweep ROC curve. Upper-left combines higher TPR with lower FPR."
                    )

                with st.expander("Group support & exact disaggregated values", expanded=False):
                    st.dataframe(group_frame, use_container_width=True, hide_index=True)
                    st.caption(
                        "Support remains auditable here and in plot hover. It is not used as marker area, so sample size cannot visually overpower the fairness quantities themselves."
                    )

                section(
                    "Fairness over time",
                    "Track whether disparity changes as the stream evolves. Predictive-change markers are shown only as temporal reference points and are not relabelled as fairness drift.",
                )
                fig = fairness_over_time_figure(selected_result)
                if fig.data:
                    apply_research_layout(
                        fig, height=470, legend="bottom",
                        title=f"{selected_fw} · fairness gaps across stream windows",
                        bottom_margin=104,
                    )
                    plot(fig, "rai_v64_fairness_over_time")
                else:
                    st.info("This run has final group evidence but no window-level fairness trajectory for the selected framework.")

                section(
                    "Fairness–performance frontier",
                    "Compare prequential accuracy against demographic-parity disparity across frameworks without reducing both objectives to one score.",
                )
                fig = fairness_performance_frontier(results, "dp_diff")
                if fig.data:
                    apply_research_layout(
                        fig, height=460, legend="none",
                        title="Accuracy–fairness frontier · demographic parity gap",
                        bottom_margin=62,
                    )
                    plot(fig, "rai_v64_fairness_performance_frontier")
                    st.caption(
                        "Outlined diamonds are nondominated for higher Accuracy and lower demographic-parity gap. This is an exploratory two-objective frontier, not a claim that demographic parity is the only fairness definition."
                    )

                with st.expander("Intersectional and threshold-sensitivity readiness", expanded=False):
                    if len(groups) > 2:
                        st.success(
                            "The selected sensitive attribute contains {} supported group labels, so the view already exposes multi-group disparities. If those labels encode intersections, interpret them intersectionally only with matching dataset provenance.".format(len(groups))
                        )
                    else:
                        st.info(
                            "The active run exposes {} supported group label(s) for one sensitive attribute. A genuine intersectional audit requires explicitly encoded intersectional groups or multiple protected attributes; AwareML does not invent them.".format(len(groups))
                        )
                    st.info(
                        "Fairness-threshold sensitivity requires stored group-level probability/label samples across thresholds. The current benchmark keeps aggregate calibration evidence, so no synthetic threshold curve is drawn."
                    )

    with overview_tab:
        if results and not frame.empty:
            section(
                "Cross-dimension Responsible-AI evidence",
                "Direction-align the measured dimensions for comparison, then inspect rank changes. Exact values remain available in hover/table and no overall Responsible-AI score is computed.",
            )

            leaders = rai_dimension_leaders(frame)
            if leaders:
                cols = st.columns(len(leaders))
                for col, item in zip(cols, leaders):
                    value = item.get("Value")
                    label = item.get("Dimension", "Dimension")
                    if value is None:
                        rendered = "N/A"
                    elif "Energy" in label or "CO₂" in label:
                        rendered = f"{float(value):.2e}"
                    else:
                        rendered = f"{float(value):.3f}"
                    col.metric(label, str(item.get("Framework") or "N/A"), rendered)

            left, right = st.columns([1.12, 0.88])
            with left:
                fig = rai_directional_profile_figure(frame)
                apply_research_layout(
                    fig, height=500, legend="bottom",
                    title="Direction-aligned Responsible-AI profiles",
                    bottom_margin=112,
                )
                plot(fig, "rai_v64_cross_dimension_profile")
                st.caption(
                    "Every axis is independently normalized within the current run only. The profile exposes trade-offs; it is not an aggregate score and values on different axes must not be numerically combined."
                )
            with right:
                fig = rai_rank_flow_figure(frame)
                apply_research_layout(
                    fig, height=500, legend="bottom",
                    title="Cross-dimension rank flow",
                    bottom_margin=112,
                )
                plot(fig, "rai_v64_cross_dimension_rank_flow")
                st.caption(
                    "Rank crossings show where a framework that leads one Responsible-AI dimension becomes weaker on another. Rank 1 is best within that dimension only."
                )

            with st.expander("Exact current-run Responsible-AI evidence", expanded=False):
                st.dataframe(frame, use_container_width=True, hide_index=True)
                st.caption(
                    "Accuracy and XAI fidelity are higher-is-better; demographic-parity gap, Energy and CO₂ are lower-is-better. Missing evidence remains N/A."
                )
        else:
            empty_state(
                "Current-run RAI metrics unavailable",
                "Run an experiment to populate fairness, explainability and sustainability evidence.",
            )

    with faith_tab:
        section(
            "Faithfulness development evidence",
            "Phase 8 evaluates whether recommendation rationales react to counterfactual evidence. It is development evidence, not a current-run fairness/XAI metric.",
        )
        if not faith:
            empty_state("Phase-8 report not found", "Run and freeze Phase 8 before using this panel.")
        else:
            deterministic = faith.get("deterministic") or {}
            ollama = faith.get("ollama") or {}
            summary = faithfulness_summary(deterministic)
            det_aef = deterministic.get("mean_evidence_fidelity_score")
            live_aef = ollama.get("mean_evidence_fidelity_score")
            try:
                delta = float(live_aef) - float(det_aef)
            except Exception:
                delta = None

            cols = st.columns(5)
            cols[0].metric("Deterministic AEF", fmt(det_aef, 3))
            cols[1].metric("Live Ollama AEF", fmt(live_aef, 3), None if delta is None else f"{delta:+.3f}")
            cols[2].metric("Lowest component", str(summary.get("lowest_label") or "N/A"), fmt(summary.get("lowest_value"), 3))
            cols[3].metric("Component spread", fmt(summary.get("spread"), 3))
            cols[4].metric("Development datasets", str(deterministic.get("n_datasets", 0)))

            left, right = st.columns([1.12, 0.88])
            with left:
                fig = faithfulness_evidence_profile_figure(deterministic)
                apply_research_layout(
                    fig, height=430, legend="none",
                    title="Phase-8 evidence profile · components vs deterministic AEF",
                    bottom_margin=66,
                )
                plot(fig, "rai_v64_faith_profile")
                st.caption(
                    "The dashed AEF line is a reference for the recorded Phase-8 components; component scores remain individually visible rather than being hidden behind the composite."
                )
            with right:
                fig = faithfulness_aef_comparison_figure(deterministic, ollama)
                apply_research_layout(
                    fig, height=430, legend="bottom",
                    title="Deterministic ↔ Live Ollama AEF",
                    bottom_margin=100,
                )
                plot(fig, "rai_v64_faith_aef_compare")
                st.caption(
                    "The dumbbell shows the observed AEF difference between deterministic and live-Ollama evaluation paths; it does not imply statistical significance without repeated uncertainty estimates."
                )

            if summary.get("lowest_value") is not None:
                st.info(
                    "Lowest recorded Phase-8 component: **{} = {:.3f}**. Component spread = **{:.3f}**. This is a descriptive development diagnostic, not a causal decomposition of AEF.".format(
                        summary.get("lowest_label"),
                        float(summary.get("lowest_value")),
                        float(summary.get("spread") or 0.0),
                    )
                )

            st.markdown(
                """
                <div class="r9-callout"><b>Scientific boundary:</b> AEF is the AwareML project-defined external evidence-faithfulness composite. Phase 8 does not claim internal Ollama PE-LRP or attention attribution. The component panels are development evidence and are intentionally kept separate from current-run model XAI metrics.</div>
                """,
                unsafe_allow_html=True,
            )
