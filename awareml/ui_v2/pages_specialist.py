from __future__ import annotations

import time
import uuid

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from awareml.engine.pareto import METRIC_SPECS
from awareml.llm import GroundedChat, ollama_status
from awareml.recommender import RecommendationService
from awareml.studies import StudyStore, TrustCalibrationStudy, classify_follow_up
from awareml.studies.information_seeking import THINK_ALOUD_PROMPTS
from awareml.types import ObjectiveWeights

from .components import hero, section, empty_state
from .page_utils import fmt, phase_pills, plot, results_frame
from .plots import FRAMEWORK_COLORS, OBJECTIVE_COLORS, apply_research_layout, temporal_metric_figure
from .state import ensure_research_state, result_dicts


FAIRNESS_OPTIONS = {
    "Composite (all available gaps)": "composite",
    "Demographic parity": "demographic_parity",
    "Equal opportunity": "equal_opportunity",
    "Equalized odds": "equalized_odds",
    "Predictive parity": "predictive_parity",
    "Error-rate parity": "error_rate",
}

FAIRNESS_POINT_KEYS = {
    "Demographic parity": "dp_diff",
    "Equal opportunity": "eo_diff",
    "Equalized odds": "equalized_odds_gap",
    "Predictive parity": "predictive_parity_diff",
    "Error-rate parity": "error_rate_gap",
}


def _state():
    return ensure_research_state()


def _run_objects():
    return _state().get("run_results") or []


def _colors(frame, col="Framework"):
    return [FRAMEWORK_COLORS.get(str(v), "#64748b") for v in frame[col]]


def _robust_unit(series: pd.Series, direction: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([np.nan] * len(s), index=s.index)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series([0.5 if pd.notna(v) else np.nan for v in s], index=s.index)
    unit = ((s.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)
    return unit if direction == "max" else 1.0 - unit


def _ollama_controls(prefix: str):
    status = ollama_status()
    models = status.get("models") or []
    enabled = st.toggle(
        "Use local Ollama for the grounded answer",
        value=False,
        key=f"{prefix}_enabled",
    )
    default = _state().get("ollama_model") or status.get("resolved_model") or (models[0] if models else "llama3.1:8b")
    model = default
    if enabled and models:
        model = st.selectbox(
            "Ollama model",
            models,
            index=models.index(default) if default in models else 0,
            key=f"{prefix}_model",
        )
        _state()["ollama_model"] = model
    return enabled, model, status


def decision_lab_v2_page():
    hero(
        "POST-RUN DECISION ANALYSIS",
        "Decision Lab",
        (
            "Rank the frameworks using the outcomes that were actually observed in the current benchmark. "
            "This page is deliberately different from the pre-run 3D Decision Space."
        ),
        pills=phase_pills(),
    )

    if not _run_objects():
        empty_state("Run evidence required", "Run the five-framework benchmark in Run Studio first.")
        return

    st.markdown(
        """
        <div class="r9-callout">
          <b>Purpose of Decision Lab:</b> this is an <b>observed post-run ranking</b>.
          A framework is "recommended" here because it has the highest weighted utility
          after the real benchmark results are known. It is not the same as the pre-run
          ML recommendation in 3D Decision Space.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("Observed-objective weights", "Set the importance of each measured criterion. Weights are normalized by the ranking engine over available evidence.")
    defaults = {
        "accuracy": 0.45,
        "runtime": 0.15,
        "energy": 0.10,
        "co2": 0.10,
        "fairness": 0.15,
        "interpretability": 0.05,
    }
    cols = st.columns(6)
    vals = {}
    for col, key in zip(cols, defaults):
        with col:
            vals[key] = st.slider(
                key.replace("co2", "CO₂").title(),
                0.0,
                1.0,
                float(defaults[key]),
                0.05,
                key=f"r95_decision_{key}",
            )

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fair_label = st.selectbox("Fairness criterion in utility", list(FAIRNESS_OPTIONS), key="r95_decision_fairness_metric")
        fair_metric = FAIRNESS_OPTIONS[fair_label]
    with c2:
        epsilon = st.slider("Near-Pareto ε", 0.0, 0.20, 0.05, 0.01, key="r95_decision_eps")

    weights = ObjectiveWeights(**vals)
    service = RecommendationService(epsilon=epsilon, fairness_metric=fair_metric)
    frame, recs, corr, warnings = service.rank(_run_objects(), weights)
    _state()["ranking"] = frame.to_dict(orient="records")
    _state()["fairness_metric"] = fair_metric

    top = recs[0]
    top_row = frame.iloc[0]
    cards = st.columns(5)
    cards[0].metric("Recommended framework", top.framework)
    cards[1].metric("Observed utility", fmt(top.utility, 3))
    cards[2].metric("Observed accuracy", fmt(top_row.get("accuracy"), 3))
    cards[3].metric("Fairness score", fmt(top_row.get("fairness_score"), 3))
    cards[4].metric("Near-Pareto", "Yes" if top.near_pareto else "No")

    st.success(
        "{} is ranked #1 because, under the current weights, it has the highest utility calculated from the observed run evidence.".format(top.framework)
    )
    for warning in warnings:
        st.warning(warning)

    mapping = {
        "accuracy": ("accuracy", "max"),
        "runtime": ("runtime_sec", "min"),
        "energy": ("energy_kwh", "min"),
        "co2": ("co2_kg", "min"),
        "fairness": ("fairness_score", "max"),
        "interpretability": ("interpretability_score", "max"),
    }
    weights_dict = weights.as_dict()
    norm = pd.DataFrame({"framework": frame["framework"]})
    contrib = pd.DataFrame({"framework": frame["framework"]})
    for name, (metric, direction) in mapping.items():
        score = _robust_unit(frame[metric], direction)
        norm[name] = score
        contrib[name] = score * float(weights_dict.get(name, 0.0))

    left, right = st.columns([1.35, 1])
    with left:
        long = contrib.melt(id_vars="framework", var_name="Objective", value_name="Weighted contribution")
        fig = px.bar(
            long,
            x="framework",
            y="Weighted contribution",
            color="Objective",
            barmode="stack",
            title="Utility contribution by objective",
            color_discrete_sequence=["#2563eb", "#0ea5e9", "#10b981", "#14b8a6", "#f59e0b", "#8b5cf6"],
        )
        apply_research_layout(fig, height=420, legend="bottom", title="Utility contribution by objective", bottom_margin=96)
        fig.update_layout(margin=dict(l=52, r=22, t=52, b=100), xaxis_title="Framework")
        plot(fig, "r95_decision_contrib")
    with right:
        heat = norm.set_index("framework")
        fig = px.imshow(
            heat,
            text_auto=".2f",
            zmin=0,
            zmax=1,
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Observed objective desirability · higher is better",
        )
        apply_research_layout(fig, height=420, legend="none", title="Observed objective desirability · higher is better", bottom_margin=54)
        fig.update_layout(margin=dict(l=72, r=42, t=54, b=62), coloraxis_colorbar=dict(len=0.78, thickness=12))
        plot(fig, "r95_decision_heat")

    section("Observed ranking table", "This is the auditable post-run ranking used by this page.")
    display = frame.rename(columns={
        "framework": "Framework", "rank": "Rank", "utility": "Utility", "near_pareto": "Near-Pareto",
        "accuracy": "Accuracy", "runtime_sec": "Runtime (s)", "energy_kwh": "Energy (kWh)",
        "co2_kg": "CO₂ (kg)", "fairness_score": "Fairness score", "interpretability_score": "Interpretability score",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    section("Objective correlation", "Use this to detect redundant objectives such as Energy and CO₂. Correlation is descriptive, not causal.")
    if corr is not None and not corr.empty:
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
        apply_research_layout(fig, height=430, legend="none", bottom_margin=78)
        fig.update_layout(margin=dict(l=82, r=52, t=30, b=90), coloraxis_colorbar=dict(len=0.78, thickness=12))
        plot(fig, "r95_decision_corr")


def drift_temporal_v2_page():
    hero(
        "TEMPORAL SPECIALIST VIEW",
        "Drift & Temporal Lab",
        "Deep-dive into rolling performance, drift detections, recovery and window-level behavior after the benchmark has run.",
        pills=phase_pills(),
    )
    results = result_dicts()
    if not results:
        empty_state("Run evidence required", "Run a benchmark first.")
        return

    st.markdown(
        """
        <div class="r9-callout">
          <b>Event semantics:</b> red markers are drift detections. Green markers are shown only when the framework explicitly records a refit/retrain event. Continuous online adaptation is not mislabeled as a refit.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        plot(temporal_metric_figure(results, "rolling_accuracy", "Rolling accuracy", "Rolling accuracy"), "r95_drift_acc")
    with c2:
        plot(temporal_metric_figure(results, "rolling_f1_macro", "Rolling Macro-F1", "Macro-F1"), "r95_drift_f1")

    c3, c4 = st.columns(2)
    with c3:
        plot(temporal_metric_figure(results, "mean_prediction_latency_ms", "Prediction latency", "Milliseconds"), "r95_drift_latency")
    with c4:
        plot(temporal_metric_figure(results, "throughput_samples_sec", "Throughput", "Samples / second"), "r95_drift_throughput")

    rows = []
    heat_rows = []
    for r in results:
        ds = r.get("drift_summary") or {}
        rows.append({
            "Framework": r.get("framework"),
            "Drift events": len(r.get("drift_events") or []),
            "Recovery-applicable": ds.get("n_recovery_applicable"),
            "Recovered": ds.get("n_recovered"),
            "Recovery rate": ds.get("recovery_rate"),
            "Median recovery samples": ds.get("median_recovery_samples"),
            "Mean accuracy drop": ds.get("mean_accuracy_drop"),
            "Max accuracy drop": ds.get("max_accuracy_drop"),
        })
        for pt in r.get("points") or []:
            if pt.get("sample") is not None and pt.get("rolling_accuracy") is not None:
                heat_rows.append({"Framework": r.get("framework"), "Sample": pt.get("sample"), "Rolling accuracy": pt.get("rolling_accuracy")})

    section("Recovery summary", "Missing recovery evidence remains missing rather than being converted to zero.")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if heat_rows:
        heat = pd.DataFrame(heat_rows).pivot_table(index="Framework", columns="Sample", values="Rolling accuracy", aggfunc="last")
        fig = px.imshow(
            heat,
            text_auto=".2f",
            zmin=max(0.0, float(np.nanmin(heat.to_numpy())) - 0.05),
            zmax=min(1.0, float(np.nanmax(heat.to_numpy())) + 0.02),
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Window-level rolling accuracy",
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=20))
        plot(fig, "r95_drift_heat")


def fairness_v2_page():
    hero(
        "STREAMING FAIRNESS",
        "Fairness Lab",
        "Compare complementary disparity criteria and inspect how fairness changes over stream windows and drift events.",
        pills=phase_pills(),
    )
    state = _state()
    results = result_dicts()
    if not results:
        empty_state("Run evidence required", "Run a benchmark first.")
        return
    if not state.get("sensitive"):
        st.warning("No sensitive attribute was confirmed for this run. Re-run from Run Studio with a sensitive attribute to compute fairness evidence.")
        return

    metric_map = {
        "Demographic parity": "dp_diff",
        "Equal opportunity": "equal_opportunity_diff",
        "Equalized odds": "equalized_odds_gap",
        "Predictive parity": "predictive_parity_diff",
        "Error-rate parity": "error_rate_gap",
    }
    rows = []
    for r in results:
        f = r.get("fairness") or {}
        row = {"Framework": r.get("framework"), "Status": f.get("status"), "Window N": f.get("window_n")}
        for label, key in metric_map.items():
            row[label] = f.get(key)
        rows.append(row)
    fair = pd.DataFrame(rows)
    numeric = fair[list(metric_map)].apply(pd.to_numeric, errors="coerce")
    fair["Mean gap"] = numeric.mean(axis=1, skipna=True)
    fair["Worst gap"] = numeric.max(axis=1, skipna=True)

    cards = st.columns(4)
    best = fair.sort_values("Mean gap").iloc[0] if fair["Mean gap"].notna().any() else None
    cards[0].metric("Lowest mean disparity", best["Framework"] if best is not None else "N/A", fmt(best["Mean gap"], 3) if best is not None else None)
    cards[1].metric("Sensitive attribute", str(state.get("sensitive")))
    cards[2].metric("Positive label", str(state.get("positive_label")))
    cards[3].metric("Available gap values", f"{int(numeric.notna().sum().sum())}/{numeric.size}")

    section("Aggregate fairness profile", "Heatmap gives the complete criterion matrix; the adjacent chart summarizes mean and worst disparity without a crowded legend.")
    left, right = st.columns([1.28, 1])
    with left:
        heat = fair.set_index("Framework")[list(metric_map)]
        arr = heat.to_numpy(dtype=float)
        vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
        fig = px.imshow(
            heat, text_auto=".3f", zmin=0, zmax=max(0.05, vmax), aspect="auto",
            color_continuous_scale="YlOrRd", title="Disparity matrix · lower is better"
        )
        apply_research_layout(fig, height=410, legend="none", title="Disparity matrix · lower is better", bottom_margin=64)
        fig.update_layout(margin=dict(l=90, r=50, t=54, b=78), coloraxis_colorbar=dict(len=0.78, thickness=12, title="Gap"))
        plot(fig, "r96_fair_heat")
    with right:
        summary = fair[["Framework", "Mean gap", "Worst gap"]].melt(id_vars="Framework", var_name="Summary", value_name="Gap").dropna()
        fig = px.bar(
            summary, x="Gap", y="Framework", color="Summary", barmode="group", orientation="h",
            color_discrete_map={"Mean gap": "#2563eb", "Worst gap": "#ef4444"},
            title="Mean vs worst observed disparity"
        )
        apply_research_layout(fig, height=410, legend="bottom", title="Mean vs worst observed disparity", bottom_margin=92)
        fig.update_layout(margin=dict(l=96, r=24, t=54, b=94), legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center"))
        plot(fig, "r96_fair_summary")

    section("Temporal fairness", "Select one criterion. A separate event strip shows drift and explicitly recorded refit/retrain events without covering the fairness trajectories.")
    criterion = st.selectbox("Temporal fairness criterion", list(FAIRNESS_POINT_KEYS), key="r95_fair_metric")
    key = FAIRNESS_POINT_KEYS[criterion]
    plot(temporal_metric_figure(results, key, f"{criterion} over stream windows", "Gap ↓"), "r95_fair_temporal")

    temp_rows = []
    for r in results:
        vals = [pt.get(key) for pt in (r.get("points") or []) if pt.get(key) is not None]
        if vals:
            temp_rows.append({
                "Framework": r.get("framework"),
                "Mean gap": float(np.mean(vals)),
                "P95 gap": float(np.quantile(vals, 0.95)),
                "Worst-window gap": float(np.max(vals)),
                "Temporal volatility": float(np.std(vals)),
                "Windows": len(vals),
            })
    if temp_rows:
        tdf = pd.DataFrame(temp_rows).sort_values("Worst-window gap")
        c1, c2 = st.columns([1.25, 1])
        with c1:
            st.dataframe(tdf, use_container_width=True, hide_index=True)
        with c2:
            fig = px.bar(
                tdf.sort_values("Worst-window gap", ascending=True),
                x="Worst-window gap", y="Framework", orientation="h", color="Framework",
                color_discrete_map=FRAMEWORK_COLORS,
                title="Worst observed fairness window · lower is better",
                text_auto=".3f",
            )
            apply_research_layout(fig, height=340, legend="none", title="Worst observed fairness window · lower is better", bottom_margin=52)
            fig.update_layout(margin=dict(l=96, r=30, t=54, b=54))
            plot(fig, "r95_fair_worst")

    st.info("Fairness metrics are complementary criteria, not interchangeable definitions of fairness. Report the sensitive attribute, positive label, group support, temporal aggregation and worst-window behavior.")

def explainability_v2_page():
    hero(
        "EXPLANATION DIAGNOSTICS",
        "Explainability Lab",
        "Separate model performance from explanation availability while preserving model-level, hyperparameter-level and system-level explanation context.",
        pills=phase_pills(),
    )
    results = result_dicts()
    if not results:
        empty_state("Run evidence required", "Run a benchmark first.")
        return

    use_llm, model, status = _ollama_controls("r96_xai")
    st.caption(
        "Local Ollama can summarize only the structured benchmark/XAI evidence shown on this page. Raw dataset rows are not sent."
    )

    coverage = []
    for r in results:
        e = r.get("explainability") or {}
        coverage.append({
            "Framework": r.get("framework"),
            "XAI status": e.get("status"),
            "Method used": e.get("method"),
            "Fidelity": e.get("fidelity"),
            "Stability": e.get("stability"),
            "Consistency": e.get("consistency"),
            "Replay warning": bool(e.get("replay_warning")),
        })
    cdf = pd.DataFrame(coverage)
    section("Explanation availability", "A framework can have a valid predictive benchmark result even when its XAI signal is unavailable.")
    st.dataframe(cdf, use_container_width=True, hide_index=True)

    fw = st.selectbox("Framework", [r.get("framework") for r in results], key="r95_xai_fw")
    r = next(x for x in results if x.get("framework") == fw)
    e = r.get("explainability") or {}
    pred = r.get("prediction_diagnostics") or {}
    params = r.get("parameters") or {}

    model_tab, hyper_tab, system_tab = st.tabs([
        "Model-level explanations", "Hyperparameter-level context", "System-level explainability"
    ])

    with model_tab:
        if e.get("status") != "ok":
            st.warning(
                "{} has a valid benchmark result, but AwareML did not obtain a trustworthy non-degenerate explanation signal for this run.".format(fw)
            )
            st.markdown(
                """
                <div class="r9-callout">
                  This is an <b>XAI availability diagnostic</b>, not a model failure. The scientifically safer behavior is to preserve the performance result and mark the explanation as unavailable rather than drawing an all-zero importance chart.
                </div>
                """,
                unsafe_allow_html=True,
            )
            attempts = e.get("method_attempts") or []
            if attempts:
                st.dataframe(pd.DataFrame(attempts), use_container_width=True, hide_index=True)
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Prediction coverage", fmt(pred.get("prediction_coverage"), 3))
            d2.metric("Unique predicted labels", str(pred.get("unique_predicted_labels", "N/A")))
            d3.metric("Majority prediction fraction", fmt(pred.get("majority_prediction_fraction"), 3))
            d4.metric("Near-constant prediction", str(pred.get("near_constant_prediction", "N/A")))
            st.info("For AutoStreamML, this should be diagnosed as an adapter/method-availability issue before Phase 10; the UI should not manufacture a non-zero feature importance.")
        else:
            cards = st.columns(5)
            cards[0].metric("Stability", fmt(e.get("stability"), 3))
            cards[1].metric("Consistency", fmt(e.get("consistency"), 3))
            cards[2].metric("Sensitivity ↓", fmt(e.get("sensitivity"), 3))
            cards[3].metric("Deletion fidelity", fmt(e.get("fidelity"), 3))
            cards[4].metric("Sparsity", fmt(e.get("sparsity"), 3))

            imp = pd.DataFrame(e.get("feature_importance") or [])
            if not imp.empty:
                imp = imp.head(20).sort_values("importance")
                fig = px.bar(imp, x="importance", y="feature", orientation="h", title=f"Recent-window feature importance · {fw}")
                fig.update_traces(marker_color=FRAMEWORK_COLORS.get(fw, "#2563eb"))
                apply_research_layout(fig, height=480, legend="none", title=f"Recent-window feature importance · {fw}", bottom_margin=52)
                fig.update_layout(margin=dict(l=130, r=30, t=54, b=54))
                plot(fig, "r96_xai_importance")

        with st.expander("XAI method audit trail", expanded=False):
            attempts = e.get("method_attempts") or []
            if attempts:
                st.dataframe(pd.DataFrame(attempts), use_container_width=True, hide_index=True)
            st.json(e.get("method_metadata") or {}, expanded=False)

        if use_llm:
            chat = GroundedChat(model=model)
            facts = chat.build_facts(results, _state().get("ranking"))
            question = (
                f"Summarize the explanation evidence for {fw}. Distinguish predictive performance from XAI availability, "
                "and do not claim a feature is important unless the structured evidence supports it."
            )
            answer, meta = chat.answer(question, facts, use_llm=True)
            st.markdown("**Grounded Ollama summary**")
            st.write(answer)
            st.caption("Source: {} · model: {}".format(meta.get("source"), meta.get("model")))

    with hyper_tab:
        st.markdown(
            "**Purpose:** show the model/backend configuration that produced this run. This is context for reproducibility, not a causal claim that each parameter caused the observed outcome."
        )
        if params:
            rows = [{"Parameter": k, "Value": v} for k, v in sorted(params.items()) if not isinstance(v, (dict, list, tuple))]
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            nested = {k: v for k, v in params.items() if isinstance(v, (dict, list, tuple))}
            if nested:
                with st.expander("Nested parameter evidence", expanded=False):
                    st.json(nested, expanded=False)
        else:
            st.info("No hyperparameter/context dictionary was recorded for this framework run.")

        context = {
            "Framework": r.get("framework"),
            "Backend": r.get("backend"),
            "Samples": r.get("samples"),
            "Runtime (s)": r.get("runtime_sec"),
            "Accuracy": r.get("accuracy"),
            "Macro-F1": r.get("f1_macro"),
            "Drift events": len(r.get("drift_events") or []),
        }
        st.dataframe(pd.DataFrame([context]), use_container_width=True, hide_index=True)

    with system_tab:
        st.markdown(
            "**Purpose:** compare explanation availability and quality across the five frameworks rather than interpreting one model in isolation."
        )
        st.dataframe(cdf, use_container_width=True, hide_index=True)
        metrics = cdf[["Framework", "Fidelity", "Stability", "Consistency"]].melt(
            id_vars="Framework", var_name="XAI metric", value_name="Score"
        ).dropna()
        if not metrics.empty:
            fig = px.bar(
                metrics, x="Score", y="Framework", color="XAI metric", barmode="group", orientation="h",
                color_discrete_map={"Fidelity": "#2563eb", "Stability": "#10b981", "Consistency": "#8b5cf6"},
                title="Cross-framework XAI quality · available methods only",
            )
            apply_research_layout(fig, height=410, legend="bottom", title="Cross-framework XAI quality · available methods only", bottom_margin=94)
            fig.update_layout(margin=dict(l=105, r=24, t=54, b=96))
            plot(fig, "r96_xai_system")
        unavailable = cdf[cdf["XAI status"] != "ok"]["Framework"].tolist()
        if unavailable:
            st.warning("XAI unavailable/degenerate for: {}. These are availability diagnostics, not failed benchmark runs.".format(", ".join(unavailable)))

def sustainability_v2_page():
    hero(
        "MEASURED RESOURCE EVIDENCE",
        "Sustainability Lab",
        "Compare measured runtime, energy and CO₂ without converting missing measurements into zeros.",
        pills=phase_pills(),
    )
    results = result_dicts()
    if not results:
        empty_state("Run evidence required", "Run a benchmark with CodeCarbon measurement enabled.")
        return

    rows = []
    for r in results:
        s = r.get("sustainability") or {}
        rows.append({
            "Framework": r.get("framework"),
            "Status": s.get("status"),
            "Energy kWh": r.get("energy_kwh"),
            "CO₂ kg": r.get("co2_kg"),
            "Runtime s": r.get("runtime_sec"),
            "Samples": r.get("samples"),
            "Measurement duration s": s.get("duration_sec"),
            "Backend": s.get("measurement_backend"),
            "CodeCarbon": s.get("codecarbon_version"),
            "CPU": s.get("cpu"),
            "GPU": s.get("gpu"),
            "RAM GB": s.get("ram_gb"),
        })
    sdf = pd.DataFrame(rows)

    cards = st.columns(4)
    cards[0].metric("Energy measured", f"{int(sdf['Energy kWh'].notna().sum())}/{len(sdf)}")
    cards[1].metric("CO₂ measured", f"{int(sdf['CO₂ kg'].notna().sum())}/{len(sdf)}")
    cards[2].metric("Total measured energy", fmt(pd.to_numeric(sdf["Energy kWh"], errors="coerce").sum(min_count=1), 6, " kWh"))
    cards[3].metric("Total measured CO₂", fmt(pd.to_numeric(sdf["CO₂ kg"], errors="coerce").sum(min_count=1), 6, " kg"))

    st.dataframe(sdf, use_container_width=True, hide_index=True)
    measured = sdf.dropna(subset=["Energy kWh", "CO₂ kg"], how="all")
    if measured.empty:
        st.warning("No measured energy/CO₂ values are available for this run.")
        return

    c1, c2 = st.columns(2)
    with c1:
        e = measured.dropna(subset=["Energy kWh"]).sort_values("Energy kWh", ascending=True)
        fig = go.Figure(go.Bar(
            x=e["Energy kWh"], y=e["Framework"], orientation="h",
            marker_color=_colors(e),
            text=[f"{v:.3e}" for v in e["Energy kWh"]], textposition="outside",
            hovertemplate="%{y}<br>Energy %{x:.6g} kWh<extra></extra>",
        ))
        apply_research_layout(fig, height=350, legend="none", title="Measured energy · lower is better", bottom_margin=52)
        fig.update_layout(margin=dict(l=102, r=70, t=54, b=54))
        plot(fig, "r95_sustain_energy")
    with c2:
        c = measured.dropna(subset=["CO₂ kg"]).sort_values("CO₂ kg", ascending=True)
        fig = go.Figure(go.Bar(
            x=c["CO₂ kg"], y=c["Framework"], orientation="h",
            marker_color=_colors(c),
            text=[f"{v:.3e}" for v in c["CO₂ kg"]], textposition="outside",
            hovertemplate="%{y}<br>CO₂ %{x:.6g} kg<extra></extra>",
        ))
        apply_research_layout(fig, height=350, legend="none", title="Measured CO₂ · lower is better", bottom_margin=52)
        fig.update_layout(margin=dict(l=102, r=70, t=54, b=54))
        plot(fig, "r95_sustain_co2")

    section("Efficiency relationships", "These plots are descriptive across the current run and should not be interpreted causally.")
    left, right = st.columns(2)
    with left:
        both = measured.dropna(subset=["Runtime s", "Energy kWh"])
        fig = px.scatter(
            both, x="Runtime s", y="Energy kWh", color="Framework", size="Samples",
            color_discrete_map=FRAMEWORK_COLORS, hover_data=["CO₂ kg"], title="Runtime vs measured energy",
        )
        apply_research_layout(fig, height=400, legend="bottom", title="Runtime vs measured energy", bottom_margin=92)
        fig.update_layout(margin=dict(l=58, r=24, t=54, b=94))
        plot(fig, "r95_sustain_runtime_energy")
    with right:
        both = measured.dropna(subset=["Energy kWh", "CO₂ kg"])
        corr = both[["Energy kWh", "CO₂ kg"]].corr(method="spearman").iloc[0, 1] if len(both) >= 2 else np.nan
        fig = px.scatter(
            both, x="Energy kWh", y="CO₂ kg", color="Framework", text="Framework",
            color_discrete_map=FRAMEWORK_COLORS,
            title="Energy vs CO₂ · Spearman ρ={}".format("N/A" if not np.isfinite(corr) else f"{corr:.2f}"),
        )
        fig.update_traces(textposition="top center")
        apply_research_layout(fig, height=400, legend="none", title="Energy vs CO₂ · Spearman ρ={}".format("N/A" if not np.isfinite(corr) else f"{corr:.2f}"), bottom_margin=58)
        fig.update_layout(margin=dict(l=62, r=50, t=54, b=60))
        plot(fig, "r95_sustain_energy_co2")
        st.caption("ρ≈1 means energy and CO₂ rank frameworks almost identically; weighting both heavily can double-count the same efficiency signal.")


def trust_calibration_v2_page():
    hero(
        "HUMAN STUDY",
        "Trust Calibration",
        "Measure whether user trust follows recommendation reliability rather than explanation fluency alone.",
        pills=phase_pills(),
    )
    ranking = _state().get("ranking")
    if not ranking:
        empty_state("Observed ranking required", "Open Decision Lab first so the trust study has an operational ranking to manipulate.")
        return

    st.markdown(
        """
        <div class="r9-callout">
          <b>What this lab studies:</b> participants see recommendations with matched explanation style but different reliability conditions.
          The research question is whether trust appropriately increases for correct recommendations and decreases for weak/wrong ones.
          These manipulations are <b>study stimuli only</b> and never replace the operational recommendation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "r95_trust_case" not in st.session_state:
        st.session_state.r95_trust_case = None

    left, right = st.columns([1, 1.4])
    with left:
        condition = st.selectbox(
            "Reliability condition",
            ["Randomized", "correct", "weak", "wrong"],
            key="r95_trust_condition",
            help="For an actual study, Randomized is preferred. Manual conditions are useful for researcher testing.",
        )
        if st.button("Generate matched trial", type="primary", use_container_width=True, key="r95_trust_generate"):
            study = TrustCalibrationStudy(seed=int(time.time()) % 100000)
            st.session_state.r95_trust_case = study.build_case(ranking, None if condition == "Randomized" else condition)

    case = st.session_state.r95_trust_case
    with right:
        if case:
            st.markdown("**Recommendation shown to participant**")
            st.markdown(case.explanation)
            st.caption("Explanation wording and structure are intentionally held constant across reliability conditions.")
        else:
            st.info("Generate a trial to preview the participant stimulus.")

    if not case:
        return

    section("Participant response", "These items measure trust, perceived correctness, confidence and acceptance separately.")
    c1, c2, c3 = st.columns(3)
    with c1:
        trust = st.slider("Trust in recommendation", 1, 7, 4, key="r95_trust_score")
    with c2:
        correctness = st.slider("Perceived correctness", 1, 7, 4, key="r95_trust_correct")
    with c3:
        confidence = st.slider("Decision confidence", 1, 7, 4, key="r95_trust_conf")
    accept = st.radio("Would you accept this recommendation?", ["Yes", "No"], horizontal=True, key="r95_trust_accept")
    session = st.text_input("Participant/session code", value="pilot-session", key="r95_trust_session")

    if st.button("Save trial response", key="r95_trust_save"):
        StudyStore().log("trust", session, "trial_response", {
            **case.to_dict(),
            "trust": trust,
            "perceived_correctness": correctness,
            "decision_confidence": confidence,
            "accepted": accept == "Yes",
        })
        st.success("Response stored with the study audit trail.")

    with st.expander("Experimenter-only condition check", expanded=False):
        st.json({
            "condition": case.condition,
            "oracle": case.oracle_framework,
            "shown": case.shown_framework,
            "reliability": case.reliability,
        })


def information_seeking_v2_page():
    hero(
        "HUMAN STUDY",
        "Information-Seeking Lab",
        "Observe how users interrogate the evidence after a recommendation instead of assuming one explanation is sufficient.",
        pills=phase_pills(),
    )
    results = result_dicts()
    if not results:
        empty_state("Run evidence required", "Run a benchmark first so the conversation can be grounded in actual evidence.")
        return

    st.markdown(
        """
        <div class="r9-callout">
          <b>What this lab measures:</b> after seeing a recommendation, does the user ask for evidence, challenge the result,
          compare alternatives, ask for clarification, or stop probing? The system logs these follow-up behaviors for HCAI analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    use_llm, model, status = _ollama_controls("r95_info")
    if not _state().get("chat_session_id"):
        _state()["chat_session_id"] = str(uuid.uuid4())
    if "r95_chat_log" not in st.session_state:
        st.session_state.r95_chat_log = []

    chat = GroundedChat(model=model)
    facts = chat.build_facts(results, _state().get("ranking"))

    left, right = st.columns([1.6, 0.8])
    with left:
        st.markdown("**Grounded evidence conversation**")
        for item in st.session_state.r95_chat_log:
            with st.chat_message(item["role"]):
                st.markdown(item["text"])
        q = st.chat_input("Ask why a framework was recommended, compare alternatives, question fairness, drift, XAI or sustainability…")
        if q:
            st.session_state.r95_chat_log.append({"role": "user", "text": q})
            category = classify_follow_up(q)
            started = time.perf_counter()
            answer, meta = chat.answer(q, facts, use_llm=use_llm)
            latency = time.perf_counter() - started
            st.session_state.r95_chat_log.append({"role": "assistant", "text": answer})
            StudyStore().log("information_seeking", _state()["chat_session_id"], "chat_turn", {
                "question": q,
                "category": category,
                "answer_source": meta.get("source"),
                "model": meta.get("model"),
                "response_time_sec": latency,
                "turn": len(st.session_state.r95_chat_log) // 2,
            })
            st.rerun()

    with right:
        st.markdown("**Behavior summary**")
        user_turns = [x for x in st.session_state.r95_chat_log if x["role"] == "user"]
        cats = [classify_follow_up(x["text"]) for x in user_turns]
        st.metric("Follow-up questions", len(user_turns))
        if cats:
            counts = pd.Series(cats).value_counts().rename_axis("Behavior").reset_index(name="Count")
            fig = px.bar(counts, x="Count", y="Behavior", orientation="h", text_auto=True, title="Observed follow-up behavior")
            apply_research_layout(fig, height=310, legend="none", title="Observed follow-up behavior", bottom_margin=48)
            fig.update_layout(margin=dict(l=105, r=24, t=54, b=50))
            plot(fig, "r95_info_behavior")
        else:
            st.caption("No follow-up behavior has been recorded yet.")

        with st.expander("Think-aloud prompts", expanded=False):
            for prompt in THINK_ALOUD_PROMPTS:
                st.markdown(f"- {prompt}")

        accepted = st.checkbox("Participant accepted the first answer without further probing", value=False, key="r95_info_accepted")
        if st.button("Save session endpoint", key="r95_info_save"):
            StudyStore().log("information_seeking", _state()["chat_session_id"], "session_end", {
                "first_answer_accepted": accepted,
                "follow_up_depth": len(user_turns),
            })
            st.success("Session endpoint stored.")
