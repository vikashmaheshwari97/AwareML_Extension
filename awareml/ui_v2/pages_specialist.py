from __future__ import annotations

import os
import time
import uuid

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from awareml.engine.pareto import METRIC_SPECS
from awareml.llm import GroundedChat, OllamaClient, ollama_status
from awareml.recommender import RecommendationService
from awareml.studies import StudyStore, TrustCalibrationStudy, classify_follow_up
from awareml.studies.information_seeking import THINK_ALOUD_PROMPTS
from awareml.types import ObjectiveWeights

from .components import hero, section, empty_state
from .page_utils import fmt, phase_pills, plot, results_frame
from .plots import FRAMEWORK_COLORS, OBJECTIVE_COLORS, apply_research_layout, temporal_metric_figure
from .state import ensure_research_state, result_dicts
from .pre14_usability import (
    render_decision_lab_explanation,
    render_fairness_validity_panel,
)
from .study_labs_v3 import (
    information_seeking_research_page,
    trust_calibration_research_page,
)
from .phase14_integrated_sections import (
    render_phase14_fairness_details,
    render_phase14_sustainability_details,
)
from .pareto_frontier import render_sustainability_accuracy_pareto
from .advanced_visuals import (
    temporal_fairness_envelope_figure,
    temporal_fairness_robustness_figure,
)
from .research_visuals_v5 import (
    utility_decomposition_figure,
    desirability_profile_figure,
    correlation_arc_figure,
    correlation_pair_ranking,
    fairness_window_evidence_figure,
    fairness_rank_bump_figure,
    fairness_robustness_frontier,
    system_xai_ranked_panels,
    system_xai_rank_bump,
    sustainability_resource_scoreboard,
    sustainability_rank_bump,
    sustainability_efficiency_frontier,
    carbon_intensity_consistency_figure,
)
from .fairness_dynamics import (
    fairness_series_frame,
    replay_fairness_audit,
    fairness_debt_summary,
    fairness_dynamics_figure,
)
from awareml.llm.xai_grounded_summary import grounded_xai_summary


FAIRNESS_OPTIONS = {
    "Composite (common available gaps)": "composite",
    "Demographic parity": "demographic_parity",
    "Equal opportunity": "equal_opportunity",
    "Equalized odds": "equalized_odds",
    "Error-rate parity": "error_rate",
}

FAIRNESS_POINT_KEYS = {
    "Demographic parity": "dp_diff",
    "Equal opportunity": "equal_opportunity_diff",
    "Equalized odds": "equalized_odds_gap",
    "Error-rate parity": "error_rate_gap",
    "Group Brier-score gap": "group_brier_score_gap",
    "Group ECE gap": "group_ece_gap",
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


def _resolved_ollama_model() -> tuple[str, dict]:
    """Resolve a local model without exposing a separate XAI toggle.

    Explainability uses the same explainer interaction pattern as Fairness Lab:
    a button requests an explanation; Ollama is used when available and the
    deterministic grounded fallback is used otherwise.
    """
    status = ollama_status()
    models = status.get("models") or []
    preferred = _state().get("ollama_model") or status.get("resolved_model") or "llama3:8b"
    if preferred not in models and "llama3:8b" in models:
        preferred = "llama3:8b"
    elif preferred not in models and models:
        preferred = models[0]
    _state()["ollama_model"] = preferred
    return str(preferred), status


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

    st.markdown(
        """
        <div class="r9-callout">
          <b>OBSERVED POST-RUN RECOMMENDATION</b><br>
          <b>Evidence source:</b> outcomes actually measured in the current five-framework benchmark.<br>
          <b>Preference source:</b> the six post-run sliders on this page.<br>
          <b>Ranking engine:</b> observed multi-objective utility + near-Pareto analysis.<br>
          <b>Framework execution:</b> completed; this page ranks measured evidence rather than pre-run predictions.
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
        epsilon = st.slider("Near-Pareto ε (journal default 0.05)", 0.0, 0.20, 0.05, 0.01, key="r95_decision_eps")
        if abs(float(epsilon) - 0.05) > 1e-12:
            st.caption("Sensitivity view: the journal-facing canonical near-Pareto result uses ε=0.05.")

    weights = ObjectiveWeights(**vals)
    service = RecommendationService(epsilon=epsilon, fairness_metric=fair_metric)
    frame, recs, corr, warnings = service.rank(_run_objects(), weights)
    _state()["ranking"] = frame.to_dict(orient="records")
    _state()["fairness_metric"] = fair_metric

    top = recs[0]
    top_row = frame.iloc[0]
    cards = st.columns(5)
    cards[0].metric("Post-run recommended framework", top.framework)
    cards[1].metric("Observed utility", fmt(top.utility, 3))
    cards[2].metric("Observed accuracy", fmt(top_row.get("accuracy"), 3))
    fairness_score_label = (
        "Composite fairness score ↑ (1 - mean common gap)"
        if fair_metric == "composite"
        else "{} score ↑ (1 - gap)".format(fair_label)
    )
    cards[3].metric(fairness_score_label, fmt(top_row.get("fairness_score"), 3))
    cards[4].metric("Near-Pareto", "Yes" if top.near_pareto else "No")

    if fair_metric == "composite":
        common = str(top_row.get("fairness_criteria_used") or "").strip()
        count = top_row.get("fairness_criteria_count")
        total = top_row.get("fairness_criteria_total")
        if common:
            st.caption(
                "Composite fairness = 1 - mean disparity over the same criteria for "
                "every framework in this run: {} ({} of {}). Missing criteria are "
                "reported as unavailable and are not treated as zero.".format(
                    common, int(count), int(total)
                )
            )

    st.success(
        "{} is ranked #1 because, under the current weights, it has the highest utility calculated from the observed run evidence.".format(top.framework)
    )
    render_decision_lab_explanation(
        frame, top, weights, fair_metric, fair_label, _state()
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

    section(
        "Decision fingerprint",
        "A reviewer-facing decomposition of the observed decision: exact weighted utility is shown as stacked contributions, while the adjacent profile shows each framework's direction-aligned run-relative desirability before weighting. No bubble-size encoding is used.",
    )
    left, right = st.columns([1.15, 1])
    with left:
        fig = utility_decomposition_figure(contrib, frame)
        apply_research_layout(
            fig, height=455, legend="bottom",
            title="Weighted utility decomposition · exact contribution by objective",
            bottom_margin=104,
        )
        plot(fig, "r110_decision_utility_decomposition")
        st.caption(
            "Each horizontal bar sums the exact objective contributions used by Decision Lab. Segment length—not area—encodes contribution, so the ranking is quantitatively readable."
        )
    with right:
        fig = desirability_profile_figure(norm)
        apply_research_layout(
            fig, height=455, legend="bottom",
            title="Objective desirability profiles · direction aligned",
            bottom_margin=104,
        )
        plot(fig, "r110_decision_desirability_profile")
        st.caption(
            "Runtime, Energy and CO₂ are direction-flipped only for this 0–1 visualization so higher always means more desirable. The star on a profile marks that framework's strongest relative objective; no aggregate score is created here."
        )

    section(
        "Interactive Pareto Frontier",
        "Accuracy versus a combined run-relative Energy + CO₂ burden. The 2D frontier is visual; the canonical Near-Pareto flag remains the multi-objective ε-dominance result.",
    )
    render_sustainability_accuracy_pareto(frame, key="r95_decision_sustainability_pareto")

    section("Observed ranking table", "This is the auditable post-run ranking used by this page.")
    fairness_table_label = (
        "Composite fairness score ↑"
        if fair_metric == "composite"
        else "{} score ↑".format(fair_label)
    )
    display = frame.rename(columns={
        "framework": "Framework",
        "rank": "Rank",
        "utility": "Utility",
        "near_pareto": "Near-Pareto",
        "accuracy": "Accuracy",
        "runtime_sec": "Runtime (s)",
        "energy_kwh": "Energy (kWh)",
        "co2_kg": "CO₂ (kg)",
        "fairness_score": fairness_table_label,
        "interpretability_score": "Interpretability score",
    })
    display["ε"] = pd.to_numeric(frame.get("pareto_epsilon"), errors="coerce")
    visible = [
        "Rank", "Framework", "Utility", "Near-Pareto", "ε",
        "Accuracy", "Runtime (s)", "Energy (kWh)", "CO₂ (kg)",
        fairness_table_label, "Interpretability score",
    ]
    st.dataframe(
        display[[column for column in visible if column in display.columns]],
        use_container_width=True,
        hide_index=True,
    )

    section(
        "Objective relationships",
        "Objective dependence is shown in two complementary views: an arc diagram exposes structure and sign, while the ranked pair panel preserves exact Spearman values. This is descriptive evidence, not a causal graph.",
    )
    if corr is not None and not corr.empty:
        c1, c2 = st.columns([1.18, 1])
        with c1:
            fig = correlation_arc_figure(corr, threshold=0.20)
            apply_research_layout(
                fig, height=470, legend="bottom",
                title="Objective dependence arcs · thickness = |Spearman ρ|",
                bottom_margin=86,
            )
            plot(fig, "r110_decision_corr_arcs")
            st.caption(
                "Positive relationships arc upward; inverse relationships arc downward. Only |ρ| ≥ 0.20 is drawn to avoid a fully connected visual hairball."
            )
        with c2:
            fig = correlation_pair_ranking(corr)
            apply_research_layout(
                fig, height=470, legend="none",
                title="Strongest objective pairs · exact Spearman ρ",
                bottom_margin=62,
            )
            plot(fig, "r110_decision_corr_pairs")
            st.caption(
                "Pairs are ordered by relationship strength. Values near ±1 indicate strong redundancy/opposition and deserve attention when setting simultaneous preference weights."
            )

        strongest = []
        labels = list(corr.columns)
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                try:
                    rho = float(corr.loc[a, b])
                except Exception:
                    continue
                if np.isfinite(rho):
                    strongest.append((abs(rho), rho, str(a), str(b)))
        strongest.sort(reverse=True)
        if strongest:
            _, rho, a, b = strongest[0]
            st.info(
                "Strongest observed relationship: **{} ↔ {}** with Spearman ρ={:.2f}. Very strong objective correlation can cause effective double-weighting when both objectives receive large preference weights.".format(a, b, rho)
            )
        with st.expander("Exact objective-correlation matrix", expanded=False):
            st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)


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


# GROUNDED_FAIRNESS_EXPLAINER_V2
def _render_grounded_fairness_explainer(state, results, fair, common_labels):
    st.markdown("### LLM-Assisted Fairness Explainer")
    st.caption(
        "Explains the current fairness evidence in plain language. The LLM does "
        "not recompute metrics, alter results, or select the framework."
    )

    eligible = fair[
        (~fair["Prediction behavior"].isin(["constant", "near_constant"]))
        & fair["Comparable mean gap"].notna()
    ].copy().sort_values("Comparable mean gap")
    excluded = fair[
        fair["Prediction behavior"].isin(["constant", "near_constant"])
    ]["Framework"].astype(str).tolist()

    best_name = None if eligible.empty else str(eligible.iloc[0]["Framework"])
    best_gap = None if eligible.empty else float(eligible.iloc[0]["Comparable mean gap"])

    calibration = {}
    for result in results:
        values = result.get("fairness") or {}
        calibration[str(result.get("framework"))] = {
            "brier": values.get("group_brier_score_gap"),
            "ece": values.get("group_ece_gap"),
            "status": values.get("calibration_status"),
            "probability_behavior": values.get("probability_behavior_status"),
        }

    def fallback_text():
        parts = []
        if best_name is not None:
            parts.append(
                "Among frameworks with non-degenerate predictions, {} has the lowest "
                "comparable mean disparity ({:.4f}) across {}. Lower means the measured "
                "group rates are closer on average; it does not prove absolute fairness."
                .format(best_name, best_gap, ", ".join(common_labels))
            )
        else:
            parts.append(
                "No framework currently has enough comparable, non-degenerate fairness "
                "evidence for a cross-framework disparity comparison."
            )
        if excluded:
            parts.append(
                "{} is excluded from the lowest-disparity comparison because its predictions "
                "are constant or near-constant. Zero-looking parity gaps can then be mechanical "
                "rather than evidence of a useful fair classifier.".format(", ".join(excluded))
            )
        if best_name and best_name in calibration:
            brier = calibration[best_name].get("brier")
            ece = calibration[best_name].get("ece")
            parts.append(
                "For calibration, {} has Group Brier-score gap {} and Group ECE gap {}. "
                "Lower gaps mean group calibration errors are closer, but these values should "
                "be read together with predictive quality and group support.".format(
                    best_name,
                    "N/A" if brier is None else "{:.4f}".format(float(brier)),
                    "N/A" if ece is None else "{:.4f}".format(float(ece)),
                )
            )
        parts.append(
            "A human reviewer should also inspect the sensitive attribute, positive label, "
            "group support, worst-window behaviour and predictive performance. Different "
            "fairness criteria capture different notions of disparity."
        )
        return "\n\n".join(parts)

    evidence = []
    for _, row in fair.iterrows():
        evidence.append(
            "{} | comparable_mean_gap={} | worst_gap={} | prediction_behavior={} | eligibility={}".format(
                row.get("Framework"), row.get("Comparable mean gap"),
                row.get("Worst available gap"), row.get("Prediction behavior"),
                row.get("Fairness winner eligibility"),
            )
        )
    for name, values in calibration.items():
        evidence.append(
            "{} calibration | brier_gap={} | ece_gap={} | status={} | probability_behavior={}".format(
                name, values.get("brier"), values.get("ece"), values.get("status"),
                values.get("probability_behavior"),
            )
        )

    with st.container(border=True):
        a, b, c = st.columns(3)
        a.markdown("**Evidence source**"); a.caption("Current fairness and calibration records")
        b.markdown("**LLM role**"); b.caption("Explain only · no metric authority")
        c.markdown("**Privacy boundary**"); c.caption("Structured evidence only · no raw rows")

        if st.button("Explain the fairness results", key="fairness_llm_assisted_explain", use_container_width=True):
            answer = None
            source = "deterministic-fallback"
            state.pop("fairness_llm_assisted_error", None)
            try:
                status = ollama_status()
                if status.get("reachable") and "llama3:8b" in (status.get("models") or []):
                    client = OllamaClient(model="llama3:8b", timeout_sec=90.0)
                    prompt = (
                        "You are the AwareML Fairness Explainer. Explain only the structured "
                        "evidence below to a non-expert. Do not recompute metrics or invent causes. "
                        "Use four short parts: what is being compared; which eligible framework "
                        "has the lowest comparable mean disparity and why this is not proof of "
                        "absolute fairness; calibration interpretation using Brier/ECE gaps; and "
                        "human-review cautions. Explain that lower disparity gaps are better and "
                        "why constant/near-constant predictors are excluded. Do not discuss "
                        "predictive parity. Treat missing values as unavailable, never zero.\n\n"
                        "COMMON CRITERIA: {}\nSENSITIVE ATTRIBUTE: {}\nPOSITIVE LABEL: {}\n"
                        "STRUCTURED EVIDENCE:\n{}"
                    ).format(
                        ", ".join(common_labels), state.get("sensitive"),
                        state.get("positive_label"), "\n".join(evidence),
                    )
                    answer, _meta = client.generate_text(prompt)
                    source = "local-ollama"
            except Exception as exc:
                state["fairness_llm_assisted_error"] = "{}: {}".format(type(exc).__name__, exc)
            if not answer:
                answer = fallback_text()
            state["fairness_llm_assisted_explanation"] = {"answer": answer, "source": source}
            st.rerun()

        saved = state.get("fairness_llm_assisted_explanation")
        if isinstance(saved, dict) and saved.get("answer"):
            st.markdown("**Plain-language explanation**")
            st.write(saved["answer"])
            st.caption("Explanation source: {} · structured evidence only.".format(saved.get("source") or "deterministic-fallback"))

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

    render_fairness_validity_panel(state, results)

    metric_map = {
        "Demographic parity": "dp_diff",
        "Equal opportunity": "equal_opportunity_diff",
        "Equalized odds": "equalized_odds_gap",
            "Error-rate parity": "error_rate_gap",
    }
    rows = []
    for r in results:
        f = r.get("fairness") or {}
        row = {
            "Framework": r.get("framework"),
            "Status": f.get("status"),
            "Window N": f.get("window_n"),
            "Calibration status": f.get("calibration_status") or "unavailable",
            "Probability coverage": f.get("probability_coverage"),
            "Group Brier-score gap": f.get("group_brier_score_gap"),
            "Group ECE gap": f.get("group_ece_gap"),
            "Calibration reason": f.get("calibration_reason"),
        }
        for label, key in metric_map.items():
            row[label] = f.get(key)
        rows.append(row)
    fair = pd.DataFrame(rows)
    numeric = fair[list(metric_map)].apply(pd.to_numeric, errors="coerce")
    common_labels = [label for label in metric_map if numeric[label].notna().all()]
    if common_labels:
        fair["Comparable mean gap"] = numeric[common_labels].mean(axis=1)
        fair["Composite fairness score ↑"] = (
            1.0 - fair["Comparable mean gap"]
        ).clip(lower=0.0, upper=1.0)
    else:
        fair["Comparable mean gap"] = np.nan
        fair["Composite fairness score ↑"] = np.nan
    fair["Worst available gap"] = numeric.max(axis=1, skipna=True)
    fair["Metric coverage"] = numeric.notna().sum(axis=1).map(
        lambda count: "{}/{}".format(int(count), len(metric_map))
    )
    fair["Unavailable criteria"] = [
        ", ".join([label for label in metric_map if pd.isna(numeric.loc[idx, label])])
        or "None"
        for idx in fair.index
    ]

    # Phase 14: preserve every raw fairness value, but do not declare a
    # constant/near-constant predictor the fairness winner merely because
    # some parity gaps collapse to zero.
    prediction_status_by_framework = {
        r.get("framework"): (
            (r.get("fairness") or {}).get("prediction_behavior_status")
            or "unavailable"
        )
        for r in results
    }
    fair["Prediction behavior"] = fair["Framework"].map(
        prediction_status_by_framework
    ).fillna("unavailable")
    fair["Fairness winner eligibility"] = fair["Prediction behavior"].map(
        lambda status: (
            "EXCLUDED · degenerate predictions"
            if str(status) in {"constant", "near_constant"}
            else "ELIGIBLE"
        )
    )

    eligible_fair = fair[
        (~fair["Prediction behavior"].isin(["constant", "near_constant"]))
        & fair["Comparable mean gap"].notna()
    ].copy()

    cards = st.columns(4)
    best = (
        eligible_fair.sort_values("Comparable mean gap").iloc[0]
        if not eligible_fair.empty
        else None
    )
    cards[0].metric(
        "Lowest interpretable mean disparity",
        best["Framework"] if best is not None else "N/A",
        fmt(best["Comparable mean gap"], 3) if best is not None else None,
    )

    excluded_frameworks = fair.loc[
        fair["Prediction behavior"].isin(["constant", "near_constant"]),
        "Framework",
    ].astype(str).tolist()
    if excluded_frameworks:
        st.warning(
            "Fairness-winner claim excludes degenerate/near-degenerate "
            "predictors: {}. Their raw fairness values remain visible for "
            "auditability and are not changed.".format(
                ", ".join(excluded_frameworks)
            )
        )
    cards[1].metric("Sensitive attribute", str(state.get("sensitive")))
    cards[2].metric("Positive label", str(state.get("positive_label")))
    cards[3].metric(
        "Available gap values",
        "{}/{}".format(int(numeric.notna().sum().sum()), int(numeric.size)),
    )

    section(
        "Aggregate fairness profile",
        "Paper-style evidence view for the active dataset. Small points are the framework's actual stream-window fairness gaps; the larger diamond is the recorded run-level gap. This adapts the visual grammar of the attached FairStream figure without pretending that one dashboard run contains multiple datasets.",
    )
    fig = fairness_window_evidence_figure(results, metric_map)
    apply_research_layout(
        fig, height=690, legend="bottom",
        title="Fairness evidence across stream windows · window observations + run aggregate",
        bottom_margin=100,
    )
    plot(fig, "r110_fair_window_evidence")
    st.caption(
        "The FairStream paper plots datasets as rows because it aggregates a multi-dataset experiment. This interactive AwareML page is scoped to the currently active dataset, so rows are frameworks and the repeated evidence points are real stream windows. Diamonds are run-level recorded gaps; lower is better."
    )

    left, right = st.columns([1.08, 1])
    with left:
        fig = fairness_rank_bump_figure(fair, list(metric_map))
        apply_research_layout(
            fig, height=440, legend="bottom",
            title="Fairness rank stability across definitions",
            bottom_margin=102,
        )
        plot(fig, "r110_fair_rank_bump")
        st.caption(
            "Rank 1 is the lowest observed disparity within each fairness definition. Crossing lines expose frameworks whose apparent fairness depends strongly on the chosen criterion."
        )
    with right:
        fig = fairness_robustness_frontier(fair, list(metric_map))
        apply_research_layout(
            fig, height=440, legend="none",
            title="Cross-criterion robustness frontier · mean vs worst gap",
            bottom_margin=62,
        )
        plot(fig, "r110_fair_robustness_frontier")
        st.caption(
            "The preferred region is the lower-left: low average disparity and low worst-criterion disparity. Diamond/outlined points are nondominated in this two-dimensional robustness view; this does not replace the individual fairness definitions."
        )

    section(
        "All fairness metrics",
        "Exact current-run values. Lower disparity gaps are better. The composite "
        "score is 1 - the mean of only those criteria available for every framework, "
        "so frameworks are compared using the same denominator.",
    )
    fairness_display = fair[[
        "Framework",
        *list(metric_map),
        "Comparable mean gap",
        "Worst available gap",
        "Composite fairness score ↑",
        "Metric coverage",
        "Unavailable criteria",
        "Prediction behavior",
        "Fairness winner eligibility",
        "Status",
        "Window N",
    ]].copy()
    numeric_display_cols = list(metric_map) + [
        "Comparable mean gap",
        "Worst available gap",
        "Composite fairness score ↑",
    ]
    for column in numeric_display_cols:
        fairness_display[column] = fairness_display[column].map(
            lambda value: "N/A" if pd.isna(value) else "{:.4f}".format(float(value))
        )
    fairness_display["Unavailable reason"] = fairness_display[
        "Unavailable criteria"
    ].map(
        lambda value: (
            "None"
            if value == "None"
            else "Criterion undefined/unavailable in the recorded window; not treated as zero."
        )
    )
    st.dataframe(fairness_display, use_container_width=True, hide_index=True)

    render_phase14_fairness_details(results)

    _render_grounded_fairness_explainer(state, results, fair, common_labels)

    if not common_labels:
        st.warning(
            "No fairness criterion is available for every framework, so a comparable "
            "composite fairness score cannot be computed for this run."
        )
    elif len(common_labels) < len(metric_map):
        st.info(
            "Comparable composite uses: {}. Other criteria remain visible as N/A/available "
            "evidence but are excluded from the composite for every framework.".format(
                ", ".join(common_labels)
            )
        )

    section(
        "Temporal fairness",
        "Select one criterion. Predictive drift and fairness drift are intentionally separated: a model can keep similar accuracy while its group disparity changes over time.",
    )
    criterion = st.selectbox("Temporal fairness criterion", list(FAIRNESS_POINT_KEYS), key="r100_fair_metric")
    key = FAIRNESS_POINT_KEYS[criterion]
    fig = temporal_fairness_envelope_figure(results, key, criterion)
    apply_research_layout(
        fig, height=500, legend="bottom",
        title=f"{criterion} dynamics · observed trajectories + median/IQR envelope",
        bottom_margin=104,
    )
    plot(fig, "r101_fair_temporal_envelope")
    st.caption(
        "The shaded band is the cross-framework interquartile range at each observed stream position; "
        "the dashed line is the cross-framework median. Predictive-drift/refit markers are intentionally omitted "
        "from this fairness plot to avoid conflating predictive drift with fairness change."
    )

    temp_rows = []
    for r in results:
        vals = [pt.get(key) for pt in (r.get("points") or []) if pt.get(key) is not None]
        vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
        if vals:
            temp_rows.append({
                "Framework": r.get("framework"),
                "Mean gap": float(np.mean(vals)),
                "P95 gap": float(np.quantile(vals, 0.95)),
                "Worst-window gap": float(np.max(vals)),
                "Temporal volatility": float(np.std(vals)),
                "Windows": len(vals),
            })
    tdf = pd.DataFrame(temp_rows).sort_values("Worst-window gap") if temp_rows else pd.DataFrame()
    if not tdf.empty:
        c1, c2 = st.columns([1.05, 1])
        with c1:
            st.dataframe(tdf, use_container_width=True, hide_index=True)
        with c2:
            fig = temporal_fairness_robustness_figure(tdf)
            apply_research_layout(
                fig, height=430, legend="bottom",
                title="Temporal robustness profile · mean → P95 → worst + volatility",
                bottom_margin=104,
            )
            plot(fig, "r101_fair_temporal_robustness")
            st.caption(
                "Mean, P95 and worst-window gaps are shown explicitly on one axis; temporal volatility is shown on its own axis. "
                "This replaces the previous bubble-size risk encoding with quantitatively readable positions."
            )

    section(
        "Fairness drift & accumulated debt · FairStream-inspired audit",
        "Replays the already-observed AwareML fairness windows through an adaptive audit envelope and the FairStream debt equation. This is post-run diagnostic evidence only: it does not change AwareML model selection or claim that FairStream controlled this run.",
    )
    series = fairness_series_frame(results, key)
    if series.empty:
        st.info("No window-level values are available for this fairness criterion, so drift/debt auditing is unavailable.")
    else:
        frameworks = series["Framework"].drop_duplicates().astype(str).tolist()
        worst_default = frameworks[0]
        if not tdf.empty:
            worst_default = str(tdf.sort_values("Worst-window gap", ascending=False).iloc[0]["Framework"])
        c0, c1, c2 = st.columns([1.2, 1, 1])
        with c0:
            audit_framework = st.selectbox(
                "Framework for fairness-dynamics deep dive",
                frameworks,
                index=frameworks.index(worst_default) if worst_default in frameworks else 0,
                key="r100_fair_audit_framework",
            )
        with c1:
            audit_upper = st.slider(
                "Initial audit envelope u₀",
                0.02, 0.25, 0.10, 0.01,
                key="r100_fair_audit_upper",
                help="Reference upper fairness gap for the post-run audit. It is not injected into the original benchmark.",
            )
        with c2:
            debt_decay = st.slider(
                "Debt persistence λ",
                0.0, 1.0, 0.90, 0.05,
                key="r100_fair_debt_decay",
                help="Dₜ = λDₜ₋₁ + max(0, cₜ-uₜ). Higher λ remembers violations longer.",
            )
        with st.expander("Fairness-change sensitivity", expanded=False):
            a, b = st.columns(2)
            with a:
                warning_threshold = st.slider("Warning |Δ gap|", 0.005, 0.10, 0.02, 0.005, key="r100_fair_warn")
            with b:
                drift_threshold = st.slider("Fairness-change |Δ gap|", 0.01, 0.20, 0.04, 0.005, key="r100_fair_drift")
            st.caption("Defaults mirror the scale used by FairStream's fairness-drift implementation. These AwareML flags are descriptive threshold events, not a substitute for a full statistical fairness-drift experiment.")
        audit = replay_fairness_audit(
            series,
            initial_upper=float(audit_upper),
            debt_decay=float(debt_decay),
            warning_threshold=float(warning_threshold),
            drift_threshold=float(drift_threshold),
        )
        selected_audit = audit[audit["Framework"].astype(str).eq(str(audit_framework))].sort_values("Sample")
        if not selected_audit.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Windows audited", str(len(selected_audit)))
            m2.metric("Envelope violations", str(int((selected_audit["Violation"] > 0).sum())))
            m3.metric("Fairness-change flags", str(int((selected_audit["Event"] == "fairness_change").sum())))
            m4.metric("Peak fairness debt", fmt(selected_audit["Debt"].max(), 3))
            fig = fairness_dynamics_figure(audit, audit_framework)
            apply_research_layout(fig, height=720, legend="bottom", title="FairStream-inspired fairness dynamics audit · observed AwareML windows", bottom_margin=100)
            plot(fig, "r100_fair_dynamics")
            if len(selected_audit) < 10:
                st.warning(
                    "Only {} fairness windows are available for {} under the current run. The violation/debt replay is still auditable, but this is a short temporal record; run more of the stream before making a statistical fairness-drift claim.".format(
                        len(selected_audit), audit_framework
                    )
                )
        summary = fairness_debt_summary(audit)
        if not summary.empty:
            with st.expander("Cross-framework fairness debt summary", expanded=True):
                st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(
            "Interpretation boundary: FairStream treats fairness drift φₜ and predictive-performance drift ψₜ as separate signals. Therefore zero confirmed predictive drift in Streaming Observatory does not imply temporally constant fairness."
        )

    st.info("Fairness metrics are complementary criteria, not interchangeable definitions of fairness. Report the sensitive attribute, positive label, group support, temporal aggregation and worst-window behavior.")

def _render_grounded_xai_explainer(state, results, framework):
    st.markdown("### LLM-Assisted Explainability Explainer")
    st.caption(
        "Explains the selected framework's current XAI evidence in plain language. "
        "The LLM does not recompute attribution values, change metrics, or infer hidden model reasoning."
    )
    model, status = _resolved_ollama_model()
    with st.container(border=True):
        a, b, c = st.columns(3)
        a.markdown("**Evidence source**"); a.caption("Current benchmark + XAI diagnostics")
        b.markdown("**LLM role**"); b.caption("Explain only · no metric authority")
        c.markdown("**Privacy boundary**"); c.caption("Structured evidence only · no raw rows")
        if status.get("reachable"):
            st.caption("Local Ollama detected · model: {}".format(model))
        else:
            st.caption("Local Ollama is not reachable; the same button will use the deterministic grounded explanation.")
        if st.button("Explain the XAI results", key="xai_llm_assisted_explain", use_container_width=True):
            answer, meta = grounded_xai_summary(results, framework, model=model, use_llm=True)
            state["xai_llm_assisted_explanation"] = {
                "framework": framework,
                "answer": answer,
                "source": meta.get("source"),
                "model": meta.get("model") or model,
                "warning": meta.get("warning"),
            }
            st.rerun()
        saved = state.get("xai_llm_assisted_explanation")
        if isinstance(saved, dict) and saved.get("framework") == framework and saved.get("answer"):
            st.markdown("**Plain-language explanation**")
            st.markdown(saved["answer"])
            st.caption("Explanation source: {} · model: {} · structured evidence only.".format(
                saved.get("source") or "deterministic-fallback", saved.get("model") or model
            ))
            if saved.get("warning"):
                st.caption("Local Ollama fallback reason: {}".format(saved.get("warning")))


def explainability_v2_page(show_header: bool = True):
    if show_header:
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
            "Sensitivity": e.get("sensitivity"),
            "Sparsity": e.get("sparsity"),
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

    _render_grounded_xai_explainer(_state(), results, fw)

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
            st.info("For AutoStreamML, this is treated as an adapter/method-availability issue; AwareML does not manufacture a non-zero feature importance.")
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

        numeric_params = []
        for key, value in sorted(params.items()):
            if isinstance(value, bool):
                continue
            try:
                fv = float(value)
            except Exception:
                continue
            if np.isfinite(fv):
                numeric_params.append({"Parameter": str(key), "Value": fv, "Display magnitude": np.log10(abs(fv) + 1.0)})
        if numeric_params:
            hp = pd.DataFrame(numeric_params).head(24)
            fig = px.bar(
                hp.sort_values("Display magnitude"),
                x="Display magnitude", y="Parameter", orientation="h",
                hover_data={"Value": True, "Display magnitude": ":.3f"},
                title="Recorded numeric configuration · log display scale",
            )
            fig.update_traces(marker_color=FRAMEWORK_COLORS.get(fw, "#2563eb"))
            apply_research_layout(fig, height=max(360, 26 * len(hp)), legend="none", title="Recorded numeric configuration · log display scale", bottom_margin=58)
            fig.update_layout(margin=dict(l=155, r=30, t=54, b=60), xaxis_title="log10(|value| + 1) · display only")
            plot(fig, "r96_xai_hyperparameter_numeric")
            st.caption("This chart visualizes recorded configuration magnitude for readability. It is not hyperparameter importance and makes no causal claim about the observed outcome.")

        outcome_rows = []
        for rr in results:
            outcome_rows.append({
                "Framework": rr.get("framework"),
                "Accuracy": rr.get("accuracy"),
                "Macro-F1": rr.get("f1_macro"),
                "Runtime (s)": rr.get("runtime_sec"),
                "Energy (kWh)": rr.get("energy_kwh"),
                "CO₂ (kg)": rr.get("co2_kg"),
            })
        odf = pd.DataFrame(outcome_rows)
        if not odf.empty:
            fig = px.scatter(
                odf, x="Runtime (s)", y="Accuracy", color="Framework", size="Macro-F1",
                hover_data=["Energy (kWh)", "CO₂ (kg)"],
                color_discrete_map=FRAMEWORK_COLORS,
                title="Configuration outcome context · current run",
            )
            apply_research_layout(fig, height=390, legend="right", title="Configuration outcome context · current run", bottom_margin=58)
            plot(fig, "r96_xai_hyper_outcome")

    with system_tab:
        st.markdown(
            "**Purpose:** compare explanation availability and quality across the five frameworks rather than interpreting one model in isolation."
        )
        valid_system = cdf[cdf["XAI status"].eq("ok")].copy()
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("XAI evidence available", "{}/{}".format(len(valid_system), len(cdf)))
        x2.metric("Distinct XAI methods", str(int(valid_system["Method used"].nunique())) if not valid_system.empty else "0")
        if not valid_system.empty:
            cons = pd.to_numeric(valid_system["Consistency"], errors="coerce")
            sens = pd.to_numeric(valid_system["Sensitivity"], errors="coerce")
            x3.metric("Highest consistency", fmt(cons.max(), 3) if cons.notna().any() else "N/A")
            x4.metric("Lowest sensitivity", fmt(sens.min(), 3) if sens.notna().any() else "N/A")
        else:
            x3.metric("Highest consistency", "N/A")
            x4.metric("Lowest sensitivity", "N/A")
        st.caption("Diagnostics are descriptive. Compare like methods and window sizes; fidelity here is a deletion accuracy drop, not a universal explanation score.")
        if not valid_system.empty:
            fig = system_xai_ranked_panels(valid_system)
            apply_research_layout(
                fig, height=650, legend="none",
                title="System-level XAI diagnostics · exact metric-wise evidence",
                bottom_margin=56,
            )
            plot(fig, "r110_xai_system_ranked_panels")
            st.caption(
                "Each diagnostic has its own quantitative axis. Fidelity, Stability and Consistency are read higher-is-better; Sensitivity is lower-is-better; Sparsity is descriptive and deliberately has no universal winner direction."
            )

            fig = system_xai_rank_bump(valid_system)
            apply_research_layout(
                fig, height=430, legend="bottom",
                title="Directional XAI rank flow · where framework strengths change",
                bottom_margin=104,
            )
            plot(fig, "r110_xai_system_rank_flow")
            st.caption(
                "This rank-flow view uses only diagnostics with an explicit direction. It intentionally excludes Sparsity from ranking because more sparse is not universally better. No overall XAI score is manufactured."
            )

            with st.expander("Metric semantics and exact system values", expanded=False):
                exact = valid_system[["Framework", "Method used", "Fidelity", "Stability", "Consistency", "Sensitivity", "Sparsity", "Replay warning"]].copy()
                st.dataframe(exact, use_container_width=True, hide_index=True)
                st.markdown(
                    """
- **Deletion fidelity ↑:** observed replay accuracy drop after deleting important features; context-dependent, not a universal faithfulness score.
- **Stability ↑:** similar explanations under repeated/replayed evaluation.
- **Consistency ↑:** agreement of repeated explanation estimates under the configured method.
- **Sensitivity ↓:** smaller perturbation-driven explanation change is more stable.
- **Sparsity:** concentration/compactness of the explanation; direction is task-dependent.
                    """
                )

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
            "Physical CPUs": s.get("physical_cpus"),
            "Country": s.get("country_iso"),
            "Region": s.get("region"),
            "Carbon intensity gCO2/kWh": s.get(
                "carbon_intensity_g_per_kwh"
            ),
            "Warm-up s": s.get("warmup_sec"),
            "Repetition": s.get("repetition_id"),
            "Repetitions planned": s.get("repetitions_planned"),
            "Failure reason": s.get("measurement_failure_reason"),
        })
    sdf = pd.DataFrame(rows)

    st.info(
        "records CPU/GPU/RAM, country/region, CodeCarbon version, "
        "carbon intensity, measurement duration, warm-up, repetition metadata "
        "and failure reasons. Missing measurements remain N/A rather than zero."
    )

    cards = st.columns(4)
    cards[0].metric("Energy measured", f"{int(sdf['Energy kWh'].notna().sum())}/{len(sdf)}")
    cards[1].metric("CO₂ measured", f"{int(sdf['CO₂ kg'].notna().sum())}/{len(sdf)}")
    cards[2].metric("Total measured energy", fmt(pd.to_numeric(sdf["Energy kWh"], errors="coerce").sum(min_count=1), 6, " kWh"))
    cards[3].metric("Total measured CO₂", fmt(pd.to_numeric(sdf["CO₂ kg"], errors="coerce").sum(min_count=1), 6, " kg"))

    with st.expander("Measurement provenance & exact values", expanded=False):
        st.dataframe(sdf, use_container_width=True, hide_index=True)
    measured = sdf.dropna(subset=["Energy kWh", "CO₂ kg"], how="all")
    if measured.empty:
        st.warning("No measured energy/CO₂ values are available for this run.")
        return

    section(
        "Measured resource profile",
        "Runtime, Energy and CO₂ are shown on separate quantitative axes so reviewers can compare exact measurements without bubble area, mixed units or hidden normalization.",
    )
    fig = sustainability_resource_scoreboard(sdf)
    apply_research_layout(
        fig, height=440, legend="none",
        title="Measured resource scoreboard · exact lower-is-better evidence",
        bottom_margin=62,
    )
    plot(fig, "r110_sustain_resource_scoreboard")

    section(
        "Efficiency structure",
        "The rank-flow panel reveals whether a framework is consistently efficient across Runtime, Energy and CO₂. The frontier panel then isolates the measured Runtime–Energy trade-off without collapsing the resources into one score.",
    )
    left, right = st.columns([1, 1.08])
    with left:
        fig = sustainability_rank_bump(sdf)
        apply_research_layout(
            fig, height=430, legend="bottom",
            title="Resource-efficiency rank flow",
            bottom_margin=102,
        )
        plot(fig, "r110_sustain_rank_flow")
        st.caption(
            "Rank 1 is best (lowest measured resource use) on each axis. Parallel lines indicate consistent efficiency; crossings expose resource-specific trade-offs."
        )
    with right:
        fig = sustainability_efficiency_frontier(sdf)
        apply_research_layout(
            fig, height=430, legend="none",
            title="Runtime–Energy efficiency frontier · measured evidence",
            bottom_margin=62,
        )
        plot(fig, "r110_sustain_efficiency_frontier")
        st.caption(
            "Outlined diamonds are nondominated in Runtime and Energy. The star is the run-relative 2D resource compromise (closest Pareto point to the minimum-runtime/minimum-energy ideal); it is not a new AutoML recommendation."
        )

    intensity = sdf.copy()
    intensity["Energy kWh"] = pd.to_numeric(intensity["Energy kWh"], errors="coerce")
    intensity["CO₂ kg"] = pd.to_numeric(intensity["CO₂ kg"], errors="coerce")
    if ((intensity["Energy kWh"] > 0) & intensity["CO₂ kg"].notna()).any():
        section(
            "Carbon-accounting consistency",
            "Observed CO₂/Energy intensity is compared with the carbon-intensity metadata recorded by the measurement backend. This makes the Energy–CO₂ relationship auditable instead of showing another redundant scatter plot.",
        )
        fig = carbon_intensity_consistency_figure(sdf)
        apply_research_layout(
            fig, height=390, legend="bottom",
            title="Carbon intensity audit · reported metadata vs observed CO₂/Energy",
            bottom_margin=96,
        )
        plot(fig, "r110_sustain_carbon_intensity")
        st.caption(
            "Observed intensity = measured CO₂ ÷ measured Energy, expressed in gCO₂/kWh. A short reported-to-observed segment indicates internally consistent accounting; differences should be interpreted with CodeCarbon's measurement scope and metadata."
        )

    render_phase14_sustainability_details(results)


def trust_calibration_v2_page():
    return trust_calibration_research_page()


def information_seeking_v2_page():
    return information_seeking_research_page()
