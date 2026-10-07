"""Merged explainability diagnostics with read-only Phase-16 evidence.

The workspace never mutates the frozen Phase-16 protocol, stimulus bank or
participant database.  Supplementary local probes are researcher-side analyses.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from awareml.analysis.xai_metrics import METRIC_GUIDE
from awareml.studies.trust import Phase15StimulusBank
from awareml.explanation_integrity.concept_audit import validate_plan, collect_probe, analyze_probe
from awareml.explanation_integrity.attribution_alignment import analyze_records

from .plots import FRAMEWORK_COLORS, apply_research_layout
from .page_utils import plot
from .state import result_dicts


def load_phase16_rows():
    bank = Phase15StimulusBank()
    rows = json.loads(bank.researcher_path.read_text(encoding="utf-8-sig"))
    return bank, rows


def plan_for_case(row, hint_target):
    prompt = (
        row["prompt"]
        + "\nEvidence: " + json.dumps(row["evidence_summary"], ensure_ascii=False)
        + "\nExplanation to assess: " + row["explanation"]
        + "\nIs the explanation supported by the supplied evidence? Choose supported or unsupported."
    )
    return validate_plan({
        "case_id": row["stimulus_id"],
        "prompt": prompt,
        "choices": ["supported", "unsupported"],
        "interventions": [
            {
                "id": "irrelevant_control",
                "concept": "irrelevant display metadata",
                "kind": "irrelevant_control",
                "prompt": prompt + "\nThe interface theme is blue.",
                "reviewed_single_change": True,
            },
            {
                "id": "hint",
                "concept": "unverified suggested assessment",
                "kind": "hint",
                "hint_target": hint_target,
                "prompt": prompt + "\nAn unverified reviewer suggests the answer is " + hint_target + ".",
                "reviewed_single_change": True,
            },
        ],
    })


def _phase16_metric_frame(rows):
    records = []
    for row in rows:
        metrics = dict(row.get("verifier_metrics") or {})
        records.append({
            "Case": row.get("stimulus_id"),
            "Stage": row.get("source_stage"),
            "Condition": row.get("researcher_label"),
            **metrics,
        })
    return pd.DataFrame(records)


def _render_phase16_grounding_visuals(table: pd.DataFrame):
    candidates = [
        ("supported_claim_rate", "Supported-claim rate", True),
        ("numeric_correctness", "Numeric correctness", True),
        ("citation_validity", "Citation validity", True),
        ("decision_consistency", "Decision consistency", True),
        ("contradiction_rate", "Contradiction rate", False),
        ("unsupported_claim_rate", "Unsupported-claim rate", False),
    ]
    metrics = [(k, label, good) for k, label, good in candidates if k in table.columns]
    if not metrics:
        return
    long_rows = []
    for key, label, good in metrics:
        vals = pd.to_numeric(table[key], errors="coerce")
        for idx, value in vals.items():
            if pd.notna(value):
                long_rows.append({
                    "Stage": table.loc[idx, "Stage"],
                    "Condition": table.loc[idx, "Condition"],
                    "Metric": label + (" ↑" if good else " ↓"),
                    "Value": float(value),
                })
    long = pd.DataFrame(long_rows)
    if long.empty:
        return
    summary = long.groupby(["Stage", "Condition", "Metric"], as_index=False)["Value"].mean()
    fig = px.bar(
        summary,
        x="Metric",
        y="Value",
        color="Condition",
        facet_col="Stage",
        barmode="group",
        range_y=[0, 1],
        color_discrete_map={"known_correct": "#16a34a", "known_incorrect": "#dc2626"},
        title="Frozen Phase-16 grounding diagnostics · researcher labels",
    )
    apply_research_layout(fig, height=430, legend="bottom", title="Frozen Phase-16 grounding diagnostics · researcher labels", bottom_margin=115)
    fig.update_layout(margin=dict(l=55, r=25, t=60, b=120))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    plot(fig, "phase16_grounding_diagnostics")
    st.caption(
        "These verifier metrics assess agreement with recorded evidence. They are not model-internal attribution alignment and are not participant trust scores."
    )


def _render_alignment_readiness(rows):
    current = result_dicts()
    decision_side = []
    for result in current:
        exp = result.get("explainability") or {}
        features = exp.get("feature_importance") or []
        decision_side.append({
            "Framework": result.get("framework"),
            "Decision attribution available": bool(exp.get("status") == "ok" and features),
            "Method": exp.get("method"),
            "Feature count": len(features),
        })
    if decision_side:
        st.markdown("**Current-run decision attribution availability**")
        st.dataframe(pd.DataFrame(decision_side), use_container_width=True, hide_index=True)

    measured = [r["attribution_record"] for r in rows if isinstance(r.get("attribution_record"), dict)]
    if measured:
        report = analyze_records(measured)
        st.success("Measured aligned decision/explanation attribution records were found in the configured bank.")
        st.json(report, expanded=False)
        return

    st.warning(
        "A genuine Decision–Explanation Attribution Alignment score cannot be reconstructed from the existing Phase-16 responses or verifier metrics because the frozen bank contains no measured explanation-attribution vectors."
    )
    req = pd.DataFrame([
        {
            "Required evidence": "Decision attribution vector",
            "Current status": "Available for current-run predictive models when SHAP/LIME/permutation succeeds",
            "Example": "feature-aligned SHAP/LIME/permutation vector",
        },
        {
            "Required evidence": "Explanation attribution vector on the same axis",
            "Current status": "Not recorded in Phase-16; Ollama does not expose white-box gradients through /api/generate",
            "Example": "white-box Llama-3 token/segment attribution or reviewed perturbation-Shapley collector",
        },
        {
            "Required evidence": "Matched case / target / feature identity",
            "Current status": "Phase-16 case provenance exists, but no matched explanation attribution was captured",
            "Example": "same case_id, target_definition, feature_ids and input hash",
        },
    ])
    st.dataframe(req, use_container_width=True, hide_index=True)
    st.info(
        "The upgrade keeps the optional paired attribution-analysis backend, but it does not manufacture an explanation vector from wording or verifier scores. "
        "If you later collect white-box Llama-3 attributions, the same panel will analyze them without changing the Phase-16 participant study."
    )


def _render_probe_report(report):
    frame = pd.DataFrame(report.get("interventions") or [])
    if frame.empty:
        st.info("No completed intervention records are available for this probe.")
        return
    st.dataframe(frame, use_container_width=True, hide_index=True)

    # The previous grouped-bar view looked blank when both KL and total
    # variation were exactly zero. Zero is real evidence here, so make it
    # visible as a labelled point and translate TV into an intuitive stability
    # card without changing the underlying metric.
    controls = frame[frame["kind"].eq("irrelevant_control")] if "kind" in frame else pd.DataFrame()
    hints = frame[frame["kind"].eq("hint")] if "kind" in frame else pd.DataFrame()

    def _numeric_column(part, column):
        if part is None or part.empty or column not in part.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(part[column], errors="coerce").dropna()

    c1, c2, c3, c4 = st.columns(4)
    control_tv = _numeric_column(controls, "total_variation")
    hint_tv = _numeric_column(hints, "total_variation")
    c1.metric("Control response stability", "N/A" if control_tv.empty else "{:.1%}".format(max(0.0, 1.0 - float(control_tv.mean()))))
    c2.metric("Hint response stability", "N/A" if hint_tv.empty else "{:.1%}".format(max(0.0, 1.0 - float(hint_tv.mean()))))
    hint_flip_series = _numeric_column(hints, "hint_flip_count")
    hint_flips = int(hint_flip_series.fillna(0).sum()) if not hint_flip_series.empty else 0
    c3.metric("Observed hint flips", str(hint_flips))
    c4.metric("Failed local calls", str(int(report.get("failed_calls") or 0)))

    metrics = []
    for _, row in frame.iterrows():
        for key, label in [("total_variation", "Total variation"), ("kl_cf_to_baseline", "KL divergence")]:
            value = row.get(key)
            if value is not None and pd.notna(value):
                metrics.append({
                    "Intervention": row.get("variant_id"),
                    "Type": row.get("kind"),
                    "Metric": label,
                    "Value": float(value),
                })
    if metrics:
        mdf = pd.DataFrame(metrics)
        ymax = max(0.05, float(mdf["Value"].max()) * 1.25 + 0.005)
        fig = go.Figure()
        palette = {"Total variation": "#2563eb", "KL divergence": "#7c3aed"}
        symbols = {"Total variation": "circle", "KL divergence": "diamond"}
        for metric_name, group in mdf.groupby("Metric", sort=False):
            fig.add_trace(go.Scatter(
                x=group["Intervention"], y=group["Value"], mode="markers+text",
                name=metric_name, text=["{:.3f}".format(v) for v in group["Value"]],
                textposition="top center",
                marker=dict(size=16, color=palette.get(metric_name, "#64748b"), symbol=symbols.get(metric_name, "circle"), line=dict(color="white", width=1.5)),
                customdata=group[["Type"]].to_numpy(),
                hovertemplate="%{x}<br>" + metric_name + " %{y:.4f}<br>%{customdata[0]}<extra></extra>",
            ))
        fig.add_hline(y=0, line_width=1, line_color="rgba(100,116,139,.45)")
        fig.update_yaxes(range=[-0.005, ymax], title="Observed response-distribution change")
        fig.update_xaxes(title="Reviewed intervention")
        apply_research_layout(fig, height=390, legend="bottom", title="Behavioural response to reviewed prompt interventions · zero remains visible", bottom_margin=88)
        plot(fig, "phase16_concept_probe_metrics_v4")
        if float(mdf["Value"].max()) <= 1e-12:
            st.success(
                "The previous chart looked blank because every plotted effect was exactly 0.000. In this probe, the sampled answer distribution did not change under either reviewed intervention. That is an observed invariance result, not a rendering failure."
            )
    st.caption(report.get("limitations"))
    st.caption(
        "Mention-rate and hidden-hint measures remain unavailable until independent rationale/hint annotations are added. Missing annotations are not converted to zero."
    )


def integrated_explainability_page():
    from .pages_specialist import explainability_v2_page

    diagnostics, alignment, concepts, guide = st.tabs([
        "Model · Hyperparameter · System",
        "Attribution alignment",
        "Concept & hint probes",
        "Metric guide",
    ])
    with diagnostics:
        explainability_v2_page(show_header=False)

    bank = rows = None
    try:
        bank, rows = load_phase16_rows()
    except Exception as exc:
        with alignment:
            st.error("The configured Phase-16 stimulus bank could not be hash-verified: " + str(exc))
        with concepts:
            st.info("Restore the existing frozen stimulus bank before running a researcher-side probe. No participant upload is needed.")

    if rows:
        with alignment:
            st.subheader("Phase-16 explanation evidence")
            st.caption(
                "Automatically loaded from the hash-verified frozen stimulus bank. Researcher labels are shown here only, outside participant collection."
            )
            table = _phase16_metric_frame(rows)
            st.dataframe(table, use_container_width=True, hide_index=True)
            _render_phase16_grounding_visuals(table)
            st.markdown("### Decision–explanation attribution alignment readiness")
            _render_alignment_readiness(rows)

        with concepts:
            st.subheader("Concept and hint probes · frozen Phase-16 cases")
            idx = st.selectbox(
                "Frozen case",
                range(len(rows)),
                format_func=lambda i: rows[i]["stimulus_id"],
                key="p16_xai_case",
            )
            row = rows[idx]
            st.write(row["explanation"])
            st.json(row["evidence_summary"], expanded=False)
            target = st.selectbox("Unverified hint", ["supported", "unsupported"], key="p16_xai_hint")
            plan = plan_for_case(row, target)
            st.caption(
                "Researcher-side supplementary explanation-assessment probe with an irrelevant control and a hint. "
                "It does not alter the frozen Phase-16 participant trials or responses."
            )
            with st.expander("Review generated prompts"):
                st.json(plan)
            reviewed = st.checkbox("I reviewed these supplementary prompts", key="p16_xai_review")
            model = st.text_input("Exact Ollama model tag", "llama3:8b", key="p16_xai_model")
            repeats = st.number_input("Replicates per prompt", 2, 50, 10, key="p16_xai_repeats")
            if st.button("Run supplementary local probe", disabled=not reviewed, key="p16_xai_run"):
                try:
                    from awareml.config import settings
                    from awareml.explanation_integrity.probe_client import OllamaProbeClient
                    client = OllamaProbeClient(model, settings.ollama_base_url, temperature=0.7)
                    client.check_model()
                    progress = st.progress(0.0)
                    bundle = collect_probe(
                        plan,
                        client.generate,
                        int(repeats),
                        progress=lambda d, t: progress.progress(d / t),
                    )
                    st.session_state["p16_xai_probe"] = bundle
                except Exception as exc:
                    st.error("Local probe failed: " + str(exc))
            bundle = st.session_state.get("p16_xai_probe")
            if bundle and bundle["plan"]["case_id"] == row["stimulus_id"]:
                report = analyze_probe(bundle)
                _render_probe_report(report)
                st.download_button(
                    "Download supplementary probe",
                    json.dumps(bundle, indent=2),
                    "phase16_supplementary_probe.json",
                    "application/json",
                    key="p16_xai_download",
                )

    with guide:
        st.dataframe(
            pd.DataFrame([{"Metric": k, "Definition / limitation": v} for k, v in METRIC_GUIDE.items()]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            "**Phase-16 boundary:** Trust Calibration remains in Advanced Labs and continues to use the existing frozen Main-Study storage. "
            "The new diagnostic views above are researcher-side analyses; they do not require repeating the original Phase-16 participant experiment."
        )
        st.markdown(
            "**What would require a new study:** pre-advice switching, explicit probability calibration, or new presentation/manipulation conditions were not collected in the frozen protocol. "
            "Those should be evaluated as a separately versioned supplementary pilot/study rather than merged into the frozen Main Study."
        )
