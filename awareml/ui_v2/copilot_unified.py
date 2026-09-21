from __future__ import annotations

import json

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from awareml.llm import (
    CopilotService,
    GoalParser,
    GroundedCopilotChat,
    EvidenceGroundedObjectiveSelectorV33,
    OllamaClient,
    ReviewStore,
)
from awareml.llm.review import review_proposal

from .components import hero, humanize_rationale_text
from .copilot_v31_components import (
    clear_previous_copilot_result,
    render_copilot_clarification,
    set_copilot_clarification,
)
from .data import load_v2_recommender
from .page_utils import dataset_ready, phase_pills
from .copilot_workspace_v5 import render_goal_framework_guidance
from .pre14_usability import (
    apply_human_objective_override,
    clean_pre_run_rationale,
    persist_objective_review,
)
from .state import ROOT, ensure_research_state


CANONICAL_OBJECTIVES = ("Accuracy", "Runtime", "Energy", "CO2")
WEIGHT_KEYS = {
    "Accuracy": "accuracy",
    "Runtime": "runtime",
    "Energy": "energy",
    "CO2": "co2",
}
DISPLAY_OBJECTIVE = {
    "Accuracy": "Accuracy",
    "Runtime": "Runtime",
    "Energy": "Energy",
    "CO2": "CO₂",
}
EXACT_MODEL = "llama3:8b"


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


def _selected_objectives(interpretation: Any) -> Sequence[str]:
    data = _as_dict(interpretation)
    selected = set(data.get("selected_objectives") or [])
    return [name for name in CANONICAL_OBJECTIVES if name in selected]


def _weights(interpretation: Any) -> Dict[str, float]:
    data = _as_dict(interpretation)
    raw = _as_dict(data.get("primary_weights"))
    return {
        "accuracy": float(raw.get("accuracy", 0.0) or 0.0),
        "runtime": float(raw.get("runtime", 0.0) or 0.0),
        "energy": float(raw.get("energy", 0.0) or 0.0),
        "co2": float(raw.get("co2", 0.0) or 0.0),
    }


def _ranked_frame(state: Mapping[str, Any]) -> pd.DataFrame:
    ranked = state.get("copilot_ranked")
    if isinstance(ranked, pd.DataFrame):
        return ranked.copy()
    if ranked is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(ranked)
    except Exception:
        return pd.DataFrame()


def _clear_scenario_state(state: Dict[str, Any]) -> None:
    clear_previous_copilot_result(state)
    for key in (
        "copilot_human_corrected_objectives",
        "copilot_human_corrected_weights",
        "copilot_objective_review_persisted",
    ):
        state.pop(key, None)


def _priority_summary(selected: Sequence[str], weights: Mapping[str, float]) -> str:
    parts = []
    for objective in selected:
        key = WEIGHT_KEYS[objective]
        parts.append(
            "{} {:.0%}".format(
                DISPLAY_OBJECTIVE[objective],
                float(weights.get(key, 0.0)),
            )
        )
    return " · ".join(parts) if parts else "No active objective priorities"


def _weighting_explanation(selected: Sequence[str]) -> str:
    count = len(list(selected or []))
    if count <= 0:
        return "No objective is selected, so no objective weights are assigned."
    share = 1.0 / float(count)
    return (
        "{} objective{} selected. The equal-selected policy assigns {:.0%} "
        "to each selected objective and 0% to each unselected objective."
    ).format(count, "" if count == 1 else "s", share)


def _prediction_table(
    framework: str,
    selected: Sequence[str],
    weights: Mapping[str, float],
    ranked: pd.DataFrame,
) -> pd.DataFrame:
    if ranked.empty or "framework" not in ranked.columns:
        return pd.DataFrame()

    match = ranked[ranked["framework"].astype(str).eq(str(framework))]
    if match.empty:
        return pd.DataFrame()

    row = match.iloc[0]
    specs = {
        "Accuracy": ("accuracy", "Higher is better", lambda v: "{:.4f}".format(float(v))),
        "Runtime": ("runtime", "Lower is better", lambda v: "{:.3f} s".format(float(v))),
        "Energy": ("energy", "Lower is better", lambda v: "{:.6f} kWh".format(float(v))),
        "CO2": ("co2", "Lower is better", lambda v: "{:.6f} kg".format(float(v))),
    }

    rows = []
    for objective in selected:
        metric, direction, formatter = specs[objective]
        value = row.get(metric)
        rendered = "N/A" if value is None or pd.isna(value) else formatter(value)
        rows.append(
            {
                "Priority": DISPLAY_OBJECTIVE[objective],
                "Weight": "{:.0%}".format(float(weights.get(WEIGHT_KEYS[objective], 0.0))),
                "Predicted outcome": rendered,
                "Goal": direction,
            }
        )
    return pd.DataFrame(rows)


def _fairness_plan_text(
    fairness: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Tuple[str, str]:
    requested = bool(fairness.get("requested"))
    status = str(fairness.get("status") or "")
    sensitive = fairness.get("sensitive_attribute") or state.get("sensitive")

    if requested and status == "requires_sensitive_attribute":
        constraint = "Requested — select a sensitive attribute"
    elif requested:
        constraint = "Requested in this goal"
    else:
        constraint = "Not requested by this goal"

    if sensitive:
        audit = "Available after execution for '{}'".format(sensitive)
    else:
        audit = "Select a sensitive attribute in Run Studio to audit fairness"
    return constraint, audit



def _render_css() -> None:
    st.markdown(
        """
        <style>
        .awareml-flow-card{border:1px solid rgba(148,163,184,.38);border-radius:15px;padding:16px 17px;min-height:145px;background:linear-gradient(180deg,#fff 0%,#f8fbff 100%);box-shadow:0 7px 22px rgba(15,23,42,.045)}
        .awareml-step{display:inline-block;min-width:34px;padding:5px 8px;border-radius:999px;background:#eef4ff;border:1px solid #cddbf5;color:#1d4ed8;font-weight:700;font-size:12px;letter-spacing:.04em;margin-bottom:11px}
        .awareml-card-title{font-size:18px;font-weight:750;margin-bottom:8px;color:#0f172a}
        .awareml-card-copy{font-size:13px;line-height:1.55;color:#334155}
        .awareml-objective-card,.awareml-hcai-card,.awareml-summary-card{border:1px solid rgba(148,163,184,.40);border-radius:15px;padding:15px 16px;background:#fff}
        .awareml-objective-card{min-height:128px}.awareml-hcai-card{min-height:132px;background:linear-gradient(180deg,#fff 0%,#fafcff 100%)}.awareml-summary-card{min-height:126px}
        .awareml-objective-name,.awareml-hcai-name{font-size:15px;font-weight:750;color:#0f172a}.awareml-objective-value{font-size:30px;font-weight:800;margin:8px 0 4px;color:#0f172a}.awareml-objective-state,.awareml-hcai-copy,.awareml-summary-label{font-size:12px;color:#64748b}.awareml-hcai-value{font-size:20px;font-weight:800;color:#0f172a;margin:8px 0}.awareml-summary-value{font-size:24px;line-height:1.25;font-weight:800;color:#0f172a}
        .awareml-decision-banner{border-left:4px solid #2563eb;background:#eef5ff;border-radius:10px;padding:14px 16px;margin:8px 0 16px}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _render_step_cards() -> None:
    st.markdown("## How Goal Copilot works")
    items = [
        ("01", "Understand", "AwareML interprets the natural-language deployment goal and identifies the objectives implied by the wording."),
        ("02", "Weight", "Selected objectives receive equal preference weight under the documented equal-selected policy."),
        ("03", "Predict", "ML Recommender ranks the five AutoML frameworks from dataset meta-features and the active objective weights."),
        ("04", "You decide", "You review the interpretation and recommendation, then approve, adjust or reject the plan."),
    ]
    cols = st.columns(4)
    for col, (number, title, body) in zip(cols, items):
        with col:
            st.markdown(
                '<div class="awareml-flow-card"><div class="awareml-step">{}</div><div class="awareml-card-title">{}</div><div class="awareml-card-copy">{}</div></div>'.format(number,title,body),
                unsafe_allow_html=True,
            )
    st.caption("Responsibility boundary: the LLM interprets the goal; ML Recommender produces the framework ranking; the human makes the final decision.")

def _render_context_strip(has_dataset: bool, has_observed_run: bool, state: Mapping[str, Any]) -> None:
    cols = st.columns(3)
    with cols[0]:
        with st.container(border=True):
            st.caption("DATASET CONTEXT")
            if has_dataset:
                st.markdown("**Ready for dataset-aware ranking**")
                st.write(str(state.get("dataset_name") or "Loaded dataset"))
                st.caption("Target: {}".format(state.get("target") or "not selected"))
            else:
                st.markdown("**Objective-only mode**")
                st.caption("No dataset is required to interpret the deployment goal.")
    with cols[1]:
        with st.container(border=True):
            st.caption("DECISION ENGINE")
            st.markdown("**LLaMA 3 8B → ML Recommender**")
            st.caption("Objective weighting: equal_selected_v1")
    with cols[2]:
        with st.container(border=True):
            st.caption("EVIDENCE STATE")
            if has_observed_run:
                st.markdown("**Observed benchmark available**")
                st.caption("Measured post-run evidence remains available in Decision Lab.")
            elif has_dataset:
                st.markdown("**Prediction-ready**")
                st.caption("Framework execution is not required before recommendation.")
            else:
                st.markdown("**Framework prediction pending**")
                st.caption("Load a dataset + target when you want a framework ranking.")

def _render_understanding(interpretation: Any, parse_meta: Mapping[str, Any], state: Mapping[str, Any]) -> Sequence[str]:
    data = _as_dict(interpretation)
    selected = list(_selected_objectives(data))
    weights = _weights(data)
    st.markdown("## 1 · What AwareML understood")
    human_override = list(state.get("copilot_human_corrected_objectives") or [])
    if human_override:
        st.success("Human-reviewed priorities are active: **{}**. The current framework ranking uses the corrected weights.".format(" + ".join(DISPLAY_OBJECTIVE.get(x,x) for x in human_override)))
    elif selected:
        st.success("AwareML currently interprets the scenario as: **{}**.".format(" + ".join(DISPLAY_OBJECTIVE[x] for x in selected)))
    else:
        st.warning("AwareML did not produce a supported objective set. Clarify the goal before continuing.")
    cols = st.columns(4)
    for col, objective in zip(cols, CANONICAL_OBJECTIVES):
        value=float(weights.get(WEIGHT_KEYS[objective],0.0)); status="Selected" if objective in selected else "Not selected"
        with col:
            st.markdown('<div class="awareml-objective-card"><div class="awareml-objective-name">{}</div><div class="awareml-objective-value">{:.0%}</div><div class="awareml-objective-state">{}</div></div>'.format(DISPLAY_OBJECTIVE[objective],value,status),unsafe_allow_html=True)
    st.info("Preference weights are not confidence scores. " + _weighting_explanation(selected))
    hcai=_as_dict(data.get("hcai_requirements"))
    if hcai:
        from awareml.llm.objective_selection import infer_hcai_evidence
        scenario_text=str(state.get("copilot_goal") or "")
        hcai_evidence=infer_hcai_evidence(scenario_text)
        st.markdown("### Human-centred AI oversight")
        st.caption("These controls do not change the four-objective recommender utility. They define how the selected plan should be monitored, audited and explained.")
        drift_e=_as_dict(hcai_evidence.get("drift")); fairness_e=_as_dict(hcai_evidence.get("fairness")); explain_e=_as_dict(hcai_evidence.get("explainability"))
        sensitive=state.get("sensitive")
        drift_level=str(hcai.get("drift_sensitivity") or "moderate").title()
        fairness_required=bool(hcai.get("fairness_required"))
        explain_level=str(hcai.get("explainability_level") or "moderate").title()
        def _evidence_label(item):
            status=str(item.get("status") or "")
            return {"scenario_supported":"Scenario-supported","platform_baseline":"AwareML baseline","not_requested":"Not requested"}.get(status,status.replace("_"," ").title() or "N/A")
        cards=st.columns(3)
        content=[
            ("Drift monitoring",drift_level,_evidence_label(drift_e),"Purpose: detect changes in the streaming data or model behaviour.","Plan: keep drift monitoring enabled; review alerts and recovery behaviour after execution.",drift_e),
            ("Fairness audit","Required" if fairness_required else "Not requested",_evidence_label(fairness_e),"Purpose: check whether model quality or errors differ across protected groups.",("Plan: audit the selected sensitive attribute '{}' after execution.".format(sensitive) if sensitive else "Plan: select a sensitive attribute before enabling a group fairness audit."),fairness_e),
            ("Explanation support",explain_level,_evidence_label(explain_e),"Purpose: make the recommendation and post-run model behaviour inspectable.","Plan: use the configured explanation method and preserve evidence provenance.",explain_e),
        ]
        for col,(name,value,status,purpose,plan,item) in zip(cards,content):
            with col:
                with st.container(border=True):
                    st.caption(name.upper()); st.markdown("### {}".format(value)); st.write("**Evidence state:** {}".format(status))
                    if item.get("evidence"): st.caption("Scenario cue: “{}”".format(item.get("evidence")))
                    st.write(purpose); st.caption(plan)
        with st.expander("HCAI oversight boundary", expanded=False):
            st.write("Accuracy, Runtime, Energy and CO₂ are the optimization objectives passed to the ML Recommender. Drift, fairness and explainability remain separate oversight/configuration controls so they stay visible without becoming extra utility dimensions.")

    return selected

def _render_decision_provenance(has_dataset: bool) -> None:
    with st.expander("How the recommendation is produced", expanded=False):
        rows=[
            {"Stage":"Goal interpretation","Source":"LLaMA 3 8B + V3.3 hybrid evidence guard","Role":"Infers Accuracy / Runtime / Energy / CO₂ from the user's wording."},
            {"Stage":"Objective weighting","Source":"equal_selected_v1","Role":"Assigns equal weight to selected objectives."},
            {"Stage":"Framework prediction","Source":"ML Recommender","Role":"Ranks five frameworks from the dataset meta-profile." if has_dataset else "Waits for dataset + target because framework behavior is dataset-dependent."},
            {"Stage":"Configuration synthesis","Source":"Configuration synthesizer","Role":"Maps the recommended framework to supported implementation defaults."},
            {"Stage":"Decision authority","Source":"Human review","Role":"Accept, correct, edit or reject the plan."},
        ]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

def _render_objective_review(
    state: Dict[str, Any],
    selected: Sequence[str],
    step_number: int,
) -> None:
    st.markdown("## {} · Review the priorities".format(step_number))
    st.caption(
        "The objective interpretation is advisory. Confirm it, correct it, or reject it."
    )

    choice = st.segmented_control(
        "Did AwareML understand your priorities correctly?",
        ["Accept", "Correct priorities", "Reject"],
        default="Accept",
        key="goal_v2_objective_review_choice_{}".format(step_number),
    ) or "Accept"

    corrected = None
    if choice == "Correct priorities":
        corrected = st.multiselect(
            "Select the priorities AwareML should use",
            list(CANONICAL_OBJECTIVES),
            default=list(selected),
            key="goal_v2_corrected_objectives_{}".format(step_number),
            help=(
                "If a dataset is loaded, saving this correction reranks ML Recommender "
                "using the same dataset meta-profile without asking LLaMA again."
            ),
        )

    note = st.text_input(
        "Review note (optional)",
        key="goal_v2_objective_review_note_{}".format(step_number),
        placeholder="Example: Energy also matters because the service must preserve battery life.",
    )

    if st.button(
        "Save priority review",
        key="goal_v2_save_objective_review_{}".format(step_number),
    ):
        if choice == "Correct priorities" and not corrected:
            st.warning("Choose at least one corrected priority before saving.")
        else:
            final_set = list(corrected or selected)
            record = persist_objective_review(
                state,
                decision=choice,
                selected_seen=list(selected),
                corrected=(final_set if choice == "Correct priorities" else None),
                note=note or None,
                source="copilot_unified_view_v2",
            )
            state["copilot_objective_review_persisted"] = record

            if choice == "Correct priorities":
                new_top = apply_human_objective_override(state, final_set)
                if new_top:
                    state["copilot_unified_flash"] = (
                        "Priority correction saved. ML Recommender reranked the "
                        "same dataset profile and now predicts {} as #1.".format(new_top)
                    )
                else:
                    state["copilot_unified_flash"] = (
                        "Priority correction saved. The corrected priorities will be "
                        "used when a dataset-aware framework ranking is available."
                    )
                st.rerun()
            elif choice == "Reject":
                state["copilot_unified_flash"] = (
                    "Priority interpretation rejected and recorded. Revise the scenario before approving a plan."
                )
                st.rerun()
            else:
                state["copilot_unified_flash"] = (
                    "Priority interpretation accepted and recorded in the Copilot audit trail."
                )
                st.rerun()

    saved = state.get("copilot_objective_review_persisted")
    if isinstance(saved, dict):
        st.caption(
            "Saved priority review: {} · {}".format(
                saved.get("review_id") or "N/A",
                str(saved.get("decision") or ""),
            )
        )


def _render_context_free_next_step() -> None:
    st.markdown("## 4 · Next step")
    st.info(
        "Your objectives are ready. To obtain a framework recommendation, load a dataset and choose its target, then generate the Goal Copilot proposal again."
    )
    with st.container(border=True):
        st.markdown("**What becomes available with a dataset**")
        st.write(
            "Recommended framework · predicted rank · objective outcomes · top alternatives · supported configuration"
        )
        st.caption(
            "Goal Copilot does not invent a framework when no dataset context is available."
        )


def _render_post_approval_state(state: Mapping[str, Any]) -> None:
    saved=_as_dict(state.get("copilot_review"))
    if not saved:
        return
    decision=str(saved.get("decision") or "")
    if decision in {"approved","approved_with_edits"}:
        st.success("Plan approved{}.".format(" with recorded edits" if decision=="approved_with_edits" else ""))
        st.markdown("### What happens next")
        cols=st.columns(3)
        items=[
            ("1 · Decision recorded","The approval is stored in the append-only Copilot review log for auditability."),
            ("2 · Execute in Run Studio","Approval does not automatically start the benchmark. Use the approved settings in Run Studio when ready."),
            ("3 · Validate after execution","Inspect measured behavior in Streaming Observatory and compare post-run outcomes in Decision Lab."),
        ]
        for col,(title,body) in zip(cols,items):
            with col:
                with st.container(border=True):
                    st.markdown("**{}**".format(title)); st.caption(body)
    elif decision=="rejected":
        st.warning("This plan was rejected. Revise the deployment goal, priorities or configuration before approving another plan.")


def _render_final_plan_review(state: Dict[str, Any], proposal: Mapping[str, Any]) -> None:
    st.markdown("## 5 · Final plan decision")
    st.caption("Approve, refine or reject the recommendation. Advanced edits are configuration changes only: they do not silently re-rank the recommended framework.")
    proposal_obj=state.get("copilot_proposal")
    if proposal_obj is None:
        st.warning("No dataset-aware proposal is available to review."); return
    config=_as_dict(_as_dict(proposal).get("proposed_config"))
    mode=st.segmented_control("Decision",["Approve plan","Approve with edits","Reject plan"],default="Approve plan",key="goal_v2_final_review_mode") or "Approve plan"
    edits=None; edit_validation_ok=True
    if mode=="Approve with edits":
        st.markdown("### Advanced plan controls")
        st.caption("Edit only the configuration you intend to execute. Every accepted change is stored in the append-only human-review audit trail.")
        with st.expander("Execution and drift controls",expanded=True):
            c1,c2=st.columns(2)
            with c1: window=st.number_input("Window size",min_value=50,max_value=100000,value=int(config.get("window_size",1000)),step=50,key="goal_v2_review_window")
            with c2: budget=st.number_input("Time budget (seconds)",min_value=1.0,max_value=86400.0,value=float(config.get("time_budget_sec",60.0)),step=5.0,key="goal_v2_review_budget")
            drift_cfg=_as_dict(config.get("drift")); d1,d2,d3=st.columns(3)
            with d1: drift_enabled=st.toggle("Enable drift monitoring",value=bool(drift_cfg.get("monitoring_enabled",True)),key="goal_v2_review_drift_enabled")
            detector_options=["ADWIN","None"]; current_detector=str(drift_cfg.get("detector") or "ADWIN")
            with d2: detector=st.selectbox("Drift detector",detector_options,index=detector_options.index(current_detector) if current_detector in detector_options else 0,disabled=not drift_enabled,key="goal_v2_review_drift_detector")
            sens_options=["low","moderate","high"]; current_sens=str(drift_cfg.get("sensitivity") or "moderate").lower()
            with d3: drift_sensitivity=st.selectbox("Drift sensitivity",sens_options,index=sens_options.index(current_sens) if current_sens in sens_options else 1,disabled=not drift_enabled,key="goal_v2_review_drift_sensitivity")
        with st.expander("Framework hyperparameters",expanded=False):
            params=_as_dict(config.get("framework_parameters")); params_text=st.text_area("Framework parameters (JSON)",value=json.dumps(params,indent=2,sort_keys=True),height=170,key="goal_v2_review_framework_parameters",help="Only parameters supported by the selected framework should be changed.")
            try:
                parsed_params=json.loads(params_text or "{}")
                if not isinstance(parsed_params,dict): raise ValueError("Framework parameters must be a JSON object.")
            except Exception as exc:
                parsed_params={}; edit_validation_ok=False; st.error("Invalid framework-parameter JSON: {}".format(exc))
        with st.expander("Fairness and explainability controls",expanded=True):
            fairness_cfg=_as_dict(config.get("fairness")); explain_cfg=_as_dict(config.get("explainability")); f1,f2=st.columns(2)
            with f1:
                fairness_enabled=st.toggle("Enable fairness audit",value=bool(fairness_cfg.get("requested",False)),key="goal_v2_review_fairness_enabled")
                fairness_metrics=["composite","demographic_parity","equal_opportunity","equalized_odds","predictive_parity","group_ece"]; current_metric=str(fairness_cfg.get("metric") or "composite")
                fairness_metric=st.selectbox("Fairness metric",fairness_metrics,index=fairness_metrics.index(current_metric) if current_metric in fairness_metrics else 0,disabled=not fairness_enabled,key="goal_v2_review_fairness_metric")
                sensitive=state.get("sensitive"); st.caption("Sensitive attribute: {}".format(sensitive or "not selected in Run Studio"))
                if fairness_enabled and not sensitive: st.warning("A group fairness audit requires a sensitive attribute before execution.")
            with f2:
                levels=["low","moderate","high"]; current_level=str(explain_cfg.get("level") or "moderate").lower(); explain_level=st.selectbox("Explainability level",levels,index=levels.index(current_level) if current_level in levels else 1,key="goal_v2_review_explainability_level")
                methods=["auto","SHAP","LIME","permutation"]; current_method=str(explain_cfg.get("method") or "auto"); explain_method=st.selectbox("Explanation method",methods,index=methods.index(current_method) if current_method in methods else 0,key="goal_v2_review_explainability_method")
        with st.expander("Sustainability measurement",expanded=False):
            sustain_cfg=_as_dict(config.get("sustainability")); s1,s2=st.columns(2)
            with s1: track_energy=st.toggle("Track energy",value=bool(sustain_cfg.get("track_energy",True)),key="goal_v2_review_track_energy")
            with s2: track_co2=st.toggle("Track CO₂",value=bool(sustain_cfg.get("track_co2",True)),key="goal_v2_review_track_co2")
        fairness_status="enabled" if fairness_enabled and state.get("sensitive") else "requires_sensitive_attribute" if fairness_enabled else "disabled"
        edits={"window_size":int(window),"time_budget_sec":float(budget),"framework_parameters":parsed_params,"drift":{**_as_dict(config.get("drift")),"monitoring_enabled":bool(drift_enabled),"detector":detector if drift_enabled else None,"sensitivity":drift_sensitivity},"fairness":{**_as_dict(config.get("fairness")),"requested":bool(fairness_enabled),"status":fairness_status,"sensitive_attribute":state.get("sensitive"),"metric":fairness_metric,"audit_only":True},"explainability":{**_as_dict(config.get("explainability")),"level":explain_level,"method":explain_method},"sustainability":{**_as_dict(config.get("sustainability")),"enabled":bool(track_energy or track_co2),"track_energy":bool(track_energy),"track_co2":bool(track_co2)}}
    note=st.text_input("Final review note (optional)",key="goal_v2_final_review_note",placeholder="Example: Approved with fairness audit and SHAP explanations enabled.")
    if st.button("Save final plan decision",type="primary",key="goal_v2_save_final_review",disabled=(mode=="Approve with edits" and not edit_validation_ok)):
        decision_map={"Approve plan":"approved","Approve with edits":"approved_with_edits","Reject plan":"rejected"}
        try:
            review=review_proposal(proposal_obj,decision=decision_map[mode],edits=edits,note=note or None)
            ReviewStore(ROOT/"artifacts"/"copilot"/"reviews.jsonl").append(proposal_obj,review)
            state["copilot_review"]=review.model_dump(); state["copilot_unified_flash"]="Final plan decision saved: {}.".format(review.decision.replace("_"," ").title()); st.rerun()
        except Exception as exc: st.error("The final plan decision could not be saved: {}".format(exc))
    saved=_as_dict(state.get("copilot_review"))
    if saved: st.caption("Saved final decision: {}".format(str(saved.get("decision") or "").replace("_"," ").title()))
    _render_post_approval_state(state)

def _render_plan(proposal: Mapping[str, Any], state: Dict[str, Any], parse_meta: Mapping[str, Any], has_observed_run: bool) -> None:
    proposal=_as_dict(proposal); config=_as_dict(proposal.get("proposed_config")); interpretation=_as_dict(proposal.get("interpretation"))
    selected=list(_selected_objectives(interpretation)); weights=_weights(interpretation); ranked=_ranked_frame(state)
    framework=str(proposal.get("ml_recommender_framework") or config.get("framework") or "N/A"); algorithm=str(config.get("algorithm") or "N/A")
    rank=int(proposal.get("ml_recommender_rank") or 1); utility=proposal.get("ml_recommender_utility")
    saved=_as_dict(state.get("copilot_review")); review_status={"approved":"APPROVED","approved_with_edits":"APPROVED WITH EDITS","rejected":"REJECTED"}.get(saved.get("decision"),"REVIEW REQUIRED")
    st.markdown("## 3 · Evidence-backed recommendation brief")
    if has_observed_run:
        st.markdown('<div class="awareml-decision-banner"><b>Observed benchmark available.</b><br>This recommendation remains the dataset-aware prediction path from the scenario and meta-profile. Measured post-run evidence is available separately in Decision Lab and Streaming Observatory for validation.</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="awareml-decision-banner"><b>Dataset-aware recommendation.</b><br>ML Recommender combines the active priorities with the dataset meta-profile to rank the five candidate frameworks.</div>',unsafe_allow_html=True)
    cards=st.columns(4); card_data=[("Recommended framework",framework),("Predicted rank","#{} of 5".format(rank)),("Ranking utility","—" if utility is None else "{:.4f}".format(float(utility))),("Review status",review_status)]
    for col,(label,value) in zip(cards,card_data):
        with col:
            st.markdown('<div class="awareml-summary-card"><div class="awareml-summary-label">{}</div><div class="awareml-summary-value">{}</div></div>'.format(label,value),unsafe_allow_html=True)
    st.caption("Ranking utility is a relative preference score among the five frameworks under the current weights. It is not a probability, correctness score or model confidence.")
    st.markdown("### Decision basis")
    basis=st.columns(3)
    with basis[0]:
        with st.container(border=True):
            st.markdown("**Active priorities**"); st.write(_priority_summary(selected,weights)); st.caption("Human-corrected priorities are active." if state.get("copilot_human_corrected_objectives") else "Derived from the current natural-language interpretation.")
    with basis[1]:
        with st.container(border=True):
            st.markdown("**Dataset context**"); st.write(str(state.get("dataset_name") or "Loaded dataset")); st.caption("Target: {}".format(state.get("target") or "not selected"))
    with basis[2]:
        with st.container(border=True):
            st.markdown("**Decision source**"); st.write("ML Recommender"); st.caption("The LLM interprets the goal; it does not choose the framework.")
    rationale=humanize_rationale_text(clean_pre_run_rationale(proposal.get("rationale")))
    drift=_as_dict(config.get("drift")); fairness=_as_dict(config.get("fairness")); xai=_as_dict(config.get("explainability")); sustain=_as_dict(config.get("sustainability")); fairness_constraint,fairness_audit=_fairness_plan_text(fairness,state)
    st.markdown("### Recommendation evidence and execution plan")
    left,right=st.columns([1.05,.95])
    with left:
        with st.container(border=True):
            st.markdown("**Why this framework is ranked first**")
            if utility is None:
                st.write("ML Recommender compared all five frameworks using the dataset meta-profile and the active priorities ({}). {} is currently predicted rank #1.".format(_priority_summary(selected,weights),framework))
            else:
                st.write("ML Recommender compared all five frameworks using the dataset meta-profile and the active priorities ({}). {} has the highest current ranking utility ({:.4f}) and is predicted rank #1.".format(_priority_summary(selected,weights),framework,float(utility)))
            if rationale: st.caption(rationale)
            evidence=_prediction_table(framework,selected,weights,ranked)
            if not evidence.empty:
                st.markdown("**Predicted outcomes under the active priorities**"); st.dataframe(evidence,use_container_width=True,hide_index=True)
    with right:
        with st.container(border=True):
            st.markdown("**Approved-plan configuration**")
            rows=[
                {"Setting":"Framework","Value":framework},{"Setting":"Algorithm","Value":algorithm},{"Setting":"Window size","Value":config.get("window_size","N/A")},{"Setting":"Time budget","Value":"{} s".format(config.get("time_budget_sec","N/A"))},{"Setting":"Drift monitoring","Value":drift.get("detector") or "N/A"},{"Setting":"Fairness constraint","Value":fairness_constraint},{"Setting":"Explainability","Value":xai.get("method") or xai.get("level") or "N/A"},{"Setting":"Energy / CO₂ tracking","Value":"Enabled" if sustain.get("track_energy") or sustain.get("track_co2") else "Not requested"},
            ]
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True); st.caption("Fairness audit: {}".format(fairness_audit))
    if not ranked.empty and {"framework","utility"}.issubset(ranked.columns):
        with st.expander("Compare all predicted alternatives",expanded=False):
            top=ranked.sort_values("rank",ascending=True).head(5).copy() if "rank" in ranked.columns else ranked.sort_values("utility",ascending=False).head(5).copy(); cols=[c for c in ("rank","framework","utility") if c in top.columns]; top=top[cols].rename(columns={"rank":"Predicted rank","framework":"Framework","utility":"Ranking utility"}); st.dataframe(top,use_container_width=True,hide_index=True)
    with st.expander("Technical recommendation evidence",expanded=False):
        st.markdown("**Evidence provenance**")
        provenance=pd.DataFrame([
            {"Evidence layer":"Goal interpretation","Source":"LLaMA 3 8B + V3.3 hybrid evidence guard","Used for":"Objective selection only"},{"Evidence layer":"Objective weights","Source":"equal_selected_v1","Used for":"Preference weighting"},{"Evidence layer":"Dataset evidence","Source":"Dataset meta-profile","Used for":"ML Recommender prediction"},{"Evidence layer":"Framework recommendation","Source":"ML Recommender","Used for":"Ranked framework recommendation"},{"Evidence layer":"Human oversight","Source":"Objective + plan review logs","Used for":"Correction / approval / rejection"},
        ]); st.dataframe(provenance,use_container_width=True,hide_index=True)
        st.markdown("**Grounded recommendation rationale**"); st.write(rationale or "No additional rationale was returned.")
        keys=list(proposal.get("evidence_keys") or [])
        if keys:
            st.markdown("**Evidence identifiers**"); st.caption("These identifiers link the explanation back to the structured recommendation evidence used by AwareML."); st.code("\n".join(str(k) for k in keys),language="text")
        warnings=[]
        for item in list(parse_meta.get("warnings") or []) + list(proposal.get("warnings") or []):
            text=str(item)
            if text not in warnings: warnings.append(text)
        if warnings:
            st.markdown("**Interpretation / proposal notes**")
            for warning in warnings: st.info(warning)
        st.markdown("**Configuration provenance**"); status=pd.DataFrame([{"Field":"Framework","Value":framework},{"Field":"Algorithm","Value":algorithm},{"Field":"Objective weights","Value":_priority_summary(selected,weights)},{"Field":"Fairness audit","Value":fairness_audit},{"Field":"Recommendation status","Value":review_status}]); st.dataframe(status,use_container_width=True,hide_index=True)
        st.markdown("**Raw supported configuration**"); st.json(config,expanded=False)
    _render_objective_review(state,selected,step_number=4); _render_final_plan_review(state,proposal)

def render_goal_copilot_unified_page() -> None:
    state=ensure_research_state(); has_dataset=bool(dataset_ready()); has_observed_run=bool(state.get("run_results")); state.pop("copilot_view_mode",None); _render_css()
    flash=state.pop("copilot_unified_flash",None)
    if flash: st.success(str(flash))
    _render_step_cards()  # navigation cards already expose dataset/evidence readiness
    st.markdown("## Describe your deployment goal")
    goal=st.text_area("Streaming AutoML scenario",value=state.get("copilot_goal") or "Suitable for deployment in a low-impact edge environment while still providing strong performance.",height=135,key="goal_v3_text",placeholder="Example: The service will run on a battery-powered edge device and needs dependable predictions with a small environmental footprint.")
    state["copilot_goal"]=goal
    c1,c2=st.columns([.65,.35])
    with c1:
        use_llm=st.toggle("Use hybrid evidence-grounded LLaMA 3 8B selector V3.3",value=True,key="goal_v3_use_llm"); st.caption("Optimization objectives used by ML Recommender: Accuracy · Runtime · Energy · CO₂. Fairness, drift and explainability are tracked separately as HCAI oversight requirements; they do not receive recommender utility weight in the current journal protocol.")
    with c2:
        st.caption("Recommendation source"); st.markdown("**ML Recommender** when dataset context is available")
    if st.button("Generate Copilot plan",type="primary",use_container_width=True,key="goal_v3_generate"):
        _clear_scenario_state(state)
        try:
            service=CopilotService(recommender=(load_v2_recommender() if has_dataset else None),goal_parser=GoalParser(selector=EvidenceGroundedObjectiveSelectorV33(root=ROOT)),chat=GroundedCopilotChat(client=OllamaClient(model=EXACT_MODEL)),review_store=ReviewStore(ROOT / "artifacts" / "copilot" / "reviews.jsonl"))
            if has_dataset:
                proposal,ranked,evidence,meta=service.propose_from_dataframe(goal=goal,df=state["dataset"],target=state["target"],sensitive_attribute=state.get("sensitive"),current_config=None,use_llm=use_llm); state["copilot_proposal"]=proposal; state["copilot_ranked"]=ranked; state["copilot_evidence"]=evidence; state["copilot_meta"]=meta; state["copilot_review"]=None; state.pop("copilot_context_free_interpretation",None); state.pop("copilot_context_free_meta",None); state["copilot_unified_flash"]="Dataset-aware Goal Copilot plan generated. Review the interpreted priorities and evidence-backed recommendation below."
            else:
                interpretation,meta=service.interpret_goal(goal,use_llm=use_llm); state["copilot_context_free_interpretation"]=interpretation; state["copilot_context_free_meta"]=meta
                for key in ("copilot_proposal","copilot_ranked","copilot_evidence","copilot_meta","copilot_review"): state.pop(key,None)
                state["copilot_unified_flash"]="Goal interpreted. Add dataset context when you want a framework recommendation."
            st.rerun()
        except Exception as exc:
            error_name=type(exc).__name__; message="AwareML could not identify a sufficiently supported objective set from this wording." if error_name=="GoalSelectionError" else "AwareML could not generate a new proposal from this scenario."; detail="Add clearer deployment priorities such as prediction reliability, response speed, battery or power limits, or environmental impact." if error_name=="GoalSelectionError" else "Review the wording and model/runtime status before retrying."; set_copilot_clarification(state,message=message,detail=detail,error_type=error_name); st.rerun()
    if render_copilot_clarification(state): return
    context_free=state.get("copilot_context_free_interpretation"); context_free_meta=state.get("copilot_context_free_meta") or {}; proposal=state.get("copilot_proposal")
    if proposal is None and context_free is not None:
        selected=_render_understanding(context_free,context_free_meta.get("goal_parse") or {},state); _render_decision_provenance(False); render_goal_framework_guidance(state,context_free,None); _render_objective_review(state,selected,step_number=3); _render_context_free_next_step(); return
    if proposal is None:
        st.caption("Enter a deployment scenario and choose **Generate Copilot plan** to begin."); return
    proposal_dict=proposal.model_dump() if hasattr(proposal,"model_dump") else dict(proposal); interpretation=proposal_dict.get("interpretation") or {}; parse_meta=((state.get("copilot_meta") or {}).get("goal_parse") or {})
    _render_understanding(interpretation,parse_meta,state); _render_decision_provenance(True); render_goal_framework_guidance(state,interpretation,proposal_dict); _render_plan(proposal_dict,state,parse_meta,has_observed_run)
