from __future__ import annotations

import copy
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from awareml.llm.configuration import synthesize_configuration
from awareml.llm.schemas import PrimaryObjectiveWeights
from awareml.llm.review import ReviewStore, review_proposal
from awareml.recommender.v2_profile import profile_from_dataframe_v2

from .data import load_v2_recommender
from .state import ROOT, dataset_signature, ensure_research_state, result_dicts

CANONICAL_OBJECTIVES = ["Accuracy", "Runtime", "Energy", "CO2"]
FAIRNESS_METRICS = {
    "Demographic parity": "dp_diff",
    "Equal opportunity": "equal_opportunity_diff",
    "Equalized odds": "equalized_odds_gap",
    "Predictive parity": "predictive_parity_diff",
    "Error-rate parity": "error_rate_gap",
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


def _normalized_weights(values: Mapping[str, Any]) -> Dict[str, float]:
    out = {
        "accuracy": max(0.0, float(values.get("accuracy", 0.0) or 0.0)),
        "runtime": max(0.0, float(values.get("runtime", 0.0) or 0.0)),
        "energy": max(0.0, float(values.get("energy", 0.0) or 0.0)),
        "co2": max(0.0, float(values.get("co2", 0.0) or 0.0)),
    }
    total = float(sum(out.values()))
    if total <= 0:
        return {"accuracy": 0.55, "runtime": 0.15, "energy": 0.15, "co2": 0.15}
    return {k: v / total for k, v in out.items()}


def weights_for_objectives(selected: Sequence[str]) -> Dict[str, float]:
    selected = [x for x in CANONICAL_OBJECTIVES if x in set(selected or [])]
    if not selected:
        return {"accuracy": 0.0, "runtime": 0.0, "energy": 0.0, "co2": 0.0}
    share = 1.0 / float(len(selected))
    return {
        "accuracy": share if "Accuracy" in selected else 0.0,
        "runtime": share if "Runtime" in selected else 0.0,
        "energy": share if "Energy" in selected else 0.0,
        "co2": share if "CO2" in selected else 0.0,
    }


def clean_pre_run_rationale(text: Any) -> str:
    """Humanize pre-run rationale text while keeping technical evidence separately auditable."""
    value = str(text or "").strip()
    replacements = {
        "current empirical top-ranked framework": "current predicted top-ranked framework from ML Recommender V2",
        "empirical top-ranked framework": "predicted top-ranked framework from ML Recommender V2",
        "current empirical top-ranked": "current predicted top-ranked",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    # Evidence IDs are useful for Research View, but noisy and confusing in beginner-facing prose.
    value = re.sub(r"\s*\[evidence\.[^\]]+\]", "", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()

def copilot_weights_from_state(state: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    proposal = state.get("copilot_proposal")
    if proposal is not None:
        data = _as_dict(proposal)
        interpretation = _as_dict(data.get("interpretation"))
        weights = _as_dict(interpretation.get("primary_weights"))
        if weights:
            return _normalized_weights(weights)
    interpretation = state.get("copilot_context_free_interpretation")
    if interpretation is not None:
        data = _as_dict(interpretation)
        weights = _as_dict(data.get("primary_weights"))
        if weights:
            return _normalized_weights(weights)
    override = state.get("copilot_human_corrected_weights")
    if isinstance(override, dict):
        return _normalized_weights(override)
    return None


def preference_context(state: Dict[str, Any], current: Mapping[str, float]) -> Tuple[Dict[str, float], bool]:
    copilot = copilot_weights_from_state(state)
    options = ["Manual exploration"]
    if copilot is not None:
        options.insert(0, "Use Copilot priorities")
    default = state.get("decision_space_preference_source")
    if default not in options:
        default = options[0]
    source = st.segmented_control(
        "Preference source",
        options,
        default=default,
        key="pre14_decision_preference_source",
        help=(
            "Use Copilot priorities copies the latest reviewed/interpreted Accuracy/Runtime/Energy/CO₂ weights. "
            "Manual exploration is a what-if analysis and can intentionally differ from Copilot."
        ),
    ) or default
    state["decision_space_preference_source"] = source
    if source == "Use Copilot priorities" and copilot is not None:
        current = dict(copilot)
        state["preference_weights"] = dict(copilot)
        # Streamlit widget keys otherwise retain stale manual slider values across reruns.
        st.session_state["r9_w_accuracy"] = int(round(current["accuracy"] * 100))
        st.session_state["r9_w_runtime"] = int(round(current["runtime"] * 100))
        st.session_state["r9_w_energy"] = int(round(current["energy"] * 100))
        st.session_state["r9_w_co2"] = int(round(current["co2"] * 100))
        st.info(
            "3D Decision Space is using the latest Copilot objective priorities. "
            "Switch to Manual exploration for independent what-if analysis."
        )
        return dict(copilot), False
    st.caption("Manual what-if preferences — these can intentionally differ from the Copilot proposal.")
    return _normalized_weights(current), True


def quick_framework_selector(frameworks: Sequence[str], state: Dict[str, Any]) -> str:
    frameworks = [str(x) for x in frameworks]
    current = state.get("selected_framework")
    if current not in frameworks:
        current = frameworks[0]
    selected = st.segmented_control(
        "Quick inspect framework",
        frameworks,
        default=current,
        key="pre14_quick_inspect_framework",
        help="One click changes the framework whose predicted metrics are shown; it does not rerun AutoML.",
    ) or current
    state["selected_framework"] = selected
    return str(selected)


def render_decision_space_context(ranked: pd.DataFrame, state: Dict[str, Any], selected: str) -> None:
    if ranked is None or ranked.empty:
        return
    top = ranked.iloc[0]
    selected_row = ranked[ranked["framework"].astype(str).eq(str(selected))].iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**RECOMMENDED · pre-run prediction**")
        st.metric(str(top["framework"]), "Predicted rank #{}".format(int(top.get("rank", 1))))
    with c2:
        st.markdown("**CURRENTLY INSPECTING**")
        st.metric(str(selected), "Predicted rank #{}".format(int(selected_row.get("rank", 0))))
    st.caption(
        "Normalized preference utility is a relative score among the five candidates under the current weights; "
        "it is not a probability or model-confidence score."
    )


def _interpretation_selected(interpretation: Any) -> List[str]:
    data = _as_dict(interpretation)
    return [x for x in CANONICAL_OBJECTIVES if x in set(data.get("selected_objectives") or [])]


def _interpretation_weights(interpretation: Any) -> Dict[str, float]:
    data = _as_dict(interpretation)
    return _normalized_weights(_as_dict(data.get("primary_weights")))


def _objective_review_path() -> Path:
    return ROOT / "artifacts" / "copilot" / "objective_reviews.jsonl"


def persist_objective_review(
    state: Mapping[str, Any],
    decision: str,
    selected_seen: Sequence[str],
    corrected: Optional[Sequence[str]] = None,
    note: Optional[str] = None,
    source: str = "copilot_simple_view",
) -> Dict[str, Any]:
    record = {
        "record_type": "copilot_objective_review",
        "review_id": "OR-{}".format(uuid.uuid4().hex[:10].upper()),
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scenario": str(state.get("copilot_goal") or ""),
        "selector": "hybrid_evidence_grounded_v31",
        "model": "llama3:8b",
        "decision": str(decision),
        "selected_objectives_seen": list(selected_seen or []),
        "corrected_objectives": list(corrected or []) if corrected is not None else None,
        "note": note or None,
        "dataset_name": state.get("dataset_name"),
        "dataset_signature": dataset_signature(dict(state)) if state.get("dataset") is not None else None,
        "target": state.get("target"),
        "sensitive_attribute": state.get("sensitive"),
        "benchmark_ground_truth": False,
        "source": source,
    }
    path = _objective_review_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return record


def _set_interpretation_objectives(interpretation: Any, selected: Sequence[str]) -> Any:
    selected = [x for x in CANONICAL_OBJECTIVES if x in set(selected or [])]
    weights = weights_for_objectives(selected)
    pweights = PrimaryObjectiveWeights(**weights)
    if isinstance(interpretation, dict):
        interpretation = dict(interpretation)
        interpretation["selected_objectives"] = selected
        interpretation["primary_weights"] = weights
        interpretation["selection_source"] = "human_override"
        return interpretation
    interpretation.selected_objectives = list(selected)
    interpretation.primary_weights = pweights
    interpretation.selection_source = "human_override"
    return interpretation


def _profile_for_state(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    profile = state.get("v2_profile")
    if profile is not None:
        if isinstance(profile, dict):
            return dict(profile)
        if hasattr(profile, "model_dump"):
            return profile.model_dump()
        try:
            return dict(profile)
        except Exception:
            pass
    df = state.get("dataset")
    target = state.get("target")
    if df is None or not target:
        return None
    return profile_from_dataframe_v2(
        df,
        target=target,
        window_size=int(state.get("window_size") or 1000),
        time_budget_sec=float(state.get("time_budget_sec") or 60.0),
        dataset_family="unknown",
        source_type="uploaded" if state.get("dataset_name") else "unknown",
        drift_type="unknown",
    )


def apply_human_objective_override(state: Dict[str, Any], corrected: Sequence[str]) -> Optional[str]:
    corrected = [x for x in CANONICAL_OBJECTIVES if x in set(corrected or [])]
    if not corrected:
        raise ValueError("At least one objective is required for a human correction.")
    weights = weights_for_objectives(corrected)
    state["copilot_human_corrected_objectives"] = list(corrected)
    state["copilot_human_corrected_weights"] = dict(weights)

    context_free = state.get("copilot_context_free_interpretation")
    if context_free is not None:
        state["copilot_context_free_interpretation"] = _set_interpretation_objectives(
            context_free, corrected
        )

    proposal = state.get("copilot_proposal")
    if proposal is None:
        return None

    profile = _profile_for_state(state)
    if profile is None:
        return None

    recommender = load_v2_recommender()
    ranked, ranking_meta = recommender.recommend_profile(
        dict(profile),
        weights=weights,
        ranking_mode="point",
        coverage=0.90,
    )
    if ranked is None or ranked.empty:
        return None

    new_proposal = proposal.model_copy(deep=True) if hasattr(proposal, "model_copy") else copy.deepcopy(proposal)
    interpretation = getattr(new_proposal, "interpretation", None)
    if interpretation is None:
        return None
    interpretation = _set_interpretation_objectives(interpretation, corrected)
    new_proposal.interpretation = interpretation
    config, warnings = synthesize_configuration(
        interpretation,
        ranked,
        sensitive_attribute=state.get("sensitive"),
    )
    top = ranked.iloc[0]
    new_proposal.proposed_config = config
    new_proposal.ml_recommender_framework = str(top["framework"])
    new_proposal.ml_recommender_rank = int(top.get("rank", 1))
    new_proposal.ml_recommender_utility = float(top["utility"]) if pd.notna(top.get("utility")) else None
    new_proposal.rationale = (
        "Human review corrected the objective set to {}. ML Recommender V2 then reranked the same dataset meta-profile "
        "under the corrected weights and predicts {} as rank #1.".format(
            ", ".join(corrected), str(top["framework"])
        )
    )
    new_proposal.warnings = list(getattr(new_proposal, "warnings", []) or []) + list(warnings or [])
    state["copilot_proposal"] = new_proposal
    state["copilot_ranked"] = ranked
    state["copilot_meta"] = dict(state.get("copilot_meta") or {})
    state["copilot_meta"]["human_objective_override"] = {
        "selected_objectives": corrected,
        "weights": weights,
        "ranking_meta": ranking_meta,
    }
    state["copilot_review"] = None
    return str(top["framework"])


def _render_simple_objective_review(
    state: Dict[str, Any],
    selected: Sequence[str],
    step_number: int = 4,
) -> None:
    st.markdown("### {} · Review the objectives".format(step_number))
    st.caption(
        "The objective interpretation is advisory. Accept it, correct the priorities, or reject it before final approval."
    )
    choice = st.segmented_control(
        "Does AwareML understand your priorities correctly?",
        ["Accept", "Correct objectives", "Reject"],
        default="Accept",
        key="pre14_simple_objective_review_choice",
    ) or "Accept"
    corrected = None
    if choice == "Correct objectives":
        corrected = st.multiselect(
            "Which objectives should be used?",
            CANONICAL_OBJECTIVES,
            default=list(selected),
            key="pre14_corrected_objectives",
            help="After a dataset is loaded, changing this set reranks ML Recommender V2 without asking LLaMA again.",
        )
    note = st.text_input(
        "Optional review note",
        key="pre14_objective_review_note",
        placeholder="Example: Energy matters because this service must preserve battery life.",
    )
    if st.button("Save objective review", key="pre14_save_objective_review"):
        if choice == "Correct objectives" and not corrected:
            st.warning("Choose at least one corrected objective before saving.")
            return
        final_set = list(corrected or selected)
        record = persist_objective_review(
            state,
            decision=choice,
            selected_seen=selected,
            corrected=(final_set if choice == "Correct objectives" else None),
            note=note or None,
        )
        new_top = None
        if choice == "Correct objectives":
            new_top = apply_human_objective_override(state, final_set)
        state["copilot_objective_review_persisted"] = record
        if choice == "Correct objectives" and new_top:
            st.success(
                "Objective review saved as {}. Human correction applied; ML Recommender V2 reranked the same dataset profile and now predicts {} as #1.".format(
                    record["review_id"], new_top
                )
            )
            st.rerun()
        elif choice == "Correct objectives":
            st.success(
                "Objective review saved as {}. The corrected objective set is stored; framework reranking becomes available once a dataset profile exists.".format(
                    record["review_id"]
                )
            )
            st.rerun()
        else:
            st.success("Objective review saved as {} in the append-only Copilot audit trail.".format(record["review_id"]))

    record = state.get("copilot_objective_review_persisted")
    if isinstance(record, dict):
        st.caption(
            "Saved review: {} · {} · benchmark ground truth: no".format(
                record.get("review_id"), record.get("decision")
            )
        )


def render_accessible_objective_interpretation(
    research_renderer,
    interpretation: Any,
    parse_meta: Mapping[str, Any],
    state: Dict[str, Any],
) -> None:
    view = st.segmented_control(
        "Copilot view",
        ["Simple View", "Research View"],
        default=state.get("copilot_view_mode") if state.get("copilot_view_mode") in {"Simple View", "Research View"} else "Simple View",
        key="pre14_copilot_view_mode",
        help="Simple View explains the decision path in plain language. Research View exposes frozen baselines, development diagnostics and technical audit fields.",
    ) or "Simple View"
    state["copilot_view_mode"] = view
    if view == "Research View":
        st.info(
            "Research View exposes the complete objective-selection evidence, including the frozen Phase-12 baseline and V3/V3.1 development diagnostics."
        )
        # Important: render directly at page level. The research renderer already contains
        # expanders, and Streamlit does not allow expanders to be nested.
        research_renderer(interpretation, parse_meta, state)
        return

    selected = _interpretation_selected(interpretation)
    weights = _interpretation_weights(interpretation)
    st.markdown("## 1 · What AwareML understood")
    if selected:
        st.success("Your scenario currently maps to: **{}**".format(" + ".join(selected)))
    else:
        st.warning("No supported objective set is currently available; clarification is required.")
    cols = st.columns(4)
    labels = [("Accuracy", "accuracy"), ("Runtime", "runtime"), ("Energy", "energy"), ("CO₂", "co2")]
    for col, (label, key) in zip(cols, labels):
        with col:
            st.metric(label, "{:.3f}".format(float(weights.get(key, 0.0))))
    st.caption(
        "Selected objectives share the total preference weight equally. These weights do not come from the dataset and are not confidence scores."
    )

    st.markdown("### 2 · How AwareML makes the decision")
    provenance = pd.DataFrame(
        [
            {"Decision": "Understand the goal", "Source": "LLaMA 3 8B + V3.1 evidence guard", "Role": "Interprets the user's wording; does not choose a framework."},
            {"Decision": "Set objective priorities", "Source": "equal_selected_v1", "Role": "Applies the documented equal-weight policy to selected objectives."},
            {"Decision": "Predict the framework", "Source": "ML Recommender V2", "Role": "Uses objective weights + dataset meta-profile; requires a dataset."},
            {"Decision": "Choose supported algorithm / defaults", "Source": "Configuration synthesizer", "Role": "Maps the predicted framework to validated implementation defaults."},
            {"Decision": "Final action", "Source": "You", "Role": "Accept, correct, edit or reject the proposal."},
        ]
    )
    st.dataframe(provenance, use_container_width=True, hide_index=True)

    # In context-free mode there is no AutoML plan yet, so objective review is step 3.
    # With a dataset, review is shown after the plan to keep the beginner flow 1→2→3→4→5.
    if state.get("copilot_proposal") is None:
        _render_simple_objective_review(state, selected, step_number=3)
        st.caption("Need benchmark diagnostics or the full V3.1 audit? Switch to Research View above.")


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


def _priority_summary(selected: Sequence[str], weights: Mapping[str, float]) -> str:
    key_map = {"Accuracy": "accuracy", "Runtime": "runtime", "Energy": "energy", "CO2": "co2"}
    parts = []
    for objective in selected:
        key = key_map.get(objective)
        if key:
            parts.append("{} {:.0%}".format("CO₂" if objective == "CO2" else objective, float(weights.get(key, 0.0))))
    return " · ".join(parts) if parts else "no active objective weights"


def _priority_prediction_table(framework: str, selected: Sequence[str], weights: Mapping[str, float], ranked: pd.DataFrame) -> pd.DataFrame:
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
    key_map = {"Accuracy": "accuracy", "Runtime": "runtime", "Energy": "energy", "CO2": "co2"}
    rows = []
    for objective in selected:
        metric, direction, formatter = specs[objective]
        value = row.get(metric)
        if value is None or pd.isna(value):
            rendered = "N/A"
        else:
            rendered = formatter(value)
        rows.append({
            "Priority": "CO₂" if objective == "CO2" else objective,
            "Your weight": "{:.0%}".format(float(weights.get(key_map[objective], 0.0))),
            "Predicted outcome": rendered,
            "Direction": direction,
        })
    return pd.DataFrame(rows)


def _fairness_plan_text(fairness: Mapping[str, Any], state: Mapping[str, Any]) -> Tuple[str, str]:
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
        audit = "Post-run fairness audit available for '{}'".format(sensitive)
    else:
        audit = "Select a sensitive attribute in Run Studio to audit fairness"
    return constraint, audit


def _render_simple_full_proposal_review(state: Dict[str, Any], proposal: Mapping[str, Any]) -> None:
    st.markdown("### 5 · Final plan decision")
    st.caption(
        "Approve the proposed configuration, make limited execution edits, or reject it. This decision is stored in the append-only Copilot review log."
    )
    proposal_obj = state.get("copilot_proposal")
    if proposal_obj is None:
        st.warning("No dataset-aware proposal is available to review.")
        return
    config = _as_dict(_as_dict(proposal).get("proposed_config"))
    mode = st.segmented_control(
        "What do you want to do with this plan?",
        ["Approve plan", "Approve with edits", "Reject plan"],
        default="Approve plan",
        key="pre14_simple_full_plan_review_mode",
    ) or "Approve plan"
    edits = None
    if mode == "Approve with edits":
        c1, c2 = st.columns(2)
        with c1:
            window = st.number_input(
                "Window size",
                min_value=50,
                max_value=100000,
                value=int(config.get("window_size", 1000)),
                step=50,
                key="pre14_simple_review_window",
            )
        with c2:
            budget = st.number_input(
                "Time budget (seconds)",
                min_value=1.0,
                max_value=86400.0,
                value=float(config.get("time_budget_sec", 60.0)),
                step=5.0,
                key="pre14_simple_review_budget",
            )
        edits = {"window_size": int(window), "time_budget_sec": float(budget)}
    note = st.text_input(
        "Final review note (optional)",
        key="pre14_simple_full_plan_review_note",
        placeholder="Example: Approved for the next benchmark run.",
    )
    if st.button("Save final plan decision", key="pre14_simple_save_full_plan_review"):
        decision_map = {
            "Approve plan": "approved",
            "Approve with edits": "approved_with_edits",
            "Reject plan": "rejected",
        }
        try:
            review = review_proposal(
                proposal_obj,
                decision=decision_map[mode],
                edits=edits,
                note=note or None,
            )
            ReviewStore(ROOT / "artifacts" / "copilot" / "reviews.jsonl").append(proposal_obj, review)
            state["copilot_review"] = review.model_dump()
            st.success(
                "Final plan decision saved: {}. Proposal ID: {}.".format(
                    review.decision.replace("_", " ").title(), str(review.proposal_id)[:12]
                )
            )
            st.rerun()
        except Exception as exc:
            st.error("The final plan decision could not be saved: {}".format(exc))
    saved = state.get("copilot_review")
    if isinstance(saved, dict):
        st.caption("Saved final decision: {}".format(str(saved.get("decision") or "").replace("_", " ").title()))


def render_copilot_plan_summary(proposal: Mapping[str, Any], state: Dict[str, Any], has_observed_run: bool) -> None:
    proposal = _as_dict(proposal)
    config = _as_dict(proposal.get("proposed_config"))
    interpretation = _as_dict(proposal.get("interpretation"))
    selected = interpretation.get("selected_objectives") or []
    weights = _normalized_weights(_as_dict(interpretation.get("primary_weights")))
    framework = str(proposal.get("ml_recommender_framework") or config.get("framework") or "N/A")
    algorithm = str(config.get("algorithm") or "N/A")
    utility = proposal.get("ml_recommender_utility")
    rank = int(proposal.get("ml_recommender_rank") or 1)
    ranked = _ranked_frame(state)

    st.markdown("## 3 · Copilot AutoML Plan")
    st.info(
        "This is a **pre-run, reviewable proposal**. LLaMA/V3.1 interprets your objectives; ML Recommender V2 predicts the framework from the dataset meta-profile; "
        "the configuration synthesizer supplies supported algorithm/configuration defaults; you make the final decision."
    )
    cards = st.columns(4)
    with cards[0]:
        st.metric("Predicted framework", framework)
    with cards[1]:
        st.metric("Predicted rank", "#{} of 5".format(rank))
    with cards[2]:
        st.metric(
            "Normalized preference utility",
            "—" if utility is None else "{:.4f}".format(float(utility)),
        )
    with cards[3]:
        saved_review = _as_dict(state.get("copilot_review"))
        status_map = {
            "approved": "APPROVED",
            "approved_with_edits": "APPROVED WITH EDITS",
            "rejected": "REJECTED",
        }
        st.metric("Proposal status", status_map.get(saved_review.get("decision"), "REVIEW REQUIRED"))
    st.caption(
        "Normalized preference utility is a relative ranking score under the current preferences. It is not a probability, correctness score or model confidence."
    )

    st.markdown("### Why this framework fits your priorities")
    priority_text = _priority_summary(selected, weights)
    if utility is None:
        st.info(
            "ML Recommender V2 compared the five candidate frameworks using the current dataset meta-profile and your priorities ({}). {} is currently predicted rank #1.".format(
                priority_text, framework
            )
        )
    else:
        st.info(
            "ML Recommender V2 compared the five candidate frameworks using the current dataset meta-profile and your priorities ({}). "
            "{} has the highest normalized preference utility ({:.4f}) and is therefore predicted rank #1.".format(
                priority_text, framework, float(utility)
            )
        )
    st.caption(
        "The LLM did not choose {}. It only helped interpret your goal; the framework ranking comes from ML Recommender V2.".format(framework)
    )

    evidence = _priority_prediction_table(framework, selected, weights, ranked)
    if not evidence.empty:
        st.markdown("**Predicted evidence for the priorities you selected**")
        st.dataframe(evidence, use_container_width=True, hide_index=True)

    if not ranked.empty and {"framework", "utility"}.issubset(ranked.columns):
        top3 = ranked.sort_values("rank" if "rank" in ranked.columns else "utility", ascending=True if "rank" in ranked.columns else False).head(3).copy()
        cols = [c for c in ["rank", "framework", "utility"] if c in top3.columns]
        top3 = top3[cols].rename(columns={"rank": "Predicted rank", "framework": "Framework", "utility": "Normalized utility"})
        st.markdown("**Top predicted alternatives**")
        st.dataframe(top3, use_container_width=True, hide_index=True)

    drift = _as_dict(config.get("drift"))
    fairness = _as_dict(config.get("fairness"))
    xai = _as_dict(config.get("explainability"))
    sustain = _as_dict(config.get("sustainability"))
    fairness_constraint, fairness_audit = _fairness_plan_text(fairness, state)
    rows = [
        {"Setting": "Framework", "Proposed value": framework},
        {"Setting": "Algorithm", "Proposed value": algorithm},
        {"Setting": "Window size", "Proposed value": config.get("window_size", "N/A")},
        {"Setting": "Time budget", "Proposed value": "{} s".format(config.get("time_budget_sec", "N/A"))},
        {"Setting": "Drift monitoring", "Proposed value": drift.get("detector") or "N/A"},
        {"Setting": "Fairness constraint", "Proposed value": fairness_constraint},
        {"Setting": "Fairness audit", "Proposed value": fairness_audit},
        {"Setting": "Explainability", "Proposed value": xai.get("method") or xai.get("level") or "N/A"},
        {"Setting": "Energy / CO₂ tracking", "Proposed value": "Enabled" if sustain.get("track_energy") or sustain.get("track_co2") else "Not requested"},
    ]
    st.markdown("### What will run if you approve")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if not bool(fairness.get("requested")):
        st.caption(
            "Fairness constraint is 'Not requested by this goal' because the scenario did not ask AwareML to optimize or constrain fairness. "
            "This does not disable post-run fairness analysis: if a sensitive attribute is configured in Run Studio, fairness evidence can still be audited after execution."
        )

    if state.get("copilot_human_corrected_objectives"):
        st.success(
            "Human objective correction is active: **{}**. The current framework ranking reflects the corrected weights.".format(
                " + ".join(state.get("copilot_human_corrected_objectives") or [])
            )
        )
    if has_observed_run:
        st.caption(
            "A benchmark has already been executed, but this Copilot plan remains a pre-run prediction. Observed evidence is kept separate in Streaming Observatory and Decision Lab."
        )

    if state.get("copilot_view_mode") == "Simple View":
        _render_simple_objective_review(state, selected, step_number=4)
        _render_simple_full_proposal_review(state, proposal)
        st.caption("Need frozen benchmark diagnostics, V3/V3.1 details, evidence IDs or raw configuration? Switch to Research View above.")

def analyze_target_task(df: Optional[pd.DataFrame], target: Optional[str]) -> Dict[str, Any]:
    if df is None or not target or target not in df.columns:
        return {"available": False}
    y = df[target].dropna()
    n = int(len(y))
    unique = int(y.nunique(dropna=True))
    numeric = bool(pd.api.types.is_numeric_dtype(y))
    counts = y.value_counts(dropna=True)
    top_share = float(counts.iloc[0] / n) if n and not counts.empty else None
    high_card_numeric = bool(numeric and unique >= 20)
    extreme_multiclass = bool(unique >= 50)
    return {
        "available": True,
        "n": n,
        "unique": unique,
        "numeric": numeric,
        "top_share": top_share,
        "high_cardinality_numeric": high_card_numeric,
        "extreme_multiclass": extreme_multiclass,
        "likely_binary": unique == 2,
        "likely_multiclass": unique > 2,
    }


def _positive_mask(series: pd.Series, positive_label: Any) -> pd.Series:
    direct = series.eq(positive_label)
    if bool(direct.any()):
        return direct.fillna(False)
    return series.astype(str).eq(str(positive_label)).fillna(False)


def fairness_support_table(
    df: Optional[pd.DataFrame],
    target: Optional[str],
    sensitive: Optional[str],
    positive_label: Any,
) -> pd.DataFrame:
    if df is None or not target or not sensitive or target not in df.columns or sensitive not in df.columns:
        return pd.DataFrame()
    work = df[[target, sensitive]].dropna().copy()
    work["__positive__"] = _positive_mask(work[target], positive_label)
    rows = []
    for group, part in work.groupby(sensitive, dropna=False):
        total = int(len(part))
        positive = int(part["__positive__"].sum())
        rows.append(
            {
                "Group": str(group),
                "N": total,
                "Positive N": positive,
                "Positive rate": (positive / total if total else np.nan),
                "Support status": (
                    "weak" if total < 50 or positive < 20 else "adequate"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("N", ascending=False).reset_index(drop=True)


def render_dataset_task_guard(state: Dict[str, Any]) -> None:
    df = state.get("dataset")
    target = state.get("target")
    if df is None or not target:
        return
    analysis = analyze_target_task(df, target)
    st.markdown("### Task validity check")
    if analysis.get("high_cardinality_numeric"):
        st.warning(
            "**High-cardinality numeric target detected.** `{}` has {} distinct numeric values. "
            "AwareML's current runner is configured for streaming classification, so this target may actually represent a regression/ordinal problem. "
            "Confirm the task definition before interpreting accuracy, Macro-F1 or fairness.".format(
                target, analysis.get("unique")
            )
        )
    elif analysis.get("likely_multiclass"):
        st.info(
            "`{}` is a multiclass classification target with {} classes. Macro-F1 is especially important because plain accuracy can be dominated by frequent classes.".format(
                target, analysis.get("unique")
            )
        )
    else:
        st.success("Target shape is compatible with a low-cardinality classification task.")
    if analysis.get("top_share") is not None and analysis.get("top_share") >= 0.50:
        st.warning(
            "The most common target value accounts for {:.1%} of non-missing rows. Treat accuracy cautiously and inspect Macro-F1/class support.".format(
                analysis.get("top_share")
            )
        )

    support = fairness_support_table(df, target, state.get("sensitive"), state.get("positive_label"))
    if not support.empty:
        st.markdown("**Fairness support for the selected positive label**")
        st.dataframe(support, use_container_width=True, hide_index=True)
        if analysis.get("likely_multiclass"):
            st.info(
                "For this multiclass target, positive-label fairness is a **one-vs-rest** audit for `{}`. "
                "It is not a complete multiclass fairness characterization.".format(state.get("positive_label"))
            )
        if (support["Support status"] == "weak").any():
            st.warning(
                "At least one sensitive group has weak support (N < 50 or fewer than 20 positive examples). "
                "A small/zero disparity may therefore be unstable or degenerate rather than evidence of meaningful fairness."
            )
        total_pos = int(support["Positive N"].sum())
        total_n = int(support["N"].sum())
        if total_n and total_pos / total_n < 0.05:
            st.warning(
                "The selected positive label occurs in only {:.2%} of audited rows. Fairness gaps may be sensitive to rare-event support.".format(
                    total_pos / total_n
                )
            )


def render_fairness_validity_panel(state: Dict[str, Any], results: Sequence[Mapping[str, Any]]) -> None:
    df = state.get("dataset")
    target = state.get("target")
    sensitive = state.get("sensitive")
    if df is None or not target or not sensitive:
        return
    analysis = analyze_target_task(df, target)
    support = fairness_support_table(df, target, sensitive, state.get("positive_label"))
    with st.expander("Fairness validity & support", expanded=True):
        if analysis.get("likely_multiclass"):
            st.warning(
                "Current positive-label disparity metrics are one-vs-rest for class `{}` across `{}` groups. "
                "Do not interpret a zero gap as automatically meaning a fair multiclass model.".format(
                    state.get("positive_label"), sensitive
                )
            )
        if not support.empty:
            st.dataframe(support, use_container_width=True, hide_index=True)
            weak = bool((support["Support status"] == "weak").any())
            if weak:
                st.warning("One or more group/positive-label cells have weak support; interpret disparity estimates cautiously.")
        degenerate = []
        for r in results or []:
            f = _as_dict(r.get("fairness"))
            if f.get("status") in {"insufficient_support", "undefined", "not_applicable"}:
                degenerate.append(str(r.get("framework")))
        if degenerate:
            st.info("Fairness evidence is explicitly unavailable/limited for: {}.".format(", ".join(degenerate)))


def _metric_best(frame: pd.DataFrame, column: str, direction: str) -> Optional[str]:
    if column not in frame.columns:
        return None
    s = pd.to_numeric(frame[column], errors="coerce")
    if not s.notna().any():
        return None
    idx = s.idxmax() if direction == "max" else s.idxmin()
    return str(frame.loc[idx, "framework"])


def _common_fairness_coverage(results: Sequence[Mapping[str, Any]]) -> Tuple[List[str], int]:
    sets = []
    labels_by_key = {v: k for k, v in FAIRNESS_METRICS.items()}
    for r in results or []:
        f = _as_dict(r.get("fairness"))
        available = {key for key in FAIRNESS_METRICS.values() if f.get(key) is not None and pd.notna(f.get(key))}
        sets.append(available)
    common = set.intersection(*sets) if sets else set()
    ordered = [labels_by_key[k] for k in FAIRNESS_METRICS.values() if k in common]
    return ordered, len(FAIRNESS_METRICS)


def _prediction_frame_from_state(state: Mapping[str, Any]) -> Optional[pd.DataFrame]:
    ranked = state.get("copilot_ranked")
    if isinstance(ranked, pd.DataFrame) and not ranked.empty:
        return ranked.copy()
    ranked = state.get("v2_candidates")
    if isinstance(ranked, pd.DataFrame) and not ranked.empty:
        return ranked.copy()
    return None


def render_pre_post_calibration(state: Dict[str, Any], observed_frame: pd.DataFrame) -> None:
    predicted = _prediction_frame_from_state(state)
    if predicted is None or observed_frame is None or observed_frame.empty:
        return
    with st.expander("Pre-run vs post-run recommendation comparison", expanded=False):
        pred = predicted.copy()
        obs = observed_frame.copy()
        pred_fw = str(pred.sort_values("rank").iloc[0]["framework"]) if "rank" in pred.columns else str(pred.iloc[0]["framework"])
        obs_fw = str(obs.iloc[0]["framework"])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Pre-run predicted #1", pred_fw)
        with c2:
            st.metric("Post-run observed #1", obs_fw)
        with c3:
            st.metric("Recommendation agreement", "Same" if pred_fw == obs_fw else "Changed")
        st.caption(
            "A changed winner is not automatically an error: the pre-run ranking is based on dataset meta-features and four predicted objectives, "
            "whereas Decision Lab can use observed outcomes plus fairness and interpretability. The table below shows prediction error where metrics are comparable."
        )
        rename_obs = {
            "runtime_sec": "runtime",
            "energy_kwh": "energy",
            "co2_kg": "co2",
        }
        obs2 = obs.rename(columns=rename_obs)
        rows = []
        for fw in sorted(set(pred["framework"].astype(str)) & set(obs2["framework"].astype(str))):
            pr = pred[pred["framework"].astype(str).eq(fw)].iloc[0]
            ob = obs2[obs2["framework"].astype(str).eq(fw)].iloc[0]
            row = {"Framework": fw}
            for metric in ["accuracy", "runtime", "energy", "co2"]:
                pv, ov = pr.get(metric), ob.get(metric)
                if pv is None or ov is None or pd.isna(pv) or pd.isna(ov):
                    continue
                row["Pred {}".format(metric)] = float(pv)
                row["Obs {}".format(metric)] = float(ov)
                row["Abs error {}".format(metric)] = abs(float(pv) - float(ov))
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_decision_lab_explanation(
    frame: pd.DataFrame,
    top: Any,
    weights: Any,
    fair_metric: str,
    fair_label: str,
    state: Dict[str, Any],
) -> None:
    if frame is None or frame.empty:
        return
    top_fw = str(getattr(top, "framework", frame.iloc[0]["framework"]))
    top_row = frame[frame["framework"].astype(str).eq(top_fw)].iloc[0]
    accuracy_fw = _metric_best(frame, "accuracy", "max")
    accuracy_value = None
    if accuracy_fw:
        accuracy_value = float(pd.to_numeric(frame.loc[frame["framework"].astype(str).eq(accuracy_fw), "accuracy"], errors="coerce").iloc[0])

    st.markdown("### Why this framework wins overall")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Highest overall utility", "{} · {:.3f}".format(top_fw, float(top_row.get("utility", 0.0))))
    with c2:
        st.metric(
            "Highest observed accuracy",
            "{} · {:.3f}".format(accuracy_fw, accuracy_value) if accuracy_fw and accuracy_value is not None else "N/A",
        )

    leaders = []
    for label, col, direction in [
        ("runtime", "runtime_sec", "min"),
        ("energy", "energy_kwh", "min"),
        ("CO₂", "co2_kg", "min"),
        ("fairness", "fairness_score", "max"),
        ("interpretability", "interpretability_score", "max"),
    ]:
        if _metric_best(frame, col, direction) == top_fw:
            leaders.append(label)
    weight_dict = weights.as_dict() if hasattr(weights, "as_dict") else _as_dict(weights)
    if top_fw != accuracy_fw:
        reason = (
            "**{} is not the accuracy winner.** It is ranked #1 because Decision Lab optimizes the weighted combination of all selected observed criteria. ".format(top_fw)
        )
        if leaders:
            reason += "It is strongest on {} in this run, which offsets its lower accuracy under the current weights.".format(
                ", ".join(leaders)
            )
        else:
            reason += "Its combined trade-off produces the highest normalized utility under the current weights."
        st.info(reason)
    else:
        st.success(
            "{} is both the observed accuracy leader and the highest overall utility choice under the current weights.".format(top_fw)
        )

    if fair_metric == "composite":
        common, total = _common_fairness_coverage(result_dicts(state))
        coverage = len(common)
        st.info(
            "**Common-criteria fairness coverage: {}/{}.** Composite fairness uses the same available criteria for every framework: {}. "
            "Missing criteria remain unavailable and are not converted to zero.".format(
                coverage, total, ", ".join(common) if common else "none"
            )
        )

    near = bool(getattr(top, "near_pareto", top_row.get("near_pareto", False)))
    if near:
        st.caption(
            "**Near-Pareto: Yes.** {} is one of the non-clearly-dominated trade-off choices under the current ε tolerance. "
            "This does not mean it is best on every metric.".format(top_fw)
        )
    else:
        st.caption("Near-Pareto: No under the current ε tolerance; the framework can still rank highly under the chosen weighted utility.")

    render_pre_post_calibration(state, frame)


def _event_sample(event: Any) -> Optional[int]:
    if isinstance(event, dict):
        event = event.get("sample") or event.get("index") or event.get("position")
    try:
        return int(event) if event is not None else None
    except Exception:
        return None


def cluster_positions(positions: Iterable[int], gap: int) -> List[int]:
    vals = sorted({int(x) for x in positions if x is not None})
    if not vals:
        return []
    gap = max(1, int(gap))
    groups: List[List[int]] = [[vals[0]]]
    for value in vals[1:]:
        if value - groups[-1][-1] < gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [int(round(float(np.median(group)))) for group in groups]


def _adaptation_samples(result: Mapping[str, Any]) -> List[int]:
    values: List[int] = []
    candidates = list(result.get("refit_events") or [])
    params = _as_dict(result.get("parameters"))
    candidates.extend(params.get("refit_events") or [])
    for episode in _as_dict(result.get("drift_summary")).get("episodes") or []:
        if isinstance(episode, dict):
            candidates.append(
                episode.get("refit_sample") or episode.get("retrain_sample") or episode.get("adaptation_sample")
            )
    for event in candidates:
        sample = _event_sample(event)
        if sample is not None:
            values.append(sample)
    return sorted(set(values))


def _recovery_lags(result: Mapping[str, Any]) -> List[int]:
    lags: List[int] = []
    for episode in _as_dict(result.get("drift_summary")).get("episodes") or []:
        if not isinstance(episode, dict):
            continue
        drift = episode.get("drift_sample") or episode.get("sample") or episode.get("drift_index")
        recovery = (
            episode.get("recovery_sample")
            or episode.get("recovered_at_sample")
            or episode.get("recovery_end_sample")
            or episode.get("recovery_sample_index")
        )
        try:
            if drift is not None and recovery is not None and int(recovery) >= int(drift):
                lags.append(int(recovery) - int(drift))
        except Exception:
            pass
    return lags


def prepare_drift_display(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    raw = [dict(r) for r in results or []]
    if not raw:
        return raw
    state = ensure_research_state()
    frameworks = [str(r.get("framework")) for r in raw]
    mode = st.segmented_control(
        "Drift display",
        ["Selected framework", "All frameworks", "Clustered drift episodes"],
        default=state.get("drift_display_mode") if state.get("drift_display_mode") in {"Selected framework", "All frameworks", "Clustered drift episodes"} else "Selected framework",
        key="pre14_drift_display_mode",
        help="Raw drift events are never deleted. Clustered mode groups nearby framework alerts only for visualization.",
    ) or "Selected framework"
    state["drift_display_mode"] = mode
    selected = state.get("drift_selected_framework")
    if selected not in frameworks:
        selected = frameworks[0]
    if mode == "Selected framework":
        selected = st.segmented_control(
            "Framework",
            frameworks,
            default=selected,
            key="pre14_drift_framework",
        ) or selected
        state["drift_selected_framework"] = selected

    max_sample = 0
    for r in raw:
        for p in r.get("points") or []:
            try:
                max_sample = max(max_sample, int(p.get("sample") or 0))
            except Exception:
                pass
    summary_rows = []
    for r in raw:
        drifts = [_event_sample(x) for x in r.get("drift_events") or []]
        drifts = [x for x in drifts if x is not None]
        lags = _recovery_lags(r)
        ds = _as_dict(r.get("drift_summary"))
        summary_rows.append(
            {
                "Framework": str(r.get("framework")),
                "Drift alerts": len(drifts),
                "Alerts / 1k samples": (len(drifts) * 1000.0 / max_sample) if max_sample else np.nan,
                "Explicit adaptation actions": len(_adaptation_samples(r)),
                "Recovered episodes": ds.get("n_recovered"),
                "Median recovery lag": int(round(float(np.median(lags)))) if lags else ds.get("median_recovery_samples"),
            }
        )
    st.caption(
        "The total is the sum of **framework-specific detector alerts**, not the number of unique dataset-wide changes."
    )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if mode == "Selected framework":
        return [copy.deepcopy(r) for r in raw if str(r.get("framework")) == str(selected)]
    if mode == "All frameworks":
        return [copy.deepcopy(r) for r in raw]

    gap = int(state.get("window_size") or 500)
    all_positions = []
    for r in raw:
        all_positions.extend(_event_sample(x) for x in r.get("drift_events") or [])
    all_positions = [x for x in all_positions if x is not None]
    episodes = cluster_positions(all_positions, gap)
    clustered = [copy.deepcopy(r) for r in raw]
    for r in clustered:
        r["drift_events"] = []
    if clustered:
        clustered[0]["drift_events"] = [{"sample": x, "display_clustered": True} for x in episodes]
    st.info(
        "Display-only clustering grouped {} raw framework alerts into {} drift episodes using a <1-window gap ({} samples). "
        "Raw detections remain unchanged in the run artifacts.".format(len(all_positions), len(episodes), gap)
    )
    return clustered
