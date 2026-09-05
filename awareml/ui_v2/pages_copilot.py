from __future__ import annotations

import streamlit as st

from awareml.llm import (
    CopilotService,
    EvidenceGroundedObjectiveSelectorV3,
    HybridEvidenceGroundedObjectiveSelectorV31,
    GoalParser,
    GroundedCopilotChat,
    OllamaClient,
    ReviewStore,
    StrictJournalOllamaClient,
)

from .components import (
    evidence_chips,
    hero,
    humanize_rationale_text,
    section,
    status_panel,
)
from .data import load_v2_recommender
from .page_utils import dataset_ready, fmt, phase_pills
from .research_evidence import (
    load_phase12_objective_selection_reliability,
    load_phase12_v3_posthoc_diagnostic,
    load_phase12_v31_posthoc_diagnostic,
)
from .state import ROOT, ensure_research_state
from .copilot_three_path import render_historical_preference_prior_tab, render_dataset_aware_v2_tab
from .copilot_historical import render_historical_meta_recommender_tab
from .pre14_usability import (
    clean_pre_run_rationale,
    persist_objective_review,
    render_accessible_objective_interpretation,
    render_copilot_plan_summary,
)
from .copilot_v31_components import (
    clear_previous_copilot_result,
    render_copilot_clarification,
    set_copilot_clarification,
    v31_audit_rows,
)


def _as_dict(value):
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value)


def _clear_framework_state(state):
    state["copilot_proposal"] = None
    state["copilot_ranked"] = None
    state["copilot_evidence"] = None
    state["copilot_meta"] = None
    state["copilot_review"] = None


def _clear_context_free_state(state):
    state["copilot_context_free_interpretation"] = None
    state["copilot_context_free_meta"] = None


def _weighting_explanation(selected):
    selected = list(selected or [])
    count = len(selected)
    if count <= 0:
        return "No objective is currently selected, so no downstream weights can be assigned."
    share = 1.0 / float(count)
    return (
        "{} objective{} selected → equal_selected_v1 assigns 1/{} = {:.3f} "
        "to each selected objective and 0.000 to every unselected objective."
    ).format(
        count,
        "" if count == 1 else "s",
        count,
        share,
    )


def _render_objective_interpretation(interpretation, parse_meta, state):
    interpretation = _as_dict(interpretation)

    section(
        "Objective interpretation",
        (
            "This is the primary journal-facing output. It comes from the "
            "natural-language scenario and does not require a dataset."
        ),
    )

    selected = interpretation.get("selected_objectives") or []
    st.markdown(
        "**LLM-selected objectives:** {}".format(
            " · ".join(selected) if selected else "None"
        )
    )

    phase12 = load_phase12_objective_selection_reliability(ROOT)
    if phase12:
        st.warning(
            "These objectives are the model's interpretation of the scenario, not "
            "human ground truth. In the frozen Phase-12 benchmark, {} achieved "
            "Micro-F1 {:.3f} and {:.1%} exact-set match, with a dominant adversarial "
            "failure tendency of {}. Human review is therefore required.".format(
                phase12.get("model") or "LLaMA 3 8B",
                float(phase12.get("micro_f1") or 0.0),
                float(phase12.get("exact_match_rate") or 0.0),
                str(phase12.get("primary_failure_tendency") or "not available").replace("_", " "),
            )
        )
        with st.expander("Objective-selection reliability · frozen Phase 12", expanded=False):
            status_panel(
                {
                    "Benchmark cases": str(phase12.get("benchmark_cases") or "N/A"),
                    "Micro-F1": fmt(phase12.get("micro_f1"), 3),
                    "Macro-F1": fmt(phase12.get("macro_f1"), 3),
                    "Exact-set match": (
                        "{:.1%}".format(float(phase12.get("exact_match_rate")))
                        if phase12.get("exact_match_rate") is not None
                        else "N/A"
                    ),
                    "Paraphrase consistency": (
                        "{:.1%}".format(float(phase12.get("paraphrase_consistency_rate")))
                        if phase12.get("paraphrase_consistency_rate") is not None
                        else "N/A"
                    ),
                    "Paraphrase set-change": (
                        "{:.1%}".format(float(phase12.get("paraphrase_set_change_rate")))
                        if phase12.get("paraphrase_set_change_rate") is not None
                        else "N/A"
                    ),
                    "Primary failure tendency": str(
                        phase12.get("primary_failure_tendency") or "N/A"
                    ).replace("_", " "),
                    "Benchmark status": str(phase12.get("release_status") or "N/A"),
                }
            )
            st.caption(
                "This is aggregate calibration evidence from the frozen benchmark, "
                "not a confidence score for the current sentence. The Phase-12 "
                "artifact is read-only and is not modified by this UI."
            )
    else:
        st.warning(
            "These objectives are the model's interpretation of the scenario, not "
            "human ground truth. Frozen Phase-12 calibration evidence is not "
            "available locally, so human review remains required."
        )

    v3_diag = load_phase12_v3_posthoc_diagnostic(ROOT)
    if v3_diag:
        st.info(
            "An evidence-grounded V3 improvement diagnostic is available locally. "
            "These V3 numbers are POST-HOC development evidence on the already-inspected "
            "Phase-12 cases; they do not replace the frozen v1 journal result. A fresh "
            "independently annotated benchmark is still required for a new final claim."
        )
        with st.expander("V3 improvement diagnostic · post-hoc development only", expanded=False):
            status_panel(
                {
                    "Micro precision": fmt(v3_diag.get("micro_precision"), 3),
                    "Micro recall": fmt(v3_diag.get("micro_recall"), 3),
                    "Micro-F1": fmt(v3_diag.get("micro_f1"), 3),
                    "Macro-F1": fmt(v3_diag.get("macro_f1"), 3),
                    "Exact-set match": (
                        "{:.1%}".format(float(v3_diag.get("exact_match_rate")))
                        if v3_diag.get("exact_match_rate") is not None
                        else "N/A"
                    ),
                    "Paraphrase consistency": (
                        "{:.1%}".format(float(v3_diag.get("paraphrase_consistency_rate")))
                        if v3_diag.get("paraphrase_consistency_rate") is not None
                        else "not run"
                    ),
                    "Adversarial over-selection": (
                        "{} / {}".format(
                            v3_diag.get("adversarial_over_selection_count"),
                            v3_diag.get("adversarial_cases"),
                        )
                        if v3_diag.get("adversarial_cases") is not None
                        else "not run"
                    ),
                    "Evidence role": "post-hoc development diagnostic",
                }
            )

    v31_diag = load_phase12_v31_posthoc_diagnostic(ROOT)
    if v31_diag:
        st.markdown("### V3.1 development checkpoint")
        st.info(
            "V3.1 is a recall-balanced hybrid improvement candidate. These values are POST-HOC development evidence "
            "on already-inspected Phase-12 material and are not a new independent journal result."
        )
        with st.expander("V3.1 diagnostic · development only", expanded=False):
            status_panel(
                {
                    "Micro precision": fmt(v31_diag.get("micro_precision"), 3),
                    "Micro recall": fmt(v31_diag.get("micro_recall"), 3),
                    "Micro-F1": fmt(v31_diag.get("micro_f1"), 3),
                    "Macro-F1": fmt(v31_diag.get("macro_f1"), 3),
                    "Exact-set match": (
                        "{:.1%}".format(float(v31_diag.get("exact_match_rate")))
                        if v31_diag.get("exact_match_rate") is not None else "N/A"
                    ),
                    "Paraphrase consistency": (
                        "{:.1%}".format(float(v31_diag.get("paraphrase_consistency_rate")))
                        if v31_diag.get("paraphrase_consistency_rate") is not None else "not run"
                    ),
                    "Paraphrase denominator": (
                        "{} comparisons".format(v31_diag.get("paraphrase_evaluations"))
                        if v31_diag.get("paraphrase_evaluations") is not None else "not run"
                    ),
                    "Adversarial over-selection": (
                        "{} / {}".format(v31_diag.get("adversarial_over_selection_count"), v31_diag.get("adversarial_cases"))
                        if v31_diag.get("adversarial_cases") is not None else "not run"
                    ),
                    "Evidence role": "post-hoc development diagnostic",
                }
            )

    evidence_audit = dict(parse_meta.get("evidence_audit") or {})
    decisions = dict(evidence_audit.get("decisions") or {})
    if decisions:
        st.markdown("**Hybrid evidence-grounded V3.1 selection audit**")
        st.dataframe(v31_audit_rows(evidence_audit), use_container_width=True, hide_index=True)
        recovered = list(evidence_audit.get("semantic_recovered_objectives") or [])
        if recovered:
            st.info(
                "Semantic recovery was used for: {}. These objectives had explicit scenario-local evidence even though "
                "the LLaMA proposal omitted them. Human review is required before acceptance.".format(", ".join(recovered))
            )
        st.caption(
            "The audit separates: (1) what LLaMA proposed, (2) whether an evidence phrase was grounded, "
            "(3) whether the scenario contains objective-specific semantic support, and (4) the final V3.1 decision. "
            "Broad phrases such as sustained deployment or generic sustainability are not decisive by themselves."
        )

    c1, c2, c3, c4 = st.columns(4)
    weights = interpretation.get("primary_weights") or {}
    with c1:
        st.metric("Accuracy weight", fmt(weights.get("accuracy"), 3))
    with c2:
        st.metric("Runtime weight", fmt(weights.get("runtime"), 3))
    with c3:
        st.metric("Energy weight", fmt(weights.get("energy"), 3))
    with c4:
        st.metric("CO2 weight", fmt(weights.get("co2"), 3))

    st.info(_weighting_explanation(selected))

    with st.expander("Why do these objective weights have these values?"):
        st.write(
            "Phase 11 deliberately separates **objective selection** from "
            "**objective weighting**. The LLM only selects a subset of the "
            "four frozen objectives. The downstream policy then applies "
            "`equal_selected_v1`."
        )
        st.code(
            "\n".join(
                [
                    "Selected objectives: {}".format(
                        ", ".join(selected) if selected else "none"
                    ),
                    "Number selected: {}".format(len(selected)),
                    "Equal share: {}".format(
                        "{:.6f}".format(1.0 / len(selected))
                        if selected
                        else "not applicable"
                    ),
                    "Unselected objectives: weight 0.000000",
                ]
            ),
            language="text",
        )
        st.caption(
            "These numbers do not come from the uploaded dataset or from a completed "
            "framework run. They are produced only by the frozen weighting policy."
        )

    st.caption(
        "Selection status: {} · source: {} · model: {} · weighting policy: {} · fallback used: {}".format(
            interpretation.get("selection_status"),
            interpretation.get("selection_source"),
            interpretation.get("selection_model") or "none",
            interpretation.get("weighting_policy"),
            interpretation.get("fallback_used"),
        )
    )

    for warning in parse_meta.get("warnings") or []:
        st.warning(warning)

    hcai = interpretation.get("hcai_requirements") or {}
    st.info(
        "HCAI requirements — drift sensitivity: {} · fairness required: {} · "
        "explainability: {}. These remain outside the four-objective selection benchmark.".format(
            hcai.get("drift_sensitivity"),
            hcai.get("fairness_required"),
            hcai.get("explainability_level"),
        )
    )

    section(
        "Human review of objective interpretation",
        (
            "This review is for the interactive Copilot only. It does not become "
            "Phase-12 benchmark ground truth."
        ),
    )

    review_choice = st.segmented_control(
        "Interpretation review",
        ["Accept interpretation", "Flag for correction", "Reject"],
        default="Accept interpretation",
        key="r11_objective_review_choice",
    )
    review_note = st.text_input(
        "Objective-review note",
        value=state.get("copilot_objective_review_note", ""),
        key="r11_objective_review_note",
        placeholder=(
            "Example: CO2 should also be selected because 'low-impact' implies "
            "an environmental constraint."
        ),
    )

    if st.button(
        "Record objective review",
        key="r11_record_objective_review",
    ):
        state["copilot_objective_review"] = {
            "decision": review_choice,
            "note": review_note or None,
            "selected_objectives_seen": list(selected),
            "benchmark_ground_truth": False,
        }
        state["copilot_objective_review_note"] = review_note
        persisted = persist_objective_review(
            state,
            decision=review_choice,
            selected_seen=list(selected),
            corrected=None,
            note=review_note or None,
            source="copilot_research_view",
        )
        state["copilot_objective_review_persisted"] = persisted
        st.success(
            "Objective interpretation review recorded in the append-only Copilot audit trail. "
            "It is not treated as journal benchmark ground truth."
        )

    if state.get("copilot_objective_review"):
        review = state["copilot_objective_review"]
        st.caption(
            "Current objective-review state: {}{}".format(
                review.get("decision"),
                (
                    " · {}".format(review.get("note"))
                    if review.get("note")
                    else ""
                ),
            )
        )


def _render_framework_placeholder(has_dataset):
    section(
        "Framework recommendation",
        (
            "The interface remains structurally consistent in context-free mode. "
            "Dataset-dependent fields stay unavailable until a dataset profile exists."
        ),
    )

    cols = st.columns(4)
    with cols[0]:
        st.metric(
            "Pre-run predicted framework",
            "Awaiting dataset for V2" if not has_dataset else "Generate proposal",
        )
    with cols[1]:
        st.metric("Predicted ML rank", "—")
    with cols[2]:
        st.metric("Normalized preference utility", "—")
    with cols[3]:
        st.metric("Review state", "OBJECTIVE READY")

    st.markdown(
        """
        <div class="r9-callout">
          <b>Goal Copilot does not fabricate a dataset-specific framework in context-free mode.</b><br>
          The five frameworks have dataset-dependent behaviour, so AwareML waits
          for dataset meta-features before asking frozen ML Recommender V2 to rank
          AutoStreamML, AutoClass, EvoAutoML, OAML and ChaCha.<br><br>
          <b>Need a framework before you have a dataset?</b> Use the Historical Meta-Recommender tab for a global 705-run prior.<br><br>
          <b>You do not need to run the benchmark first for V2.</b> Loading a dataset and
          choosing its target is enough to obtain the pre-run prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("What becomes available after a dataset is loaded?"):
        st.markdown(
            """
            - pre-run predicted framework and ML rank;
            - predicted utility under the selected objective weights;
            - framework-specific proposed configuration;
            - grounded recommendation rationale and technical evidence IDs;
            - full configuration approval / approve-with-edits / reject workflow.
            """
        )


def _goal_copilot_workspace_page():
    state = ensure_research_state()
    has_dataset = bool(dataset_ready())
    has_observed_run = bool(state.get("run_results"))

    hero(
        "HUMAN-CENTRIC AI",
        "Copilot Workspace",
        (
            "Start with only a natural-language deployment scenario. A dataset is "
            "optional for objective interpretation and is needed only when you want "
            "a dataset-aware framework ranking from ML Recommender V2."
        ),
        pills=phase_pills(),
    )

    st.markdown(
        """
        <div class="r9-callout">
          <b>Objective-selection architecture:</b><br>
          Scenario → <b>selected objective set</b> → <b>equal-selected weighting</b>
          <span style="opacity:.72">[works without a dataset]</span><br>
          Dataset meta-profile → <b>Phase-6 ML Recommender V2</b> → framework ranking
          <span style="opacity:.72">[requires a dataset]</span><br><br>
          The LLM does <b>not</b> directly choose AutoClass, OAML, ChaCha,
          AutoStreamML or EvoAutoML.
        </div>
        """,
        unsafe_allow_html=True,
    )

    journal_status = StrictJournalOllamaClient().status()
    exact_model = "llama3:8b"

    left, right = st.columns([1.05, 0.95])

    with left:
        section(
            "Natural-language scenario",
            (
                "No dataset is required here. Describe deployment needs naturally; "
                "the journal task is to infer a subset of Accuracy, Runtime, Energy and CO2."
            ),
        )

        goal = st.text_area(
            "Streaming AutoML scenario",
            value=state.get("copilot_goal")
            or (
                "Suitable for deployment in a low-impact edge environment "
                "while still providing strong performance."
            ),
            height=145,
            key="r11_copilot_goal",
            placeholder=(
                "Example: The system will run on a battery-powered edge device "
                "and needs dependable predictions with a small environmental footprint."
            ),
        )
        state["copilot_goal"] = goal

        use_llm = st.toggle(
            "Use hybrid evidence-grounded LLaMA 3 8B selector V3.1",
            value=True,
            key="r11_copilot_llm",
            help=(
                "V3 keeps the exact locked LLaMA runtime but adds evidence-grounded, "
                "conservative objective filtering. The frozen Phase-12 v1 benchmark "
                "remains unchanged for reproducibility."
            ),
        )

        st.caption(
            "Frozen objective vocabulary: Accuracy · Runtime · Energy · CO2. "
            "Fairness, drift and explainability are handled separately as HCAI requirements."
        )

        with st.expander("How V3.1 balances trust and recall", expanded=False):
            st.markdown(
                """
                **V3.1 uses two transparent evidence sources.** The locked LLaMA proposes objectives, while an
                objective-specific semantic layer checks the scenario itself. Unsupported LLaMA additions are rejected;
                strong explicit cues can recover an objective the LLaMA omitted. Generic phrases such as *sustained
                deployment* do not count as evidence on their own. Every proposal still passes through human review.
                """
            )


        if st.button(
            "Generate Copilot proposal",
            type="primary",
            use_container_width=True,
            key="r11_generate_proposal",
        ):
            clear_previous_copilot_result(state)
            try:
                rationale_client = OllamaClient(model=exact_model)
                service = CopilotService(
                    recommender=(load_v2_recommender() if has_dataset else None),
                    goal_parser=GoalParser(selector=HybridEvidenceGroundedObjectiveSelectorV31()),
                    chat=GroundedCopilotChat(client=rationale_client),
                    review_store=ReviewStore(
                        ROOT / "artifacts" / "copilot" / "reviews.jsonl"
                    ),
                )

                if has_dataset:
                    proposal, ranked, evidence, meta = service.propose_from_dataframe(
                        goal=goal,
                        df=state["dataset"],
                        target=state["target"],
                        sensitive_attribute=state.get("sensitive"),
                        current_config=None,
                        use_llm=use_llm,
                    )
                    _clear_context_free_state(state)
                    state["copilot_proposal"] = proposal
                    state["copilot_ranked"] = ranked
                    state["copilot_evidence"] = evidence
                    state["copilot_meta"] = meta
                    state["copilot_review"] = None
                    st.success(
                        "Dataset-aware pre-run proposal generated. Human review is required."
                    )
                else:
                    interpretation, meta = service.interpret_goal(
                        goal,
                        use_llm=use_llm,
                    )
                    _clear_framework_state(state)
                    state["copilot_context_free_interpretation"] = interpretation
                    state["copilot_context_free_meta"] = meta
                    st.success(
                        "Objective proposal generated without dataset context. "
                        "Load a dataset later only if you want a framework ranking."
                    )
            except Exception as exc:
                error_name = type(exc).__name__
                if error_name == "GoalSelectionError":
                    message = (
                        "AwareML could not identify a sufficiently supported objective set from this wording. "
                        "No new objective weights or framework recommendation were generated."
                    )
                    detail = "The objective selector abstained because the scenario needs clearer deployment priorities."
                elif error_name == "JournalModelLockError":
                    message = (
                        "The exact journal LLaMA/Ollama runtime is not ready, so AwareML did not generate a proposal."
                    )
                    detail = "Check the model-lock status panel and restore the required frozen runtime before retrying."
                else:
                    message = "AwareML could not generate a new proposal from this scenario."
                    detail = "Review the model/status panel and clarify the scenario before retrying."
                set_copilot_clarification(
                    state,
                    message=message,
                    detail=detail,
                    error_type=error_name,
                )
                st.warning("No new proposal generated. See the clarification panel below.")

    with right:
        status_panel(
            {
                "Journal objective model": exact_model,
                "Interactive objective selector": "hybrid evidence-grounded V3.1",
                "Exact model lock": (
                    "PASS" if journal_status.get("reachable") else "NOT READY"
                ),
                "Model fallback": "Forbidden",
                "Weighting policy": "equal_selected_v1",
                "Dataset needed for objective selection": "No",
                "Dataset needed for dataset-aware V2 ranking": "Yes",
                "Raw dataset rows to LLM": "False",
                "Human review gate": "Required",
            }
        )

        if journal_status.get("error"):
            st.warning(journal_status["error"])

        st.markdown(
            """
            <div class="r9-callout" style="margin-top:14px">
              <b>Problem A:</b> Which objectives does the scenario imply?<br>
              <b>Problem B:</b> How are those selected objectives weighted?<br>
              <b>Problem C:</b> Which framework is predicted to fit this dataset?<br><br>
              A and B are context-free. C requires dataset meta-features because
              framework performance depends on the data.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not has_dataset:
        st.info(
            "Goal Copilot context-free mode: objective interpretation and weighting are available now. "
            "The dataset-aware ML Recommender V2 prediction remains pending until a dataset profile exists. "
            "For a dataset-free framework starting point, open the Historical Meta-Recommender tab."
        )
    elif not has_observed_run:
        st.info(
            "Pre-run recommendation mode: a dataset is uploaded, but no framework "
            "benchmark has been executed yet. Objective weights come from the "
            "scenario; framework performance is predicted by frozen ML Recommender "
            "V2 from the uploaded dataset's meta-profile. These are not observed results."
        )
    else:
        st.info(
            "Dataset-aware pre-run mode: Copilot still uses the scenario plus the "
            "dataset meta-profile for framework ranking. Already-observed benchmark "
            "results are separate evidence in Streaming Observatory / Decision Lab."
        )

    if render_copilot_clarification(state):
        return

    context_free = state.get("copilot_context_free_interpretation")
    context_free_meta = state.get("copilot_context_free_meta") or {}
    proposal = state.get("copilot_proposal")

    if proposal is None and context_free is not None:
        _render_objective_interpretation(
            context_free,
            context_free_meta.get("goal_parse") or {},
            state,
        )
        _render_framework_placeholder(has_dataset=has_dataset)
        return

    if proposal is None:
        return

    proposal_dict = (
        proposal.model_dump() if hasattr(proposal, "model_dump") else proposal
    )
    interpretation = proposal_dict.get("interpretation") or {}
    parse_meta = ((state.get("copilot_meta") or {}).get("goal_parse") or {})

    _render_objective_interpretation(
        interpretation,
        parse_meta,
        state,
    )

    render_copilot_plan_summary(proposal_dict, state, has_observed_run)
    # Simple View ends after the concise plan + human review controls.
    # The complete legacy/research proposal remains available in Research View.
    if state.get("copilot_view_mode") == "Simple View":
        return

    section(
        "Detailed framework proposal",
        (
            "The ML Recommender receives the objective weights plus the current "
            "dataset meta-profile. It—not the LLM—produces the framework ranking."
        ),
    )

    cols = st.columns(4)
    with cols[0]:
        st.metric(
            "Pre-run predicted framework",
            str(proposal_dict.get("ml_recommender_framework")),
        )
    with cols[1]:
        st.metric(
            "Predicted ML rank",
            "#{}".format(proposal_dict.get("ml_recommender_rank", 1)),
        )
    with cols[2]:
        st.metric(
            "Predicted utility",
            fmt(proposal_dict.get("ml_recommender_utility"), 4),
        )
    with cols[3]:
        review_state = state.get("copilot_review")
        decision = (
            review_state.get("decision")
            if isinstance(review_state, dict)
            else "PROPOSED"
        )
        st.metric("Review state", decision or "PROPOSED")

    if not has_observed_run:
        st.warning(
            "Pre-run prediction: the framework/rank/utility above are predictions "
            "from ML Recommender V2, not measured outcomes from the uploaded dataset."
        )

    st.markdown("**Why this framework?**")
    st.write(humanize_rationale_text(clean_pre_run_rationale(proposal_dict.get("rationale"))))
    evidence_chips(proposal_dict.get("evidence_keys"), show_raw=True)

    for warning in proposal_dict.get("warnings") or []:
        st.warning(warning)

    config = proposal_dict.get("proposed_config") or {}
    st.json(config, expanded=False)

    section(
        "Human review of full framework proposal",
        "Approved-with-edits records the configuration diff in the Copilot audit trail.",
    )
    review_mode = st.segmented_control(
        "Decision",
        ["Approve", "Approve with edits", "Reject"],
        default="Approve",
        key="r11_review_mode",
    )

    edits = None
    if review_mode == "Approve with edits":
        edited_window = st.number_input(
            "Window size",
            min_value=50,
            max_value=100000,
            value=int(config.get("window_size", 1000)),
            step=50,
            key="r11_review_window",
        )
        edited_budget = st.number_input(
            "Time budget (seconds)",
            min_value=1.0,
            max_value=86400.0,
            value=float(config.get("time_budget_sec", 60.0)),
            step=5.0,
            key="r11_review_budget",
        )
        edits = {
            "window_size": int(edited_window),
            "time_budget_sec": float(edited_budget),
        }

    note = st.text_input("Review note", key="r11_review_note")

    if st.button("Record human review", key="r11_record_review"):
        try:
            rationale_client = OllamaClient(model=exact_model)
            service = CopilotService(
                recommender=load_v2_recommender(),
                goal_parser=GoalParser(selector=HybridEvidenceGroundedObjectiveSelectorV31()),
                chat=GroundedCopilotChat(client=rationale_client),
                review_store=ReviewStore(
                    ROOT / "artifacts" / "copilot" / "reviews.jsonl"
                ),
            )
            decision_map = {
                "Approve": "approved",
                "Approve with edits": "approved_with_edits",
                "Reject": "rejected",
            }
            review = service.review(
                proposal,
                decision=decision_map[review_mode],
                edits=edits,
                note=note or None,
                persist=True,
            )
            state["copilot_review"] = review.model_dump()
            st.success("Human review recorded.")
        except Exception as exc:
            st.error("Review could not be recorded: {}".format(exc))

    if state.get("copilot_review"):
        st.json(state["copilot_review"], expanded=False)


# -----------------------------------------------------------------------------
# Final pre-Phase-14 progressive-disclosure wrapper. The complete pre-existing
# research renderer is intentionally preserved and exposed in Research View.
# -----------------------------------------------------------------------------
_PRE14_RESEARCH_OBJECTIVE_RENDERER = _render_objective_interpretation

def _render_objective_interpretation(interpretation, parse_meta, state):
    return render_accessible_objective_interpretation(
        _PRE14_RESEARCH_OBJECTIVE_RENDERER, interpretation, parse_meta, state
    )

# PRE14_THREE_PATH_COPILOT_WRAPPER
def copilot_workspace_page():
    st.markdown("## AwareML Copilot · choose the evidence path that fits your situation")
    st.caption(
        "Goal Copilot understands what you want. Historical Preference Prior shows what generally worked before. "
        "Dataset-aware ML Recommender V2 predicts what should work for the dataset you actually loaded."
    )

    overview = st.columns(3)
    with overview[0]:
        st.markdown("**1 · Goal Copilot**")
        st.caption("No dataset · LLaMA 3 8B + V3.1 · understands Accuracy / Runtime / Energy / CO2 · not a framework selector by itself.")
    with overview[1]:
        st.markdown("**2 · Historical Preference Prior**")
        st.caption("No dataset · 705 runs summarized · global starting point · historical aggregation, not machine learning.")
    with overview[2]:
        st.markdown("**3 · Dataset-aware ML Recommender V2**")
        st.caption("Dataset + target · learned models · dataset-specific pre-run framework prediction · true ML recommender.")

    goal_tab, prior_tab, v2_tab = st.tabs([
        "1 · Goal Copilot",
        "2 · Historical Preference Prior · No dataset",
        "3 · Dataset-aware ML Recommender V2",
    ])
    with goal_tab:
        _goal_copilot_workspace_page()
    with prior_tab:
        render_historical_preference_prior_tab()
    with v2_tab:
        render_dataset_aware_v2_tab()
