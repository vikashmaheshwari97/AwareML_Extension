from __future__ import annotations

import streamlit as st

from awareml.llm import (
    CopilotService,
    GoalParser,
    GroundedCopilotChat,
    OllamaClient,
    ReviewStore,
    StrictJournalOllamaClient,
)

from .components import (
    empty_state,
    evidence_chips,
    hero,
    humanize_rationale_text,
    section,
    status_panel,
)
from .data import load_v2_recommender
from .page_utils import dataset_ready, fmt, phase_pills
from .state import ROOT, ensure_research_state


def copilot_workspace_page():
    state = ensure_research_state()

    hero(
        "HUMAN-CENTRIC AI",
        "Copilot Workspace",
        (
            "Interpret a natural-language scenario as an explicit objective set, "
            "map that set to a documented weighting policy, rank frameworks with "
            "the frozen ML Recommender V2, and keep the human review gate."
        ),
        pills=phase_pills(),
    )

    if not dataset_ready():
        empty_state(
            "Dataset context required",
            "Load a dataset and target in Run Studio before asking the Copilot for a pre-run configuration.",
        )
        return

    st.markdown(
        """
        <div class="r9-callout">
          <b>Phase 11 architecture:</b><br>
          Scenario → <b>selected objective set</b> → <b>equal-selected weighting</b>
          → Phase-6 ML Recommender V2 → reviewable configuration.<br><br>
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
                "Describe deployment needs naturally. The primary journal task is "
                "to infer a subset of Accuracy, Runtime, Energy and CO2."
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
            "Use frozen journal LLaMA 3 8B for objective selection",
            value=True,
            key="r11_copilot_llm",
            help=(
                "Journal mode uses the exact Phase-10 model lock, prompt, schema "
                "and no silent model fallback. Turn this off only for the transparent "
                "deterministic fallback/demo parser."
            ),
        )

        st.caption(
            "Frozen objective vocabulary: Accuracy · Runtime · Energy · CO2. "
            "Fairness, drift and explainability are handled separately as HCAI requirements."
        )

        if st.button(
            "Generate Copilot proposal",
            type="primary",
            use_container_width=True,
            key="r11_generate_proposal",
        ):
            try:
                # Objective selection uses GoalParser -> StrictJournalOllamaClient.
                # Grounded rationale uses the same frozen model tag for consistency.
                rationale_client = OllamaClient(model=exact_model)
                service = CopilotService(
                    recommender=load_v2_recommender(),
                    goal_parser=GoalParser(),
                    chat=GroundedCopilotChat(client=rationale_client),
                    review_store=ReviewStore(
                        ROOT / "artifacts" / "copilot" / "reviews.jsonl"
                    ),
                )
                proposal, ranked, evidence, meta = service.propose_from_dataframe(
                    goal=goal,
                    df=state["dataset"],
                    target=state["target"],
                    sensitive_attribute=state.get("sensitive"),
                    current_config=None,
                    use_llm=use_llm,
                )
                state["copilot_proposal"] = proposal
                state["copilot_ranked"] = ranked
                state["copilot_evidence"] = evidence
                state["copilot_meta"] = meta
                state["copilot_review"] = None
                st.success("Proposal generated. Human review is required.")
            except Exception as exc:
                st.error("Copilot proposal failed: {}: {}".format(type(exc).__name__, exc))

    with right:
        status_panel(
            {
                "Journal objective model": exact_model,
                "Exact model lock": (
                    "PASS" if journal_status.get("reachable") else "NOT READY"
                ),
                "Model fallback": "Forbidden",
                "Weighting policy": "equal_selected_v1",
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
              <b>Problem B:</b> How are those selected objectives weighted?<br><br>
              Phase 11 evaluates these separately. The current downstream policy
              gives equal weight to every selected objective and zero to all
              unselected objectives.
            </div>
            """,
            unsafe_allow_html=True,
        )

    proposal = state.get("copilot_proposal")
    if proposal is None:
        return

    proposal_dict = (
        proposal.model_dump() if hasattr(proposal, "model_dump") else proposal
    )
    interpretation = proposal_dict.get("interpretation") or {}
    parse_meta = ((state.get("copilot_meta") or {}).get("goal_parse") or {})

    section(
        "Objective interpretation",
        (
            "The selected set is the primary journal-facing output. "
            "Weights are a separate documented downstream mapping."
        ),
    )

    selected = interpretation.get("selected_objectives") or []
    st.markdown(
        "**Selected objectives:** {}".format(
            " · ".join(selected) if selected else "None"
        )
    )

    s1, s2, s3, s4 = st.columns(4)
    weights = interpretation.get("primary_weights") or {}
    with s1:
        st.metric("Accuracy weight", fmt(weights.get("accuracy"), 3))
    with s2:
        st.metric("Runtime weight", fmt(weights.get("runtime"), 3))
    with s3:
        st.metric("Energy weight", fmt(weights.get("energy"), 3))
    with s4:
        st.metric("CO2 weight", fmt(weights.get("co2"), 3))

    st.caption(
        "Selection status: {} · source: {} · model: {} · weighting policy: {} · fallback used: {}".format(
            interpretation.get("selection_status"),
            interpretation.get("selection_source"),
            interpretation.get("selection_model") or "none",
            interpretation.get("weighting_policy"),
            interpretation.get("fallback_used"),
        )
    )

    if parse_meta.get("warnings"):
        for warning in parse_meta["warnings"]:
            st.warning(warning)

    section(
        "Reviewable framework proposal",
        (
            "The ML Recommender receives the objective weights plus the current "
            "dataset meta-profile. It—not the LLM—produces the framework ranking."
        ),
    )

    cols = st.columns(4)
    with cols[0]:
        st.metric("Framework", str(proposal_dict.get("ml_recommender_framework")))
    with cols[1]:
        st.metric(
            "ML rank",
            "#{}".format(proposal_dict.get("ml_recommender_rank", 1)),
        )
    with cols[2]:
        st.metric("Utility", fmt(proposal_dict.get("ml_recommender_utility"), 4))
    with cols[3]:
        review_state = state.get("copilot_review")
        decision = (
            review_state.get("decision")
            if isinstance(review_state, dict)
            else "PROPOSED"
        )
        st.metric("Review state", decision or "PROPOSED")

    hcai = interpretation.get("hcai_requirements") or {}
    st.info(
        "HCAI requirements — drift sensitivity: {} · fairness required: {} · "
        "explainability: {}. These remain outside the four-objective selection benchmark.".format(
            hcai.get("drift_sensitivity"),
            hcai.get("fairness_required"),
            hcai.get("explainability_level"),
        )
    )

    st.markdown("**Grounded rationale**")
    st.write(humanize_rationale_text(proposal_dict.get("rationale")))
    evidence_chips(proposal_dict.get("evidence_keys"), show_raw=True)

    for warning in proposal_dict.get("warnings") or []:
        st.warning(warning)

    config = proposal_dict.get("proposed_config") or {}
    st.json(config, expanded=False)

    section(
        "Human review",
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
                goal_parser=GoalParser(),
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
