from __future__ import annotations

import streamlit as st

from awareml.llm import (
    CopilotService,
    GoalParser,
    GroundedCopilotChat,
    OllamaClient,
    ReviewStore,
)

from .components import empty_state, evidence_chips, hero, section, status_panel, humanize_rationale_text
from .data import load_v2_recommender
from .page_utils import dataset_ready, fmt, phase_pills
from .state import ROOT, ensure_research_state



def copilot_workspace_page():
    state = ensure_research_state()

    hero(
        "HUMAN-CENTRIC AI",
        "Copilot Workspace",
        (
            "Translate a natural-language goal into a reviewable configuration, "
            "ground the recommendation in Phase-6 empirical evidence, and retain "
            "the human approval/edit/reject gate."
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
          <b>Copilot vs ML Recommender:</b> both start from the same dataset context, but they serve different roles.
          <b>3D Decision Space</b> lets you manually set the four objective weights and instantly rerank the frozen predictions.
          <b>Copilot Workspace</b> first interprets a natural-language goal, converts it into weights plus HCAI requirements,
          then proposes a configuration that must still pass through human review.
        </div>
        """,
        unsafe_allow_html=True,
    )

    client = OllamaClient(model=state.get("ollama_model") or None)
    ollama = client.status()

    left, right = st.columns([1.05, 0.95])

    with left:
        section(
            "Goal",
            "The LLM interprets intent. The ML recommender remains the empirical framework-selection component.",
        )
        st.markdown(
            '<div class="r9-field-label">What do you want from the streaming AutoML system?</div>'
            '<div class="r9-field-help">Describe priorities such as accuracy, drift adaptation, runtime, energy, CO₂, fairness and explainability.</div>',
            unsafe_allow_html=True,
        )
        goal = st.text_area(
            "Streaming AutoML goal",
            value=state.get("copilot_goal")
            or (
                "I need a highly accurate streaming classifier that adapts to drift, "
                "keeps energy use moderate, is fair, and provides understandable explanations."
            ),
            height=145,
            key="r9_copilot_goal",
            label_visibility="collapsed",
            placeholder="Example: Prioritize high accuracy and fast drift adaptation while keeping energy and CO₂ moderate, and require fairness auditing and understandable explanations.",
        )
        state["copilot_goal"] = goal

        use_llm = st.toggle(
            "Use Ollama to interpret the goal",
            value=False,
            key="r9_copilot_llm",
            help=(
                "For reproducible demos, leave this off. The deterministic parser "
                "produces fixed objective weights. Turning it on allows Ollama to "
                "interpret the same text differently, which can change the ML ranking."
            ),
        )
        st.caption(
            "Recommended for supervisor demos: deterministic goal interpretation. "
            "Ollama does not directly choose a framework; it can only change the "
            "objective weights supplied to the frozen Phase-6 ML recommender."
        )
        models = ollama.get("models") or []
        model = state.get("ollama_model") or ollama.get("resolved_model")
        if use_llm and models:
            model = st.selectbox(
                "Ollama model",
                models,
                index=models.index(ollama.get("resolved_model")) if ollama.get("resolved_model") in models else 0,
                key="r9_copilot_model",
            )
            state["ollama_model"] = model

        if st.button(
            "Generate Copilot proposal",
            type="primary",
            use_container_width=True,
            key="r9_generate_proposal",
        ):
            try:
                local_client = OllamaClient(model=model)
                service = CopilotService(
                    recommender=load_v2_recommender(),
                    goal_parser=GoalParser(client=local_client),
                    chat=GroundedCopilotChat(client=local_client),
                    review_store=ReviewStore(ROOT / "artifacts" / "copilot" / "reviews.jsonl"),
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
                st.error("Copilot proposal failed: {}".format(exc))

    with right:
        status_panel({
            "Ollama": "connected" if ollama.get("reachable") else "fallback only",
            "Configured model": str(ollama.get("configured_model")),
            "Resolved model": str(ollama.get("resolved_model")),
            "Raw dataset rows to LLM": "False",
            "Human review gate": "Required",
        })
        st.markdown(
            """
            <div class="r9-callout" style="margin-top:14px">
              Fairness and explainability are HCAI requirements/evidence.
              The frozen ML recommender still has exactly four empirical
              objectives: Accuracy, Runtime, Energy and CO₂.
            </div>
            """,
            unsafe_allow_html=True,
        )

    proposal = state.get("copilot_proposal")
    if proposal is None:
        return

    proposal_dict = proposal.model_dump() if hasattr(proposal, "model_dump") else proposal

    section(
        "Reviewable proposal",
        "The configuration remains proposed until you explicitly approve, edit or reject it.",
    )

    cols = st.columns(4)
    with cols[0]:
        st.metric("Framework", str(proposal_dict.get("ml_recommender_framework")))
    with cols[1]:
        st.metric("ML rank", "#{}".format(proposal_dict.get("ml_recommender_rank", 1)))
    with cols[2]:
        st.metric("Utility", fmt(proposal_dict.get("ml_recommender_utility"), 4))
    with cols[3]:
        review_state = state.get("copilot_review")
        decision = review_state.get("decision") if isinstance(review_state, dict) else "PROPOSED"
        st.metric("Review state", decision or "PROPOSED")

    interpretation = proposal_dict.get("interpretation") or {}
    weights = (interpretation.get("primary_weights") or {})
    parse_meta = ((state.get("copilot_meta") or {}).get("goal_parse") or {})
    parser_source = parse_meta.get("source", "unknown")
    parser_model = parse_meta.get("model")

    st.markdown("**How the framework was chosen**")
    st.markdown(
        (
            "The **LLM does not directly select the framework**. First, the goal is "
            "converted into four empirical preference weights. Then the frozen "
            "**Phase-6 ML Recommender V2** ranks the five frameworks using predicted "
            "Accuracy, Runtime, Energy and CO₂."
        )
    )

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.metric("Accuracy weight", fmt(weights.get("accuracy"), 3))
    with w2:
        st.metric("Runtime weight", fmt(weights.get("runtime"), 3))
    with w3:
        st.metric("Energy weight", fmt(weights.get("energy"), 3))
    with w4:
        st.metric("CO₂ weight", fmt(weights.get("co2"), 3))

    st.caption(
        "Goal parser: {}{}".format(
            parser_source,
            " · model: {}".format(parser_model) if parser_model else "",
        )
    )

    drift_sensitivity = interpretation.get("drift_sensitivity")
    fairness_required = interpretation.get("fairness_required")
    explainability_level = interpretation.get("explainability_level")
    st.info(
        "HCAI requirements from the goal — drift sensitivity: {} · fairness required: {} · "
        "explainability: {}. These requirements are retained in the configuration/evidence "
        "layer; they are not hidden extra objectives in the four-objective utility.".format(
            drift_sensitivity,
            fairness_required,
            explainability_level,
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
        "Approved-with-edits records the configuration diff in the Phase-7 audit trail.",
    )
    review_mode = st.segmented_control(
        "Decision",
        ["Approve", "Approve with edits", "Reject"],
        default="Approve",
        key="r9_review_mode",
    )

    edits = None
    if review_mode == "Approve with edits":
        edited_window = st.number_input(
            "Window size",
            min_value=50,
            max_value=100000,
            value=int(config.get("window_size", 1000)),
            step=50,
            key="r9_review_window",
        )
        edited_budget = st.number_input(
            "Time budget (seconds)",
            min_value=1.0,
            max_value=86400.0,
            value=float(config.get("time_budget_sec", 60.0)),
            step=5.0,
            key="r9_review_budget",
        )
        edits = {
            "window_size": int(edited_window),
            "time_budget_sec": float(edited_budget),
        }

    note = st.text_input("Review note", key="r9_review_note")

    if st.button("Record human review", key="r9_record_review"):
        try:
            local_client = OllamaClient(model=state.get("ollama_model") or None)
            service = CopilotService(
                recommender=load_v2_recommender(),
                goal_parser=GoalParser(client=local_client),
                chat=GroundedCopilotChat(client=local_client),
                review_store=ReviewStore(ROOT / "artifacts" / "copilot" / "reviews.jsonl"),
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
