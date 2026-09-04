from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from awareml.llm import GroundedChat, ollama_status
from awareml.studies import StudyStore, TrustCalibrationStudy, classify_follow_up
from awareml.studies.information_seeking import THINK_ALOUD_PROMPTS

from .components import empty_state, hero, section
from .page_utils import phase_pills, plot
from .plots import apply_research_layout
from .state import ensure_research_state, result_dicts


def _state():
    return ensure_research_state()


def _researcher_mode() -> bool:
    return str(os.environ.get("AWAREML_STUDY_RESEARCHER_MODE", "0")).strip().lower() in {
        "1", "true", "yes", "on"
    }


def _ollama_controls(prefix: str):
    status = ollama_status()
    models = status.get("models") or []
    enabled = st.toggle(
        "Use local Ollama to verbalize grounded evidence",
        value=False,
        key="{}_enabled".format(prefix),
        help=(
            "When off, the lab uses deterministic evidence routing. When on, Ollama "
            "may verbalize only the structured facts already available to the lab."
        ),
    )
    default = _state().get("ollama_model") or status.get("resolved_model") or (
        models[0] if models else "llama3:8b"
    )
    model = default
    if enabled and models:
        model = st.selectbox(
            "Grounded response model",
            models,
            index=models.index(default) if default in models else 0,
            key="{}_model".format(prefix),
        )
        _state()["ollama_model"] = model
    if enabled and not status.get("reachable"):
        st.warning("Ollama is not reachable; deterministic grounded responses will be used.")
    return enabled, model, status


def _study_header_cards(values):
    cols = st.columns(len(values))
    for col, (label, value, help_text) in zip(cols, values):
        with col:
            st.metric(label, value)
            if help_text:
                st.caption(help_text)


def trust_calibration_research_page():
    hero(
        "HUMAN STUDY · CALIBRATED RELIANCE",
        "Trust Calibration",
        (
            "Test whether participant trust tracks recommendation reliability while "
            "holding explanation style constant. Operational recommendations are never "
            "overwritten by the study stimulus."
        ),
        pills=phase_pills(),
    )
    ranking = _state().get("ranking")
    if not ranking:
        empty_state(
            "Observed ranking required",
            "Open Decision Lab first so the study can construct matched stimuli from observed ranking evidence.",
        )
        return

    researcher_mode = _researcher_mode()
    if "r95_trust_case" not in st.session_state:
        st.session_state.r95_trust_case = None
    if "r95_trust_saved_count" not in st.session_state:
        st.session_state.r95_trust_saved_count = 0
    if "r95_trust_session" not in st.session_state:
        st.session_state.r95_trust_session = "pilot-session"

    _study_header_cards(
        [
            ("Study mode", "RESEARCHER" if researcher_mode else "PARTICIPANT", "Condition visibility"),
            ("Blinding", "OFF" if researcher_mode else "ON", "Reliability label hidden in participant mode"),
            ("Conditions", "3", "Correct · weak · wrong, matched wording"),
            ("Saved trials", str(st.session_state.r95_trust_saved_count), "Current Streamlit session"),
        ]
    )

    st.markdown(
        """
        <div class="r9-callout">
          <b>Study logic.</b> Each trial shows the same explanation structure while the
          recommended framework is manipulated to be the highest-, second-highest-, or
          lowest-utility candidate from the observed ranking. The participant reports
          trust, perceived correctness, decision confidence, acceptance, and whether
          more evidence would be needed before acting.
        </div>
        """,
        unsafe_allow_html=True,
    )

    setup, stimulus = st.columns([0.85, 1.45])
    with setup:
        section("1 · Trial setup", "Participant mode randomizes and hides the reliability condition.")
        session = st.text_input(
            "Participant / session code",
            value=st.session_state.r95_trust_session,
            key="r95_trust_session_input",
        )
        st.session_state.r95_trust_session = session

        if researcher_mode:
            condition = st.selectbox(
                "Reliability condition · researcher only",
                ["Randomized", "correct", "weak", "wrong"],
                key="r95_trust_condition_v3",
                help="Do not expose manual condition selection in participant sessions.",
            )
        else:
            condition = "Randomized"
            st.info("Condition assignment is randomized and hidden for participant blinding.")

        generate_label = (
            "Generate first randomized trial"
            if st.session_state.r95_trust_case is None
            else "Generate next matched trial"
        )
        if st.button(
            generate_label,
            type="primary",
            use_container_width=True,
            key="r95_trust_generate_v3",
        ):
            study = TrustCalibrationStudy(seed=int(time.time_ns() % 2_147_483_647))
            st.session_state.r95_trust_case = study.build_case(
                ranking,
                None if condition == "Randomized" else condition,
            )
            st.rerun()

        if st.button("Clear current trial", use_container_width=True, key="r95_trust_clear_v3"):
            st.session_state.r95_trust_case = None
            st.rerun()

    case = st.session_state.r95_trust_case
    with stimulus:
        section("2 · Recommendation stimulus", "Matched wording; only the recommendation reliability condition changes.")
        if case is None:
            st.info("Generate a trial to display the participant-facing recommendation.")
        else:
            st.markdown(
                """
                <div class="r9-callout" style="border-left:4px solid #2563eb;padding:18px 20px">
                  <div style="font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;opacity:.65">Recommendation shown to participant</div>
                  <div style="font-size:1.55rem;font-weight:750;margin:6px 0 10px">{}</div>
                  <div style="font-size:1.02rem;line-height:1.55">{}</div>
                </div>
                """.format(case.shown_framework, case.explanation),
                unsafe_allow_html=True,
            )
            st.caption(
                "Explanation wording and structure are intentionally held constant across reliability conditions."
            )

    if case is None:
        return

    section(
        "3 · Participant response",
        "Separate trust from perceived correctness, confidence and behavioral acceptance.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        trust = st.slider(
            "Trust in recommendation",
            1, 7, 4,
            key="r95_trust_score_v3",
            help="1 = very low trust; 7 = very high trust",
        )
    with c2:
        correctness = st.slider(
            "Perceived correctness",
            1, 7, 4,
            key="r95_trust_correct_v3",
            help="How correct does the recommendation appear?",
        )
    with c3:
        confidence = st.slider(
            "Decision confidence",
            1, 7, 4,
            key="r95_trust_conf_v3",
            help="How confident would you feel making a decision using this evidence?",
        )

    b1, b2 = st.columns(2)
    with b1:
        accept = st.radio(
            "Would you accept this recommendation?",
            ["Yes", "No"],
            horizontal=True,
            key="r95_trust_accept_v3",
        )
    with b2:
        seek_more = st.radio(
            "Would you seek more evidence before acting?",
            ["Yes", "No"],
            horizontal=True,
            key="r95_trust_more_v3",
        )

    note = st.text_area(
        "Optional participant rationale",
        height=90,
        key="r95_trust_rationale_v3",
        placeholder="What most influenced your trust or hesitation?",
    )

    save_col, integrity_col = st.columns([0.65, 1.35])
    with save_col:
        save = st.button(
            "Save trial response",
            type="primary",
            use_container_width=True,
            key="r95_trust_save_v3",
        )
    with integrity_col:
        st.info(
            "Responses are logged separately from the operational recommendation. "
            "The study stimulus never changes the live AwareML ranking."
        )

    if save:
        if not str(session or "").strip():
            st.error("Enter a participant/session code before saving.")
        else:
            StudyStore().log(
                "trust",
                session,
                "trial_response",
                {
                    **case.to_dict(),
                    "trust": trust,
                    "perceived_correctness": correctness,
                    "decision_confidence": confidence,
                    "accepted": accept == "Yes",
                    "seek_more_evidence": seek_more == "Yes",
                    "participant_rationale": note or None,
                    "participant_mode_blinded": not researcher_mode,
                },
            )
            st.session_state.r95_trust_saved_count += 1
            st.success("Trial response stored with the study audit trail.")

    if researcher_mode:
        with st.expander("Researcher-only manipulation check", expanded=False):
            utility_gap = case.utility_oracle - case.utility_shown
            st.json(
                {
                    "condition": case.condition,
                    "oracle_framework": case.oracle_framework,
                    "shown_framework": case.shown_framework,
                    "reliability": case.reliability,
                    "utility_shown": case.utility_shown,
                    "utility_oracle": case.utility_oracle,
                    "utility_gap_to_oracle": utility_gap,
                }
            )
            st.caption(
                "This block is hidden in participant mode. Enable researcher mode only for study setup/debugging."
            )


def information_seeking_research_page():
    hero(
        "HUMAN STUDY · EVIDENCE INTERACTION",
        "Information-Seeking Lab",
        (
            "Study how users interrogate a recommendation: asking for evidence, "
            "challenging it, comparing alternatives, clarifying terms, or deciding "
            "that they have enough information."
        ),
        pills=phase_pills(),
    )
    results = result_dicts()
    if not results:
        empty_state(
            "Run evidence required",
            "Run a benchmark first so every answer can be grounded in measured evidence.",
        )
        return

    if not _state().get("chat_session_id"):
        _state()["chat_session_id"] = str(uuid.uuid4())
    if "r95_chat_log" not in st.session_state:
        st.session_state.r95_chat_log = []

    use_llm, model, status = _ollama_controls("r95_info_v3")
    ranking = _state().get("ranking")
    chat = GroundedChat(model=model)
    facts = chat.build_facts(results, ranking)

    source_mode = "Grounded Ollama" if use_llm and status.get("reachable") else "Deterministic evidence router"
    _study_header_cards(
        [
            ("Evidence scope", "CURRENT RUN", "No synthetic benchmark values"),
            ("Frameworks", str(len(results)), "Measured candidates"),
            ("Response mode", source_mode, "Structured evidence only"),
            ("Session", str(_state()["chat_session_id"])[:8], "Anonymous local session id"),
        ]
    )

    st.markdown(
        """
        <div class="r9-callout">
          <b>Research question.</b> Does the participant request evidence, challenge the
          recommendation, compare alternatives, ask for clarification, or stop probing?
          The behavior category, response source and latency are recorded for each turn.
          Raw dataset rows are not sent to the response model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    def process_question(question: str):
        q = str(question or "").strip()
        if not q:
            return
        category = classify_follow_up(q)
        st.session_state.r95_chat_log.append(
            {"role": "user", "text": q, "category": category}
        )
        started = time.perf_counter()
        answer, meta = chat.answer(q, facts, use_llm=use_llm, category=category)
        latency = time.perf_counter() - started
        st.session_state.r95_chat_log.append(
            {
                "role": "assistant",
                "text": answer,
                "source": meta.get("source"),
                "model": meta.get("model"),
                "category": category,
                "latency_sec": latency,
            }
        )
        StudyStore().log(
            "information_seeking",
            _state()["chat_session_id"],
            "chat_turn",
            {
                "question": q,
                "category": category,
                "answer_source": meta.get("source"),
                "model": meta.get("model"),
                "response_time_sec": latency,
                "turn": len([x for x in st.session_state.r95_chat_log if x["role"] == "user"]),
            },
        )

    section("Evidence probes", "Quick probes make it easy to test distinct information needs without retyping them.")
    quick = st.columns(5)
    quick_questions = [
        ("Why #1?", "Why was the top-ranked framework recommended?"),
        ("Compare top 2", "Compare the top two ranked frameworks using the measured evidence."),
        ("Fairness", "Show me the fairness evidence and any unavailable fairness criteria."),
        ("Drift", "What drift and recovery evidence was observed?"),
        ("Sustainability", "Compare energy and CO2 evidence for the leading frameworks."),
    ]
    pending = None
    for col, (label, question) in zip(quick, quick_questions):
        with col:
            if st.button(label, use_container_width=True, key="r95_info_quick_{}".format(label)):
                pending = question
    if pending:
        process_question(pending)
        st.rerun()

    left, right = st.columns([1.65, 0.85])
    with left:
        section("Grounded evidence conversation", "Different question types should now produce different evidence-grounded answers.")
        if not st.session_state.r95_chat_log:
            st.info(
                "Start with a quick probe above or type a question below. Good study probes include: "
                "'Why?', 'Show me the evidence', 'Compare alternatives', and 'What would change your recommendation?'"
            )
        for item in st.session_state.r95_chat_log:
            with st.chat_message(item["role"]):
                st.markdown(item["text"])
                if item["role"] == "assistant":
                    st.caption(
                        "Source: {}{} · behavior: {} · {:.2f}s".format(
                            item.get("source") or "unknown",
                            " · model: {}".format(item.get("model")) if item.get("model") else "",
                            item.get("category") or "unclassified",
                            float(item.get("latency_sec") or 0.0),
                        )
                    )
        q = st.chat_input(
            "Ask for evidence, compare alternatives, challenge the ranking, or ask about fairness, drift, XAI or sustainability…"
        )
        if q:
            process_question(q)
            st.rerun()

    with right:
        section("Behavior dashboard", "Live descriptive summary for the current study session.")
        user_turns = [x for x in st.session_state.r95_chat_log if x["role"] == "user"]
        cats = [x.get("category") or classify_follow_up(x.get("text", "")) for x in user_turns]
        unique_categories = len(set(cats)) if cats else 0
        d1, d2 = st.columns(2)
        d1.metric("Follow-up depth", len(user_turns))
        d2.metric("Behavior types", unique_categories)

        if cats:
            counts = pd.Series(cats).value_counts().rename_axis("Behavior").reset_index(name="Count")
            fig = px.bar(
                counts,
                x="Count",
                y="Behavior",
                orientation="h",
                text_auto=True,
                title="Observed follow-up behavior",
            )
            apply_research_layout(
                fig,
                height=300,
                legend="none",
                title="Observed follow-up behavior",
                bottom_margin=48,
            )
            fig.update_layout(margin=dict(l=118, r=24, t=54, b=50))
            plot(fig, "r95_info_behavior_v3")
        else:
            st.caption("No follow-up behavior has been recorded yet.")

        with st.expander("Behavior taxonomy", expanded=False):
            st.markdown(
                """
                - **Evidence request:** asks for source/data/proof.
                - **Explanation probe:** asks why/how.
                - **Challenge:** questions or disputes the recommendation.
                - **Comparison/counterfactual:** compares alternatives or asks what-if.
                - **Clarification:** asks for a definition or meaning.
                """
            )

        with st.expander("Think-aloud prompts", expanded=False):
            for prompt in THINK_ALOUD_PROMPTS:
                st.markdown("- {}".format(prompt))

        accepted = st.checkbox(
            "Participant accepted the first answer without further probing",
            value=False,
            key="r95_info_accepted_v3",
        )
        endpoint_note = st.text_area(
            "Optional session endpoint note",
            height=70,
            key="r95_info_endpoint_note_v3",
        )
        if st.button(
            "Save session endpoint",
            type="primary",
            use_container_width=True,
            key="r95_info_save_v3",
        ):
            StudyStore().log(
                "information_seeking",
                _state()["chat_session_id"],
                "session_end",
                {
                    "first_answer_accepted": accepted,
                    "follow_up_depth": len(user_turns),
                    "unique_behavior_types": unique_categories,
                    "endpoint_note": endpoint_note or None,
                },
            )
            st.success("Session endpoint stored.")

        if st.button("Start new conversation", use_container_width=True, key="r95_info_reset_v3"):
            st.session_state.r95_chat_log = []
            _state()["chat_session_id"] = str(uuid.uuid4())
            st.rerun()
