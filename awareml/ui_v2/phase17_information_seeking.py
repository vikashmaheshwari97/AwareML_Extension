from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from awareml.llm import GroundedChat
from awareml.studies.information_seeking import (
    CATEGORY_DESCRIPTIONS,
    CATEGORY_LABELS,
    THINK_ALOUD_PROMPTS,
    classify_follow_up,
)
from awareml.studies.information_seeking_analysis import (
    analyze_information_seeking,
    manual_coding,
    session_summaries,
)
from awareml.studies.information_seeking_context import (
    context_path,
    load_study_context,
    resolve_study_context,
    save_study_context,
)
from awareml.studies.information_seeking_grounding import (
    TOPIC_LABELS,
    grounded_information_answer,
)
from awareml.studies.store import StudyStore

from .components import empty_state, hero, section
from .plots import apply_research_layout
from .state import ensure_research_state, result_dicts


ROOT = Path(__file__).resolve().parents[2]
DESIGN_DIR = ROOT / "data" / "journal" / "information_seeking_v1" / "design"
PROTOCOL_PATH = DESIGN_DIR / "protocol.json"
CODING_SCHEME_PATH = DESIGN_DIR / "coding_scheme.json"
FINAL_DESIGN_MANIFEST = (
    ROOT / "data" / "journal" / "information_seeking_v1" / "frozen" / "design_manifest.json"
)
LOCAL_PILOT_RESEARCHER_KEY = "phase16-local-test-key"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _protocol() -> Dict[str, Any]:
    return _load_json(PROTOCOL_PATH)


def _coding_scheme() -> Dict[str, Any]:
    return _load_json(CODING_SCHEME_PATH)


def _state():
    return ensure_research_state()


def _study_name(mode: str) -> str:
    return "information_seeking_{}".format(mode)


def _final_collection_ready() -> bool:
    return (
        FINAL_DESIGN_MANIFEST.exists()
        and str(os.getenv("AWAREML_INFORMATION_SEEKING_FINAL_ARMED", "")).strip().upper()
        == "YES"
    )


def _resolve_collection_mode() -> str:
    requested = str(
        os.getenv("AWAREML_INFORMATION_SEEKING_COLLECTION_MODE", "pilot")
    ).strip().lower()
    if requested == "final" and _final_collection_ready():
        return "final"
    return "pilot"


def _configured_researcher_key() -> Optional[str]:
    explicit = str(os.getenv("AWAREML_STUDY_RESEARCHER_KEY", "")).strip()
    if explicit:
        return explicit
    if not FINAL_DESIGN_MANIFEST.exists() and _resolve_collection_mode() == "pilot":
        return LOCAL_PILOT_RESEARCHER_KEY
    return None


def _researcher_unlocked() -> bool:
    return bool(st.session_state.get("p17_researcher_unlocked"))


def _set_access(view: str) -> None:
    st.session_state["p17_access"] = view
    st.rerun()


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return ("{:.{digits}f}".format(float(value), digits=digits))
    except Exception:
        return "N/A"


def _top_context(results: List[Dict[str, Any]], ranking: List[Dict[str, Any]]):
    if not ranking:
        return None, None
    ordered = sorted(
        ranking,
        key=lambda row: (
            float(row.get("rank")) if row.get("rank") is not None else 999999,
            -float(row.get("utility") or 0.0),
        ),
    )
    top = ordered[0]
    framework = str(top.get("framework") or top.get("Framework") or "")
    run = None
    for row in results:
        if str(row.get("framework") or "") == framework:
            run = row
            break
    return top, run


def _initial_explanation(top: Dict[str, Any]) -> str:
    framework = str(top.get("framework") or top.get("Framework") or "the leading framework")
    utility = top.get("utility")
    if utility is None:
        return (
            "{} is ranked first in the current observed post-run ranking. "
            "You may ask for more evidence before deciding whether to rely on this recommendation."
        ).format(framework)
    return (
        "{} is ranked first in the current observed post-run ranking with utility {}. "
        "You may ask for more evidence, compare alternatives, challenge the recommendation, "
        "or request clarification before deciding whether to rely on it."
    ).format(framework, _fmt(utility, 3))


def _ranking_digest(ranking: List[Dict[str, Any]]) -> str:
    raw = json.dumps(ranking or [], sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _register_evidence_view(area: str) -> None:
    viewed = st.session_state.setdefault("p17_evidence_viewed", [])
    if area not in viewed:
        viewed.append(area)
        started = float(st.session_state.get("p17_started_epoch") or time.time())
        StudyStore().log(
            _study_name(_resolve_collection_mode()),
            st.session_state["p17_session_code"],
            "evidence_view",
            {
                "evidence_area": area,
                "time_since_session_start_sec": max(0.0, time.time() - started),
            },
        )


def _evidence_block(area: str, top: Dict[str, Any], run: Optional[Dict[str, Any]]) -> None:
    framework = str(top.get("framework") or top.get("Framework") or "N/A")
    run = run or {}
    if area == "performance":
        frame = pd.DataFrame(
            [
                {
                    "Framework": framework,
                    "Accuracy": run.get("accuracy", top.get("accuracy")),
                    "Macro-F1": run.get("f1_macro"),
                    "Runtime (s)": run.get("runtime_sec", top.get("runtime_sec")),
                    "Observed utility": top.get("utility"),
                }
            ]
        )
    elif area == "fairness":
        fairness = run.get("fairness") or {}
        frame = pd.DataFrame(
            [
                {
                    "Framework": framework,
                    "Demographic parity gap": fairness.get("dp_diff"),
                    "Equal opportunity gap": fairness.get("equal_opportunity_diff"),
                    "Equalized odds gap": fairness.get("equalized_odds_gap"),
                    "Worst-group accuracy": fairness.get("worst_group_accuracy"),
                    "Group Brier gap": fairness.get("group_brier_score_gap"),
                    "Group ECE gap": fairness.get("group_ece_gap"),
                }
            ]
        )
    elif area == "sustainability":
        frame = pd.DataFrame(
            [
                {
                    "Framework": framework,
                    "Runtime (s)": run.get("runtime_sec"),
                    "Energy (kWh)": run.get("energy_kwh"),
                    "CO2 (kg)": run.get("co2_kg"),
                }
            ]
        )
    elif area == "drift":
        drift = run.get("drift_summary") or {}
        frame = pd.DataFrame(
            [
                {
                    "Framework": framework,
                    "Drift events": len(run.get("drift_events") or []),
                    "Recovery rate": drift.get("recovery_rate"),
                    "Median recovery samples": drift.get("median_recovery_samples"),
                    "Mean accuracy drop": drift.get("mean_accuracy_drop"),
                }
            ]
        )
    else:
        xai = run.get("explainability") or {}
        frame = pd.DataFrame(
            [
                {
                    "Framework": framework,
                    "XAI status": xai.get("status"),
                    "Method": xai.get("method"),
                    "Fidelity": xai.get("fidelity"),
                    "Stability": xai.get("stability"),
                    "Top feature": xai.get("top_feature"),
                }
            ]
        )
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _process_question(
    question: str,
    results: List[Dict[str, Any]],
    ranking: List[Dict[str, Any]],
) -> None:
    q = str(question or "").strip()
    if not q:
        return
    category = classify_follow_up(q)
    log = st.session_state.setdefault("p17_chat_log", [])
    user_turn_count = len([item for item in log if item.get("role") == "user"])
    turn = user_turn_count + 1
    started_epoch = float(st.session_state.get("p17_started_epoch") or time.time())
    elapsed = max(0.0, time.time() - started_epoch)

    answer_started = time.perf_counter()
    answer, requested_topics, target_frameworks = grounded_information_answer(
        q,
        category,
        results,
        ranking,
    )
    answer_latency = max(0.0, time.perf_counter() - answer_started)

    log.append(
        {
            "role": "user",
            "text": q,
            "category": category,
            "requested_topics": requested_topics,
            "turn": turn,
            "elapsed_sec": elapsed,
        }
    )
    log.append(
        {
            "role": "assistant",
            "text": answer,
            "category": category,
            "requested_topics": requested_topics,
            "target_frameworks": target_frameworks,
            "source": "deterministic-grounded-multi-topic",
            "latency_sec": answer_latency,
        }
    )
    if turn == 1:
        st.session_state["p17_first_follow_up_category"] = category
        st.session_state["p17_time_before_first_follow_up_sec"] = elapsed

    StudyStore().log(
        _study_name(_resolve_collection_mode()),
        st.session_state["p17_session_code"],
        "chat_turn",
        {
            "question": q,
            "category": category,
            "category_label": CATEGORY_LABELS.get(category, category),
            "requested_topics": requested_topics,
            "target_frameworks": target_frameworks,
            "classifier_role": "analysis_assist_only",
            "grounding_mode": "phase17_multi_topic_grounding_v1",
            "turn": turn,
            "time_since_session_start_sec": elapsed,
            "answer_source": "deterministic-grounded-multi-topic",
            "response_time_sec": answer_latency,
        },
    )


def _participant_intro(mode: str) -> None:
    label = "Pilot Study" if mode == "pilot" else "Main Study"
    st.markdown(
        """
        <div class="r9-callout">
          <b>{}</b> · This study observes when people ask for more information before
          accepting, changing or rejecting an AwareML recommendation. Ask only the
          follow-ups you genuinely need; asking no follow-up questions is also a valid
          study outcome.
        </div>
        """.format(label),
        unsafe_allow_html=True,
    )


def _participant_page(results: List[Dict[str, Any]], ranking: List[Dict[str, Any]], context_source: str = "current_session", saved_context: Optional[Dict[str, Any]] = None) -> None:
    mode = _resolve_collection_mode()
    top, run = _top_context(results, ranking)
    if top is None:
        empty_state(
            "Observed ranking required",
            "Run the benchmark and open Decision Lab first so the study starts from an observed recommendation.",
        )
        return

    _participant_intro(mode)

    if st.session_state.get("p17_completed"):
        st.success(
            "Study session complete. Your behavior log has been saved under a pseudonymous session identifier."
        )
        c1, c2 = st.columns(2)
        if c1.button("Return to study home", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("p17_") and key not in {"p17_access"}:
                    st.session_state.pop(key, None)
            st.session_state["p17_access"] = "home"
            st.rerun()
        if c2.button("Start another Pilot session", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("p17_") and key not in {"p17_access"}:
                    st.session_state.pop(key, None)
            st.rerun()
        return

    if not st.session_state.get("p17_started"):
        section(
            "Before you begin",
            "Use the participant/session code supplied by the study team. If this follows Trust Calibration, use the same anonymous code.",
        )
        with st.expander("What will I do?", expanded=True):
            st.markdown(
                """
                You will first see the current AwareML recommendation. You may then:
                - ask **why/how** questions;
                - request **evidence or measurements**;
                - **compare** alternatives or ask a what-if question;
                - **challenge** the recommendation;
                - ask for a **clarification**;
                - or decide that you already have enough information.

                There is no required number of follow-ups. Asking **zero** follow-ups is
                a valid behavior and is recorded as such.
                """
            )
        materials = (_protocol().get("participant_materials") or {})
        with st.expander("Study information and consent", expanded=False):
            st.markdown(
                str(
                    materials.get("consent_text")
                    or "Pilot participant information is pending finalization."
                )
            )
            if mode == "pilot":
                st.caption(
                    "Pilot wording only. Final participant information and consent must be approved before Main Study recruitment."
                )
        code = st.text_input(
            "Participant / session code",
            value=st.session_state.get("p17_session_code", ""),
            key="p17_session_code_input",
        )
        group = st.selectbox(
            "Experience group",
            [
                "Novice / student",
                "ML practitioner",
                "AutoML user",
                "ML / AutoML expert",
                "Other",
            ],
            key="p17_experience_group",
        )
        expertise = st.slider(
            "Self-rated ML / AutoML expertise",
            1,
            5,
            3,
            key="p17_expertise_score",
            help="1 = very limited experience; 5 = highly experienced.",
        )
        consent = st.checkbox(
            "I have read the study information and consent to participate.",
            key="p17_consent",
        )
        if mode == "pilot":
            st.warning(
                "Pilot Study. These responses are used to test the study workflow and are stored separately from Main Study data."
            )
        if st.button(
            "Start information-seeking study",
            type="primary",
            use_container_width=True,
            key="p17_start",
        ):
            if not str(code or "").strip():
                st.error("Enter the anonymous participant/session code.")
            elif not consent:
                st.error("Consent is required before starting.")
            else:
                st.session_state["p17_session_code"] = str(code).strip()
                st.session_state["p17_started"] = True
                st.session_state["p17_started_epoch"] = time.time()
                st.session_state["p17_chat_log"] = []
                st.session_state["p17_evidence_viewed"] = []
                st.session_state["p17_first_follow_up_category"] = None
                st.session_state["p17_time_before_first_follow_up_sec"] = None
                StudyStore().log(
                    _study_name(mode),
                    st.session_state["p17_session_code"],
                    "session_start",
                    {
                        "experience_group": group,
                        "expertise_score": expertise,
                        "collection_mode": mode,
                        "initial_recommendation_framework": top.get("framework"),
                        "initial_recommendation_utility": top.get("utility"),
                        "initial_explanation": _initial_explanation(top),
                        "ranking_sha256": _ranking_digest(ranking),
                        "study_context_source": context_source,
                        "study_context_sha256": (saved_context or {}).get("sha256_without_self"),
                        "response_mode": "deterministic_grounded_multi_topic_router",
                    },
                )
                st.rerun()
        return

    st.caption(
        "{} follow-up{} recorded · {} evidence area{} viewed".format(
            len([x for x in st.session_state.get("p17_chat_log", []) if x.get("role") == "user"]),
            "" if len([x for x in st.session_state.get("p17_chat_log", []) if x.get("role") == "user"]) == 1 else "s",
            len(st.session_state.get("p17_evidence_viewed", [])),
            "" if len(st.session_state.get("p17_evidence_viewed", [])) == 1 else "s",
        )
    )

    st.markdown(
        """
        <div class="r9-panel" style="margin-bottom:1rem">
          <div class="r9-card-label">INITIAL AWAREML RECOMMENDATION</div>
          <div style="font-size:1.45rem;font-weight:760;margin:.45rem 0">{}</div>
          <div class="r9-card-note">{}</div>
        </div>
        """.format(
            html.escape(str(top.get("framework") or "N/A")),
            html.escape(_initial_explanation(top)),
        ),
        unsafe_allow_html=True,
    )

    section(
        "Explore evidence",
        "Opening evidence is optional. Every evidence area you choose to inspect is logged.",
    )
    areas = [
        ("Performance", "performance"),
        ("Fairness", "fairness"),
        ("Sustainability", "sustainability"),
        ("Drift", "drift"),
        ("Explainability", "explainability"),
    ]
    cols = st.columns(5)
    for col, (label, area) in zip(cols, areas):
        with col:
            if st.button(
                "View {}".format(label),
                use_container_width=True,
                key="p17_evidence_{}".format(area),
            ):
                _register_evidence_view(area)
                st.session_state["p17_active_evidence"] = area
                st.rerun()

    active = st.session_state.get("p17_active_evidence")
    if active:
        st.markdown("**{} evidence**".format(active.replace("_", " ").title()))
        _evidence_block(active, top, run)

    section(
        "Ask for more information",
        "Use a quick question or type your own. There is no required number of questions.",
    )
    quick_questions = [
        ("Show evidence", "Show me the measured evidence supporting the top recommendation."),
        ("Why this choice?", "Why is this framework ranked first?"),
        ("Compare top 2", "Compare the top two ranked frameworks using the measured evidence."),
        ("Challenge", "Are you sure this recommendation is the best choice? What evidence could challenge it?"),
        ("Clarify", "Explain the recommendation in simpler terms and define the most important metric."),
    ]
    pending = None
    with st.expander("Optional example follow-up questions", expanded=False):
        st.caption(
            "Use these only if they match something you genuinely want to ask. You can also type your own question below."
        )
        qcols = st.columns(5)
        for col, (label, question) in zip(qcols, quick_questions):
            with col:
                if st.button(
                    label,
                    use_container_width=True,
                    key="p17_quick_{}".format(label),
                ):
                    pending = question
    if pending:
        _process_question(pending, results, ranking)
        st.rerun()

    for item in st.session_state.get("p17_chat_log", []):
        with st.chat_message(item.get("role") or "assistant"):
            st.markdown(str(item.get("text") or ""))
            if item.get("role") == "assistant":
                topics = item.get("requested_topics") or []
                if topics:
                    st.caption(
                        "Grounded response · evidence covered: {}".format(
                            ", ".join(TOPIC_LABELS.get(topic, topic) for topic in topics)
                        )
                    )
                else:
                    st.caption("Grounded response from the current measured study context.")
    question = st.chat_input(
        "Ask for evidence, compare alternatives, challenge the recommendation, or request clarification..."
    )
    if question:
        _process_question(question, results, ranking)
        st.rerun()

    section(
        "Your decision",
        "Finish whenever you feel you have enough information. Zero follow-ups is a valid outcome.",
    )
    decision = st.radio(
        "What would you do with the current recommendation?",
        [
            "Accept — I would rely on it as shown",
            "Override — I would change/correct it before acting",
            "Reject — I would not rely on it",
        ],
        key="p17_decision",
    )
    confidence = st.slider(
        "How confident are you in this final decision?",
        1,
        7,
        4,
        key="p17_final_confidence",
    )
    rationale = st.text_area(
        "What made you ask for more information — or decide not to ask?",
        height=100,
        key="p17_rationale",
        placeholder="A short explanation is useful for qualitative analysis.",
    )
    if st.button(
        "Finish and save study session",
        type="primary",
        use_container_width=True,
        key="p17_finish",
    ):
        user_turns = [
            x
            for x in st.session_state.get("p17_chat_log", [])
            if x.get("role") == "user"
        ]
        decision_key = str(decision).split("—", 1)[0].strip().lower()
        evidence_viewed = list(
            st.session_state.get("p17_evidence_viewed", [])
        )
        payload = {
            "follow_up_occurred": bool(user_turns),
            "information_seeking_occurred": bool(user_turns or evidence_viewed),
            "follow_up_count": len(user_turns),
            "evidence_view_count": len(evidence_viewed),
            "first_follow_up_category": st.session_state.get(
                "p17_first_follow_up_category"
            ),
            "time_before_first_follow_up_sec": st.session_state.get(
                "p17_time_before_first_follow_up_sec"
            ),
            "evidence_viewed": evidence_viewed,
            "recommendation_decision": decision_key,
            "recommendation_accepted": decision_key == "accept",
            "final_confidence": confidence,
            "participant_rationale": rationale or None,
            "classifier_role": "analysis_assist_only",
        }
        StudyStore().log(
            _study_name(mode),
            st.session_state["p17_session_code"],
            "session_end",
            payload,
        )
        st.session_state["p17_completed"] = True
        st.rerun()


def _researcher_page(current_results=None, current_ranking=None) -> None:
    mode = _resolve_collection_mode()
    configured_key = _configured_researcher_key()

    st.markdown(
        """
        <div class="r9-panel" style="margin-bottom:1rem">
          <b>Researcher Workspace</b><br>
          <span class="r9-card-note">Study administration, behavior summaries,
          qualitative coding and exports. This area is not participant-facing.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not _researcher_unlocked():
        if not configured_key:
            st.error(
                "Researcher access is not configured. Add AWAREML_STUDY_RESEARCHER_KEY to the local .env or deployment secrets."
            )
            return
        entered = st.text_input(
            "Researcher key",
            type="password",
            key="p17_researcher_key_input",
        )
        if not FINAL_DESIGN_MANIFEST.exists():
            st.caption(
                "Local Pilot Study access uses the same convenience key as Trust Calibration."
            )
        if st.button(
            "Unlock researcher workspace",
            type="primary",
            key="p17_unlock",
        ):
            if hmac.compare_digest(str(entered or ""), str(configured_key)):
                st.session_state["p17_researcher_unlocked"] = True
                st.rerun()
            else:
                st.error("Incorrect researcher key.")
        return

    st.success(
        "Researcher workspace unlocked. Automated behavior categories assist analysis but do not replace qualitative coding."
    )

    protocol = _protocol()
    coding = _coding_scheme()
    pcols = st.columns(4)
    pcols[0].metric("Protocol", str(protocol.get("status") or "unknown").replace("_", " ").title())
    target = protocol.get("participant_target") or {}
    pcols[1].metric(
        "Suggested sample",
        "{}–{}".format(target.get("suggested_min", 10), target.get("suggested_max", 15)),
    )
    pcols[2].metric(
        "Coding scheme",
        str(coding.get("status") or "unknown").replace("_", " ").title(),
    )
    pcols[3].metric(
        "Design lock",
        "Frozen" if FINAL_DESIGN_MANIFEST.exists() else "Not frozen",
    )

    section(
        "Study evidence context",
        "For Pilot testing you may use the current session directly. For repeated participant sessions, save one standardized measured context so participants do not need to rerun the benchmark.",
    )
    saved_context = load_study_context(mode)
    if saved_context:
        st.success(
            "Saved {} study context is available · {} frameworks · captured {}.".format(
                "Pilot" if mode == "pilot" else "Main",
                len(saved_context.get("results") or []),
                saved_context.get("created_utc") or "time unavailable",
            )
        )
        st.caption("Context file: {}".format(context_path(mode)))
    else:
        st.info(
            "No saved {} study context yet.".format(
                "Pilot" if mode == "pilot" else "Main"
            )
        )

    current_results = list(current_results or [])
    current_ranking = list(current_ranking or [])
    if current_results and current_ranking:
        if st.button(
            "Save current measured run as {} study context".format(
                "Pilot" if mode == "pilot" else "Main"
            ),
            type="primary",
            use_container_width=True,
            key="p17_save_study_context_{}".format(mode),
        ):
            saved = save_study_context(mode, current_results, current_ranking)
            st.success(
                "Standardized study context saved. New participant sessions on this deployment can use it without rerunning Run Studio."
            )
            st.rerun()
    else:
        st.caption(
            "To create a context: run the agreed benchmark in Run Studio, open Decision Lab once to create the observed ranking, then return here."
        )

    with st.expander("Study protocol and coding scheme", expanded=False):
        st.markdown(
            "**Research question:** {}".format(
                protocol.get("central_rq") or "N/A"
            )
        )
        st.markdown(
            "**Classifier role:** analysis aid only. Manual qualitative coding remains authoritative."
        )
        st.json(
            {
                "participant_target": target,
                "measurements": protocol.get("measurements"),
                "classifier": protocol.get("classifier"),
                "qualitative_analysis": protocol.get("qualitative_analysis"),
            },
            expanded=False,
        )
        st.markdown("**Manual qualitative codes**")
        for code in coding.get("codes") or []:
            st.markdown(
                "- **{}:** {}".format(code.get("label"), code.get("definition"))
            )

    dataset_mode = st.radio(
        "Researcher dataset",
        ["Pilot Study", "Main Study"],
        horizontal=True,
        key="p17_researcher_dataset",
    )
    selected_mode = "pilot" if dataset_mode == "Pilot Study" else "final"
    study_name = _study_name(selected_mode)
    events = StudyStore().export(study_name)
    summaries = session_summaries(events)
    analysis = analyze_information_seeking(events)

    section(
        "Collection status",
        "Pilot and Main Study records are stored under separate study names in the append-only study store.",
    )
    completed = summaries[summaries["completed"] == True] if not summaries.empty else summaries
    cards = st.columns(6)
    cards[0].metric("Registered", analysis.get("registered_sessions", 0))
    cards[1].metric("Completed", analysis.get("completed_sessions", 0))
    follow_rate = analysis.get("follow_up_rate")
    cards[2].metric(
        "Asked follow-up",
        "N/A" if follow_rate is None else "{:.1%}".format(follow_rate),
    )
    mean_count = analysis.get("mean_follow_up_count")
    cards[3].metric(
        "Mean follow-ups",
        "N/A" if mean_count is None else "{:.2f}".format(mean_count),
    )
    first_time = analysis.get("median_time_before_first_follow_up_sec")
    cards[4].metric(
        "Median time to first",
        "N/A" if first_time is None else "{:.1f}s".format(first_time),
    )
    accept_rate = analysis.get("recommendation_acceptance_rate")
    cards[5].metric(
        "Acceptance rate",
        "N/A" if accept_rate is None else "{:.1%}".format(accept_rate),
    )

    if selected_mode == "final" and analysis.get("registered_sessions", 0) == 0:
        st.info(
            "Main Study is empty, which is expected before the Information-Seeking design is finalized and final collection is armed."
        )

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Download behavior events",
        data=events.to_csv(index=False).encode("utf-8"),
        file_name="information_seeking_{}_events.csv".format(selected_mode),
        mime="text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "Download session summaries",
        data=summaries.to_json(orient="records", indent=2).encode("utf-8"),
        file_name="information_seeking_{}_sessions.json".format(selected_mode),
        mime="application/json",
        use_container_width=True,
    )
    d3.download_button(
        "Download analysis summary",
        data=json.dumps(analysis, indent=2).encode("utf-8"),
        file_name="information_seeking_{}_analysis.json".format(selected_mode),
        mime="application/json",
        use_container_width=True,
    )

    section(
        "Behavior analysis",
        "These are descriptive log summaries. They are not a substitute for qualitative theme coding.",
    )
    left, right = st.columns(2)
    category_counts = analysis.get("category_counts") or {}
    with left:
        if category_counts:
            frame = pd.DataFrame(
                [
                    {
                        "Category": CATEGORY_LABELS.get(key, key),
                        "Count": value,
                    }
                    for key, value in category_counts.items()
                ]
            ).sort_values("Count", ascending=True)
            fig = px.bar(
                frame,
                x="Count",
                y="Category",
                orientation="h",
                title="Follow-up categories",
            )
            apply_research_layout(
                fig,
                height=340,
                legend="none",
                title="Follow-up categories",
                bottom_margin=50,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        else:
            st.caption("No follow-up categories have been recorded.")
    with right:
        patterns = analysis.get("representative_pattern_counts") or {}
        if patterns:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Representative pattern": key, "Sessions": value}
                        for key, value in patterns.items()
                    ]
                ).sort_values("Sessions", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No representative behavior patterns are available yet.")

    evidence_counts = analysis.get("evidence_view_counts") or {}
    decisions = analysis.get("decision_counts") or {}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Evidence viewed**")
        if evidence_counts:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Evidence area": key.replace("_", " ").title(), "Views": value}
                        for key, value in evidence_counts.items()
                    ]
                ).sort_values("Views", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No evidence views recorded.")
    with c2:
        st.markdown("**Final recommendation decision**")
        if decisions:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Decision": key.title(), "Sessions": value}
                        for key, value in decisions.items()
                    ]
                ).sort_values("Sessions", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No completed decisions recorded.")

    section(
        "Qualitative coding",
        "Manual coding is authoritative. The rule-based classifier can suggest where to look, but does not assign research themes.",
    )
    if summaries.empty:
        st.info("No participant sessions are available for coding.")
    else:
        participant_sessions = summaries["session_hash"].tolist()
        selected_session = st.selectbox(
            "Session to code",
            participant_sessions,
            key="p17_code_session",
        )
        selected_row = summaries[
            summaries["session_hash"] == selected_session
        ].iloc[0].to_dict()
        with st.expander("Session behavior summary", expanded=True):
            st.json(
                {
                    "session_hash": selected_session,
                    "follow_up_count": selected_row.get("follow_up_count"),
                    "first_follow_up_category": selected_row.get(
                        "first_follow_up_category"
                    ),
                    "time_before_first_follow_up_sec": selected_row.get(
                        "time_before_first_follow_up_sec"
                    ),
                    "evidence_viewed": selected_row.get("evidence_viewed"),
                    "recommendation_decision": selected_row.get(
                        "recommendation_decision"
                    ),
                    "participant_rationale": selected_row.get(
                        "participant_rationale"
                    ),
                    "representative_pattern": selected_row.get(
                        "representative_pattern"
                    ),
                },
                expanded=False,
            )
        code_map = {
            str(code.get("id")): str(code.get("label"))
            for code in coding.get("codes") or []
        }
        themes = st.multiselect(
            "Manual qualitative themes",
            list(code_map.keys()),
            format_func=lambda key: code_map.get(key, key),
            key="p17_manual_themes",
        )
        coder_note = st.text_area(
            "Coder note",
            height=90,
            key="p17_coder_note",
        )
        if st.button(
            "Save qualitative coding",
            type="primary",
            key="p17_save_code",
        ):
            StudyStore().log(
                study_name,
                "researcher-coding",
                "qualitative_code",
                {
                    "coded_session_hash": selected_session,
                    "themes": themes,
                    "coder_note": coder_note or None,
                    "coding_scheme": "information_seeking_qualitative_coding_scheme_v1",
                    "classifier_not_authoritative": True,
                },
            )
            st.success("Manual qualitative coding saved.")

    current_codes = manual_coding(StudyStore().export(study_name))
    section(
        "Theme analysis",
        "Manual theme counts become meaningful only after sessions have been coded. Representative behavior patterns remain descriptive log summaries.",
    )
    if current_codes.empty:
        st.info("No manually coded sessions yet.")
    else:
        theme_counts: Dict[str, int] = {}
        for values in current_codes["themes"]:
            for value in values or []:
                theme_counts[value] = theme_counts.get(value, 0) + 1
        theme_frame = pd.DataFrame(
            [
                {
                    "Theme": code_map.get(theme, theme),
                    "Coded sessions": count,
                }
                for theme, count in theme_counts.items()
            ]
        ).sort_values("Coded sessions", ascending=False)
        st.dataframe(theme_frame, use_container_width=True, hide_index=True)

    with st.expander("Think-aloud prompts for facilitated sessions", expanded=False):
        for prompt in THINK_ALOUD_PROMPTS:
            st.markdown("- {}".format(prompt))

    if not FINAL_DESIGN_MANIFEST.exists():
        st.warning(
            "Design is not frozen. Before Main Study, finalize the participant target, ethics status, participant materials and qualitative coding scheme, then run `python -m scripts.freeze_phase17_design`."
        )


def information_seeking_research_page():
    hero(
        "HUMAN STUDY · INFORMATION SEEKING",
        "Information-Seeking Study",
        (
            "Observe when users ask for evidence, explanation, comparison, clarification "
            "or challenge before accepting or rejecting an AwareML recommendation."
        ),
        pills=[
            ("Behavior logging", "good"),
            ("Grounded evidence", "good"),
            ("Qualitative coding", "good"),
            ("Pilot Study" if _resolve_collection_mode() == "pilot" else "Main Study", "warn" if _resolve_collection_mode() == "pilot" else "good"),
        ],
    )

    st.markdown(
        """
        <div class="r9-callout" style="margin-bottom:1.1rem">
          <b>Research question.</b> When do users ask for more information instead of
          accepting the initial explanation? The study records whether a follow-up occurs,
          follow-up depth, first follow-up type, time before the first follow-up, evidence
          viewed, and the final recommendation decision.
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = _resolve_collection_mode()
    current_results = result_dicts()
    current_ranking = _state().get("ranking") or []
    results, ranking, context_source, saved_context = resolve_study_context(
        mode,
        current_results,
        current_ranking,
    )

    access = st.session_state.get("p17_access", "home")
    if access == "home":
        section(
            "Choose study access",
            "Participants complete the behavioral study; researchers review logs, coding and themes.",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                <div class="r9-panel" style="min-height:165px;margin-bottom:.75rem">
                  <div class="r9-card-label">PARTICIPANT STUDY</div>
                  <div style="font-size:1.2rem;font-weight:760;margin:.5rem 0">Explore or stop when you have enough information</div>
                  <div class="r9-card-note">Ask follow-ups, inspect evidence, then accept, override or reject the recommendation.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Enter participant study",
                type="primary",
                use_container_width=True,
                key="p17_enter_participant",
            ):
                _set_access("participant")
        with c2:
            st.markdown(
                """
                <div class="r9-panel" style="min-height:165px;margin-bottom:.75rem">
                  <div class="r9-card-label">RESEARCHER WORKSPACE</div>
                  <div style="font-size:1.2rem;font-weight:760;margin:.5rem 0">Review behavior and qualitative themes</div>
                  <div class="r9-card-note">Protected access to collection summaries, exports, representative patterns and manual coding.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Open researcher workspace",
                use_container_width=True,
                key="p17_enter_researcher",
            ):
                _set_access("researcher")

        protocol = _protocol()
        target = protocol.get("participant_target") or {}
        cards = st.columns(4)
        cards[0].metric("Suggested sample", "{}–{}".format(target.get("suggested_min", 10), target.get("suggested_max", 15)))
        cards[1].metric("Follow-up categories", "5 + other")
        cards[2].metric("Qualitative coding", "Required")
        cards[3].metric("Classifier role", "Assistive only")
        if context_source == "current_session":
            st.success(
                "Participant context ready from the current AwareML session. For repeated participants, the researcher can save this measured run as a standardized Pilot context."
            )
        elif context_source == "saved_study_context":
            st.success(
                "Participant context ready from the saved standardized {} study context. Participants do not need to run the dashboard first.".format(
                    "Pilot" if mode == "pilot" else "Main"
                )
            )
        else:
            st.info(
                "Participant context is not ready. Researcher workflow: Run Studio → run the agreed five-framework benchmark → open Decision Lab once → return to Information-Seeking Study. For repeated participants, save that run in Researcher Workspace."
            )
        return

    top, _run = _top_context(results, ranking)
    if access == "participant":
        if st.button("Back to study home", key="p17_back_participant"):
            _set_access("home")
        if not results:
            empty_state(
                "Study context required",
                "Researcher setup: Run Studio → run the agreed five-framework benchmark → open Decision Lab once → return here. The researcher can then save the measured run as a standardized Pilot context so later participants do not need to rerun AwareML.",
            )
            return
        if top is None:
            empty_state(
                "Observed ranking required",
                "Open Decision Lab once after the benchmark so the Information-Seeking Study has an initial observed recommendation, then return here.",
            )
            return
        _participant_page(results, ranking, context_source=context_source, saved_context=saved_context)
        return

    if access == "researcher":
        if st.button("Back to study home", key="p17_back_researcher"):
            st.session_state["p17_access"] = "home"
            st.rerun()
        _researcher_page(current_results, current_ranking)
        return
