from __future__ import annotations

import html
import os
import secrets
import time
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from awareml.studies.trust import (
    ALLOWED_DECISION_ACTIONS,
    ALLOWED_EXPERTISE_GROUPS,
    FINAL_DESIGN_MANIFEST,
    EligibilityError,
    Phase16Error,
    ProtocolGateError,
    TrustCalibrationStudy,
    trust_measure_items,
)
from awareml.studies.trust_analysis import analyze_phase16_store

from .components import hero, section


PARTICIPANT_STATE_PREFIX = "p16_participant"
ACCESS_ROLE_KEY = "trust_calibration_access_role"
DASHBOARD_STUDY_ACTIVE_KEY = "trust_calibration_dashboard_study_active"



# -----------------------------------------------------------------------------
# Presentation helpers only. Study logic, randomization, storage and analysis are
# deliberately untouched by this UI polish layer.
# -----------------------------------------------------------------------------

def _inject_trust_ui_css() -> None:
    st.markdown(
        """
        <style>
        .tc-shell {margin-top:.2rem;}
        .tc-stat {
            min-height:116px;
            padding:18px 20px;
            border:1px solid rgba(100,116,139,.18);
            border-radius:18px;
            background:linear-gradient(145deg, rgba(255,255,255,.96), rgba(248,250,252,.88));
            box-shadow:0 10px 28px rgba(15,23,42,.055);
        }
        .tc-stat-label {
            font-size:.76rem;
            font-weight:750;
            letter-spacing:.055em;
            text-transform:uppercase;
            color:#64748b;
        }
        .tc-stat-value {
            margin-top:7px;
            font-size:1.72rem;
            line-height:1.08;
            font-weight:780;
            letter-spacing:-.025em;
            color:#0f172a;
        }
        .tc-stat-note {
            margin-top:7px;
            font-size:.79rem;
            line-height:1.4;
            color:#64748b;
        }
        .tc-status-ok {color:#047857;}
        .tc-status-warn {color:#b45309;}
        .tc-status-neutral {color:#334155;}
        .tc-design {
            margin:12px 0 4px;
            padding:15px 18px;
            border:1px solid rgba(37,99,235,.14);
            border-left:4px solid #2563eb;
            border-radius:14px;
            background:linear-gradient(90deg, rgba(239,246,255,.94), rgba(248,250,252,.84));
            color:#334155;
            line-height:1.55;
        }
        .tc-design strong {color:#0f172a;}
        .tc-chip {
            display:inline-flex;
            align-items:center;
            gap:6px;
            margin:5px 7px 0 0;
            padding:5px 9px;
            border:1px solid rgba(100,116,139,.18);
            border-radius:999px;
            background:rgba(255,255,255,.8);
            color:#475569;
            font-size:.75rem;
            font-weight:650;
        }
        .tc-dot {width:7px;height:7px;border-radius:50%;background:#10b981;display:inline-block;}
        .tc-item-card {
            padding:22px 24px;
            border:1px solid rgba(100,116,139,.18);
            border-radius:18px;
            background:linear-gradient(145deg, rgba(255,255,255,.98), rgba(248,250,252,.92));
            box-shadow:0 12px 30px rgba(15,23,42,.05);
        }
        .tc-item-meta {
            display:flex;
            flex-wrap:wrap;
            gap:8px;
            align-items:center;
            margin-bottom:16px;
        }
        .tc-badge {
            display:inline-flex;
            align-items:center;
            padding:5px 9px;
            border-radius:999px;
            background:#eff6ff;
            border:1px solid #dbeafe;
            color:#1d4ed8;
            font-size:.73rem;
            font-weight:700;
        }
        .tc-badge-muted {
            background:#f8fafc;
            border-color:#e2e8f0;
            color:#64748b;
        }
        .tc-block-label {
            font-size:.75rem;
            font-weight:760;
            letter-spacing:.055em;
            text-transform:uppercase;
            color:#64748b;
            margin-bottom:7px;
        }
        .tc-prompt {
            font-size:1rem;
            line-height:1.58;
            color:#334155;
            margin-bottom:18px;
        }
        .tc-explanation {
            padding:16px 18px;
            border-radius:14px;
            background:#f8fafc;
            border:1px solid #e2e8f0;
            color:#0f172a;
            font-size:1.02rem;
            line-height:1.68;
        }
        .tc-footnote {
            margin-top:12px;
            font-size:.79rem;
            color:#64748b;
        }
        .tc-lock {
            padding:16px 18px;
            border-radius:14px;
            border:1px solid #e2e8f0;
            background:#f8fafc;
            color:#475569;
            line-height:1.5;
        }
        .tc-warning {
            padding:14px 16px;
            border-radius:14px;
            border:1px solid #fde68a;
            background:#fffbeb;
            color:#92400e;
            line-height:1.5;
        }
        .tc-success {
            padding:14px 16px;
            border-radius:14px;
            border:1px solid #a7f3d0;
            background:#ecfdf5;
            color:#065f46;
            line-height:1.5;
        }
        .tc-mini-grid {
            display:grid;
            grid-template-columns:repeat(2,minmax(0,1fr));
            gap:10px;
            margin-top:10px;
        }
        .tc-mini {
            padding:12px 14px;
            border:1px solid #e2e8f0;
            border-radius:12px;
            background:#fff;
        }
        .tc-mini-label {font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:#64748b;font-weight:750;}
        .tc-mini-value {margin-top:5px;font-size:1rem;color:#0f172a;font-weight:700;word-break:break-word;}
        @media (max-width: 900px) {
            .tc-stat {min-height:auto;}
            .tc-mini-grid {grid-template-columns:1fr;}
            .tc-item-card {padding:18px;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _configured_researcher_key() -> str:
    """Return the administrator-configured researcher key without exposing it in the UI."""
    try:
        secret_value = st.secrets.get("AWAREML_STUDY_RESEARCHER_KEY", "")
    except Exception:
        secret_value = ""
    return str(secret_value or os.environ.get("AWAREML_STUDY_RESEARCHER_KEY", "")).strip()


def _resolve_participant_collection_mode() -> str:
    """Resolve Pilot/Main Study at server level so participants never choose a data mode."""
    configured = str(os.environ.get("AWAREML_PHASE16_COLLECTION_MODE", "")).strip().lower()
    if configured:
        if configured not in {"pilot", "final"}:
            raise ProtocolGateError("Configured study collection mode must be pilot or final.")
        return configured

    final_armed = str(os.environ.get("AWAREML_PHASE16_FINAL_ARMED", "")).strip().upper() == "YES"
    if FINAL_DESIGN_MANIFEST.exists() and final_armed:
        return "final"
    return "pilot"


def _mode_display_name(mode: str) -> str:
    return "Main Study" if str(mode).lower() == "final" else "Pilot Study"


def _hide_dashboard_navigation() -> None:
    """Hide dashboard navigation while an embedded participant session is active."""
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {display:none !important;}
        [data-testid="collapsedControl"] {display:none !important;}
        [data-testid="stSidebarCollapsedControl"] {display:none !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _leave_dashboard_study_view() -> None:
    st.session_state.pop(DASHBOARD_STUDY_ACTIVE_KEY, None)
    st.session_state.pop(ACCESS_ROLE_KEY, None)


def _access_card(title: str, body: str, badge: str) -> None:
    st.markdown(
        '<div class="tc-item-card" style="min-height:180px">'
        '<div class="tc-item-meta"><span class="tc-badge">{}</span></div>'
        '<div style="font-size:1.2rem;font-weight:780;color:#0f172a;margin-bottom:8px">{}</div>'
        '<div style="color:#64748b;line-height:1.6">{}</div>'
        '</div>'.format(html.escape(badge), html.escape(title), html.escape(body)),
        unsafe_allow_html=True,
    )


def _legacy_researcher_mode() -> bool:
    return str(os.environ.get("AWAREML_STUDY_RESEARCHER_MODE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _researcher_authorized() -> bool:
    if _legacy_researcher_mode():
        return True
    return bool(st.session_state.get("p16_researcher_unlocked"))


def _display_status(value: Any) -> str:
    raw = str(value or "").strip().lower()
    mapping = {
        "pending_morten": "Awaiting review",
        "pending": "Awaiting review",
        "draft": "Draft",
        "frozen": "Ready",
        "ready": "Ready",
        "ok": "Ready",
        "final": "Final",
        "pilot": "Pilot",
        "unknown": "Not set",
        "": "Not set",
    }
    return mapping.get(raw, str(value).replace("_", " ").strip().title())


def _status_tone(value: Any) -> str:
    text = _display_status(value).lower()
    if text in {"ready", "final"}:
        return "ok"
    if text in {"awaiting review", "draft", "not set"}:
        return "warn"
    return "neutral"


def _metric_card(label: str, value: Any, note: str = "", tone: str = "neutral") -> None:
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_note = html.escape(str(note))
    st.markdown(
        '<div class="tc-stat"><div class="tc-stat-label">{}</div>'
        '<div class="tc-stat-value tc-status-{}">{}</div>'
        '<div class="tc-stat-note">{}</div></div>'.format(
            safe_label,
            tone,
            safe_value,
            safe_note,
        ),
        unsafe_allow_html=True,
    )


def _render_researcher_unlock() -> bool:
    if _researcher_authorized():
        return True
    configured_key = _configured_researcher_key()
    st.markdown(
        '<div class="tc-lock"><b>Researcher access is locked.</b><br>'
        'Ground-truth labels, collection exports and analysis are protected from participant sessions.</div>',
        unsafe_allow_html=True,
    )
    if not configured_key:
        st.info(
            "Researcher access has not been configured by the app administrator. Add a private researcher key to the local .env file or deployment secrets, then restart the app."
        )
        return False
    with st.form("p16_researcher_unlock_form", clear_on_submit=True):
        key = st.text_input("Researcher key", type="password")
        unlock = st.form_submit_button("Unlock researcher console", type="primary", use_container_width=True)
    if unlock:
        if secrets.compare_digest(str(key), configured_key):
            st.session_state.p16_researcher_unlocked = True
            st.rerun()
        else:
            st.error("Researcher key did not match.")
    return False


def _protocol_status_cards(study: TrustCalibrationStudy) -> None:
    protocol = study.protocol
    bank = study.bank.validate_design_pool(
        int((protocol.get("design") or {}).get("shared_pool_min_per_condition", 20))
    )
    power = protocol.get("power_calculation") or {}
    trust = protocol.get("trust_measure") or {}

    cols = st.columns(5)
    with cols[0]:
        _metric_card("Correct explanations", bank["correct"], "Verified shared pool", "ok")
    with cols[1]:
        _metric_card("Incorrect explanations", bank["incorrect"], "Verified shared pool", "ok")
    with cols[2]:
        _metric_card("Matched pairs", bank["pairs"], "One variant shown per participant", "ok")
    with cols[3]:
        trust_status = _display_status(trust.get("status", "unknown"))
        _metric_card("Trust measure", trust_status, "Final instrument is externally reviewed", _status_tone(trust.get("status")))
    with cols[4]:
        required = power.get("required_completed_participants")
        value = "Awaiting review" if required is None else str(required)
        _metric_card("Power target", value, "Completed-participant requirement", "warn" if required is None else "ok")


def _participant_registration(study: TrustCalibrationStudy, collection_mode: str) -> Optional[str]:
    participant_hash_key = "{}_hash".format(PARTICIPANT_STATE_PREFIX)
    mode_key = "{}_mode".format(PARTICIPANT_STATE_PREFIX)
    if st.session_state.get(participant_hash_key) and st.session_state.get(mode_key) == collection_mode:
        return str(st.session_state[participant_hash_key])

    protocol = study.protocol
    materials = protocol.get("participant_materials") or {}
    section("Before you begin", "Enter the anonymous code supplied by the study team and confirm eligibility.")
    st.write(materials.get("instructions") or "You will evaluate a sequence of explanations one at a time.")
    with st.expander("Study information and consent", expanded=False):
        st.write(
            materials.get("consent_text")
            or "Pilot consent text has not yet been finalized. Do not use this draft for final recruitment."
        )

    with st.form("p16_registration_form"):
        c1, c2 = st.columns([1.25, 1])
        with c1:
            participant_code = st.text_input(
                "Participant / session code",
                help="Use the anonymous code provided by the study team. The raw code is not stored in the study database.",
            )
            expertise_group = st.selectbox("Experience group", list(ALLOWED_EXPERTISE_GROUPS))
        with c2:
            expertise_self_rating = st.slider(
                "Self-rated ML / AutoML expertise",
                1,
                5,
                3,
                help="1 = very limited; 5 = highly experienced",
            )
            track2_separate = st.checkbox(
                "I confirm that I did not help author the scenario bank used to create these study explanations."
            )
        consented = st.checkbox("I have read the study information and consent to participate.")
        start = st.form_submit_button("Start / resume study", type="primary", use_container_width=True)

    if start:
        try:
            registration = study.register(
                participant_code=participant_code,
                expertise_group=expertise_group,
                expertise_self_rating=expertise_self_rating,
                consented=consented,
                track1_author_attested_separate=track2_separate,
                collection_mode=collection_mode,
            )
        except (EligibilityError, ProtocolGateError, Phase16Error, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state[participant_hash_key] = registration.participant_hash
            st.session_state[mode_key] = collection_mode
            st.session_state.pop("{}_trial_started".format(PARTICIPANT_STATE_PREFIX), None)
            st.rerun()
    return None


def _reset_participant_session() -> None:
    keys = [key for key in list(st.session_state) if str(key).startswith(PARTICIPANT_STATE_PREFIX)]
    for key in keys:
        st.session_state.pop(key, None)


def _render_stimulus_card(item: Dict[str, Any], show_blinding_note: bool = True) -> None:
    source = html.escape(str(item.get("source") or "Explanation"))
    item_id = html.escape(str(item.get("item_id") or ""))
    prompt = html.escape(str(item.get("prompt") or ""))
    explanation = html.escape(str(item.get("explanation") or ""))
    note = (
        '<div class="tc-footnote">Correctness condition is intentionally hidden.</div>'
        if show_blinding_note
        else ""
    )
    st.markdown(
        '<div class="tc-item-card">'
        '<div class="tc-item-meta"><span class="tc-badge">{}</span>'
        '<span class="tc-badge tc-badge-muted">Study item {}</span></div>'
        '<div class="tc-block-label">Prompt</div><div class="tc-prompt">{}</div>'
        '<div class="tc-block-label">Explanation</div><div class="tc-explanation">{}</div>{}'
        '</div>'.format(source, item_id, prompt, explanation, note),
        unsafe_allow_html=True,
    )


def render_phase16_participant_study(
    collection_mode: Optional[str] = None,
    isolated: bool = False,
    dashboard_embedded: bool = False,
) -> None:
    _inject_trust_ui_css()
    try:
        study = TrustCalibrationStudy()
    except Exception as exc:
        st.error("Trust Calibration study could not initialize: {}".format(exc))
        st.stop()

    mode = str(collection_mode or _resolve_participant_collection_mode()).lower()
    if mode == "final":
        try:
            study.assert_final_collection_ready()
        except ProtocolGateError as exc:
            st.error("Main Study is locked: {}".format(exc))
            st.stop()

    if dashboard_embedded:
        _hide_dashboard_navigation()
        top_left, top_right = st.columns([5, 1])
        with top_left:
            st.markdown(
                '<div class="tc-lock"><b>Participant Study · {}</b><br>'
                'Dashboard navigation is hidden while you complete the study. Your progress is preserved if you leave and resume later.</div>'.format(
                    html.escape(_mode_display_name(mode))
                ),
                unsafe_allow_html=True,
            )
        with top_right:
            if st.button("Exit study view", key="trust_exit_dashboard_study", use_container_width=True):
                _leave_dashboard_study_view()
                st.rerun()

    if isolated:
        hero(
            "HUMAN STUDY · CALIBRATED RELIANCE",
            "Trust Calibration",
            "Evaluate AI explanations one item at a time. Ground-truth correctness is hidden while trust, perceived correctness and reliance decisions are recorded.",
            pills=[
                ("Blinded", "good"),
                ("Within-subject", "good"),
                ("Balanced conditions", "good"),
                (_mode_display_name(mode), "warn" if mode == "pilot" else "good"),
            ],
        )

    if mode == "pilot":
        st.markdown(
            '<div class="tc-warning"><b>Pilot Study.</b> The study instrument and recruitment target may still be provisional. '
            'Pilot records are stored separately and are never mixed into final analysis.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="tc-success"><b>Main Study.</b> The frozen study design and participant blinding are active.</div>',
            unsafe_allow_html=True,
        )

    participant_hash = _participant_registration(study, mode)
    if participant_hash is None:
        return

    progress = study.participant_progress(participant_hash, mode)
    trial = study.current_trial(participant_hash, mode)
    if trial is None:
        st.markdown(
            '<div class="tc-success"><b>Study complete.</b><br>Thank you. Your responses have been stored under a pseudonymous participant identifier.</div>',
            unsafe_allow_html=True,
        )
        st.progress(1.0)
        if dashboard_embedded:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Return to AwareML dashboard", key="trust_return_dashboard", type="primary", use_container_width=True):
                    _leave_dashboard_study_view()
                    st.rerun()
            with c2:
                if mode == "pilot" and st.button("Start another pilot session", key="p16_clear_completed", use_container_width=True):
                    _reset_participant_session()
                    st.rerun()
        elif not isolated and st.button("Clear local participant session", key="p16_clear_completed"):
            _reset_participant_session()
            st.rerun()
        return

    completed = int(progress["completed"])
    total = int(progress["total"])
    section("Study progress", "Complete each item independently before moving to the next explanation.")
    st.progress(float(completed) / float(max(total, 1)))
    st.caption("{} of {} completed · current item {}".format(completed, total, trial["order"]))

    _render_stimulus_card(trial, show_blinding_note=True)

    start_key = "{}_trial_started".format(PARTICIPANT_STATE_PREFIX)
    item_key = "{}_trial_item".format(PARTICIPANT_STATE_PREFIX)
    if st.session_state.get(item_key) != trial["item_id"]:
        st.session_state[item_key] = trial["item_id"]
        st.session_state[start_key] = time.time()

    protocol = study.protocol
    trust_measure = protocol.get("trust_measure") or {}
    scale_min = int(trust_measure.get("scale_min", 1))
    scale_max = int(trust_measure.get("scale_max", 7))
    scale_mid = int(round((scale_min + scale_max) / 2.0))
    trust_items = trust_measure_items(protocol, mode)

    section("Your response", "Rate the explanation itself, then indicate how you would act on it.")
    with st.form("p16_trial_form_{}".format(trial["item_id"])):
        ratings: Dict[str, int] = {}
        for item in trust_items:
            item_id = str(item["id"])
            ratings[item_id] = st.slider(
                str(item["text"]),
                scale_min,
                scale_max,
                scale_mid,
                key="p16_trust_{}_{}".format(trial["item_id"], item_id),
                help="{} = {}, {} = {}".format(
                    scale_min,
                    trust_measure.get("scale_min_label", "low"),
                    scale_max,
                    trust_measure.get("scale_max_label", "high"),
                ),
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            perceived_correctness = st.slider(
                "How likely is the explanation to be factually correct?",
                1,
                7,
                4,
                key="p16_correct_{}".format(trial["item_id"]),
            )
        with c2:
            fluency = st.slider(
                "How fluent and well-written does the explanation sound?",
                1,
                7,
                4,
                key="p16_fluency_{}".format(trial["item_id"]),
            )
        with c3:
            perceived_confidence = st.slider(
                "How confident or authoritative does the explanation sound?",
                1,
                7,
                4,
                key="p16_confidence_{}".format(trial["item_id"]),
            )

        decision_action = st.radio(
            "What would you do with this recommendation / explanation?",
            list(ALLOWED_DECISION_ACTIONS),
            horizontal=True,
            key="p16_action_{}".format(trial["item_id"]),
            help="Accept = rely on it as shown; Override = choose a different action; Reject = do not use it.",
        )
        submit = st.form_submit_button("Submit response and continue", type="primary", use_container_width=True)

    if submit:
        elapsed = max(0.001, time.time() - float(st.session_state.get(start_key, time.time())))
        try:
            study.submit_trial(
                participant_hash=participant_hash,
                order_index=int(trial["order"]),
                trust_items=ratings,
                perceived_correctness=perceived_correctness,
                fluency_rating=fluency,
                perceived_confidence=perceived_confidence,
                decision_action=decision_action,
                response_time_sec=elapsed,
                collection_mode=mode,
            )
        except (Phase16Error, ValueError, ProtocolGateError) as exc:
            st.error(str(exc))
        else:
            st.session_state.pop(start_key, None)
            st.session_state.pop(item_key, None)
            st.rerun()


def _download_frame(label: str, frame: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
        use_container_width=True,
    )


def _protocol_ui_snapshot(protocol: Dict[str, Any]) -> Dict[str, Any]:
    design = protocol.get("design") or {}
    trust = protocol.get("trust_measure") or {}
    power = protocol.get("power_calculation") or {}
    ethics = protocol.get("ethics") or {}
    materials = protocol.get("participant_materials") or {}
    return {
        "Study status": _display_status(protocol.get("status")),
        "Central research question": protocol.get("central_rq"),
        "Design": {
            "within_subject": design.get("within_subject"),
            "randomized": design.get("randomized"),
            "blinded": design.get("blinded"),
            "items_per_participant": design.get("items_per_participant"),
            "shared_pool_min_per_condition": design.get("shared_pool_min_per_condition"),
        },
        "Trust measure": {
            "status": _display_status(trust.get("status")),
            "scale_min": trust.get("scale_min"),
            "scale_max": trust.get("scale_max"),
        },
        "Power calculation": {
            "status": _display_status(power.get("status")),
            "required_completed_participants": power.get("required_completed_participants"),
        },
        "Ethics": {
            "status": _display_status(ethics.get("status")),
        },
        "Participant materials": {
            "status": _display_status(materials.get("status")),
        },
    }


def _analysis_ui_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in result.items() if k not in {"analysis_version"}}


def _fmt_rate(value: Any) -> str:
    try:
        return "{:.1f}%".format(float(value) * 100.0)
    except (TypeError, ValueError):
        return "N/A"


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        return ("{:,.%df}" % digits).format(float(value))
    except (TypeError, ValueError):
        return "N/A"


def _render_analysis_summary(result: Dict[str, Any]) -> None:
    status = str(result.get("status") or "")
    if status != "ok":
        st.info(
            "Analysis is waiting for complete participant data. Registered: {} · Responses: {}".format(
                result.get("participants_registered", 0),
                result.get("responses", 0),
            )
        )
        with st.expander("Analysis details", expanded=False):
            st.json(_analysis_ui_result(result), expanded=False)
        return

    primary = result.get("primary_calibration") or {}
    paired = primary.get("paired_gap") or {}
    overtrust = result.get("overtrust") or {}
    reliance = result.get("reliance") or {}
    condition = result.get("condition_summary") or {}
    correct = condition.get("correct") or {}
    incorrect = condition.get("incorrect") or {}

    cols = st.columns(4)
    with cols[0]:
        _metric_card("Trust gap", _fmt_num(paired.get("mean_gap")), "Correct minus incorrect", "ok")
    with cols[1]:
        _metric_card("Over-trust", _fmt_rate(overtrust.get("overtrust_rate")), "Accepted incorrect explanations", "warn")
    with cols[2]:
        _metric_card("Appropriate reliance", _fmt_rate(reliance.get("appropriate_reliance_rate")), "Correct accept + incorrect non-accept", "ok")
    with cols[3]:
        _metric_card("Valid participants", result.get("participants_completed_valid", 0), "Complete within-subject records", "neutral")

    summary_frame = pd.DataFrame(
        [
            {
                "Condition": "Correct",
                "Mean trust": correct.get("mean_trust"),
                "Acceptance rate": correct.get("acceptance_rate"),
                "Mean fluency": correct.get("mean_fluency"),
                "Mean response time (s)": correct.get("mean_response_time_sec"),
            },
            {
                "Condition": "Incorrect",
                "Mean trust": incorrect.get("mean_trust"),
                "Acceptance rate": incorrect.get("acceptance_rate"),
                "Mean fluency": incorrect.get("mean_fluency"),
                "Mean response time (s)": incorrect.get("mean_response_time_sec"),
            },
        ]
    )
    st.dataframe(summary_frame, use_container_width=True, hide_index=True)
    with st.expander("Complete analysis details", expanded=False):
        st.json(_analysis_ui_result(result), expanded=False)


def _render_researcher_console(study: TrustCalibrationStudy) -> None:
    if not _render_researcher_unlock():
        return

    st.markdown(
        '<div class="tc-warning"><b>Researcher console unlocked.</b> Ground-truth correctness labels below are restricted to the research team.</div>',
        unsafe_allow_html=True,
    )
    protocol = study.protocol
    bank_summary = study.bank.validate_design_pool(
        int((protocol.get("design") or {}).get("shared_pool_min_per_condition", 20))
    )

    section("Study readiness", "Final collection remains locked until the trust measure, power calculation, ethics materials and study design are finalized.")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        _metric_card("Protocol", _display_status(protocol.get("status")), "Study specification", _status_tone(protocol.get("status")))
    with p2:
        trust_status = (protocol.get("trust_measure") or {}).get("status")
        _metric_card("Trust measure", _display_status(trust_status), "Validated instrument", _status_tone(trust_status))
    with p3:
        power_status = (protocol.get("power_calculation") or {}).get("status")
        _metric_card("Power calculation", _display_status(power_status), "Recruitment target", _status_tone(power_status))
    with p4:
        _metric_card("Design lock", "Ready" if FINAL_DESIGN_MANIFEST.exists() else "Not frozen", "Final collection safeguard", "ok" if FINAL_DESIGN_MANIFEST.exists() else "warn")

    left, right = st.columns([1.15, 0.85])
    with left:
        with st.expander("Study protocol snapshot", expanded=False):
            st.json(_protocol_ui_snapshot(protocol), expanded=False)
    with right:
        with st.expander("Verified stimulus bank", expanded=False):
            st.markdown(
                '<div class="tc-mini-grid">'
                '<div class="tc-mini"><div class="tc-mini-label">Correct</div><div class="tc-mini-value">{}</div></div>'
                '<div class="tc-mini"><div class="tc-mini-label">Incorrect</div><div class="tc-mini-value">{}</div></div>'
                '<div class="tc-mini"><div class="tc-mini-label">Matched pairs</div><div class="tc-mini-value">{}</div></div>'
                '<div class="tc-mini"><div class="tc-mini-label">Integrity</div><div class="tc-mini-value">Verified</div></div>'
                '</div>'.format(bank_summary.get("correct"), bank_summary.get("incorrect"), bank_summary.get("pairs")),
                unsafe_allow_html=True,
            )
            st.caption("Manifest hash: {}".format(bank_summary.get("manifest_sha256") or "N/A"))

    section("Blinded preview", "Review exactly what participants see without exposing a correctness label or writing a response.")
    preview = study.build_preview_assignment("advanced-labs-preview")
    preview_index = st.selectbox(
        "Preview item",
        options=list(range(1, len(preview) + 1)),
        index=0,
        format_func=lambda x: "Item {} of {}".format(x, len(preview)),
        key="p16_preview_order",
    )
    item = preview[int(preview_index) - 1]
    _render_stimulus_card(item, show_blinding_note=True)

    with st.expander("Researcher-only ground truth", expanded=False):
        stimulus_ids = sorted(study.bank.stimuli)
        selected = st.selectbox("Stimulus record", stimulus_ids, key="p16_researcher_stimulus")
        stimulus = study.bank.get(selected)
        gt1, gt2, gt3 = st.columns(3)
        gt1.metric("Condition", str(stimulus.correctness_condition).title())
        gt2.metric("Source", str(stimulus.source_stage))
        gt3.metric("Error type", "None" if stimulus.error_type is None else str(stimulus.error_type).replace("_", " ").title())
        st.caption("Internal stimulus ID: {} · Pair: {}".format(stimulus.stimulus_id, stimulus.pair_id))
        st.code(str(stimulus.explanation_sha256 or "No explanation hash"), language=None)

    section("Collection status", "Pilot Study and Main Study records remain physically separated in the append-only study store.")
    dataset_label = st.radio(
        "Researcher dataset",
        ["Pilot Study", "Main Study"],
        horizontal=True,
        key="p16_research_mode",
    )
    mode = "pilot" if dataset_label == "Pilot Study" else "final"
    summary = study.store.summary(mode)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        _metric_card("Registered", summary["registered_participants"], "Participant records", "neutral")
    with m2:
        _metric_card("Completed", summary["completed_participants"], "Complete participants", "ok" if summary["completed_participants"] else "neutral")
    with m3:
        _metric_card("Responses", summary["responses"], "Trial responses", "neutral")
    with m4:
        expertise_count = len(summary["completed_by_expertise"] or {})
        _metric_card("Expertise groups", expertise_count, "Represented among completers", "neutral")

    if summary["completed_by_expertise"]:
        expertise_frame = pd.DataFrame(
            [{"Expertise group": k, "Completed": v} for k, v in summary["completed_by_expertise"].items()]
        )
        st.dataframe(expertise_frame, use_container_width=True, hide_index=True)

    participants = pd.DataFrame(study.store.participant_rows(mode))
    assignments = pd.DataFrame(study.store.all_assignment_rows(mode))
    responses = pd.DataFrame(study.store.response_rows(mode))
    d1, d2, d3 = st.columns(3)
    with d1:
        _download_frame("Download participant summary", participants, "trust_{}_participants.csv".format(mode), "p16_dl_participants")
    with d2:
        _download_frame("Download assignments", assignments, "trust_{}_assignments.csv".format(mode), "p16_dl_assignments")
    with d3:
        _download_frame("Download responses", responses, "trust_{}_responses.csv".format(mode), "p16_dl_responses")

    section("Calibration analysis", "Run a descriptive analysis without freezing or altering any collected response.")
    if st.button("Run descriptive calibration analysis", key="p16_run_analysis", use_container_width=True, type="primary"):
        result = analyze_phase16_store(study.store, mode)
        st.session_state.p16_analysis_result = result
    if st.session_state.get("p16_analysis_result"):
        _render_analysis_summary(st.session_state.p16_analysis_result)

    with st.expander("Study maintenance commands", expanded=False):
        st.caption("These commands are intentionally kept outside the interactive UI so a click cannot create journal-facing frozen evidence.")
        st.code(
            "python scripts/freeze_phase16_design.py\n"
            "python scripts/analyze_phase16_trust.py --mode final\n"
            "python scripts/freeze_phase16_results.py",
            language="powershell",
        )
        st.caption(
            "Do not freeze the design until the validated trust measure, power calculation, ethics review and participant materials are finalized."
        )


def _render_access_choice(study: TrustCalibrationStudy) -> None:
    mode = _resolve_participant_collection_mode()
    section(
        "Choose your study access",
        "Use Participant Study to complete the blinded evaluation. Researcher Workspace is protected and contains ground truth, exports and analysis.",
    )
    left, right = st.columns(2)
    with left:
        _access_card(
            "Participant Study",
            "Complete or resume the blinded Trust Calibration study. During an active study session, dashboard navigation is hidden to reduce accidental exposure to research-only information.",
            _mode_display_name(mode),
        )
        if st.button("Enter participant study", key="trust_enter_participant", type="primary", use_container_width=True):
            st.session_state[ACCESS_ROLE_KEY] = "participant"
            st.session_state[DASHBOARD_STUDY_ACTIVE_KEY] = True
            st.rerun()
    with right:
        _access_card(
            "Researcher Workspace",
            "Inspect study readiness, verified stimuli, restricted ground truth, collection status, exports and calibration analysis. A researcher key is required.",
            "Protected access",
        )
        if st.button("Open researcher workspace", key="trust_enter_researcher", use_container_width=True):
            st.session_state[ACCESS_ROLE_KEY] = "researcher"
            st.rerun()

    st.markdown(
        '<div class="tc-design"><strong>How access works.</strong> Most users only run <code>streamlit run app.py</code>. '
        'Participants choose Participant Study here; researchers choose Researcher Workspace and unlock it with the configured key. '
        'A separate participant-only app remains available for external recruitment when the full dashboard should not be exposed.</div>',
        unsafe_allow_html=True,
    )


def phase16_trust_calibration_page() -> None:
    _inject_trust_ui_css()
    role = str(st.session_state.get(ACCESS_ROLE_KEY) or "").strip().lower()

    if role == "participant":
        mode = _resolve_participant_collection_mode()
        render_phase16_participant_study(
            collection_mode=mode,
            isolated=False,
            dashboard_embedded=True,
        )
        return

    hero(
        "HUMAN STUDY · CALIBRATED RELIANCE",
        "Trust Calibration",
        "Evaluate whether trust follows explanation correctness rather than fluency or confident-sounding language.",
        pills=[
            ("Correctness", "good"),
            ("Blinded", "good"),
            ("Within-subject", "good"),
            ("Calibrated reliance", "good"),
        ],
    )
    try:
        study = TrustCalibrationStudy()
    except Exception as exc:
        st.error("Trust Calibration study could not initialize: {}".format(exc))
        return

    if role == "researcher":
        top_left, top_right = st.columns([5, 1])
        with top_left:
            st.markdown(
                '<div class="tc-lock"><b>Researcher Workspace</b><br>'
                'Study administration and analysis. This area is not participant-facing.</div>',
                unsafe_allow_html=True,
            )
        with top_right:
            if st.button("Back", key="trust_back_from_researcher", use_container_width=True):
                st.session_state.pop(ACCESS_ROLE_KEY, None)
                st.rerun()
        _render_researcher_console(study)
        return

    _protocol_status_cards(study)
    st.markdown(
        """
        <div class="tc-design">
          <strong>Study design.</strong> Within-subject · randomized · blinded · balanced correct/incorrect explanation stimuli ·
          one variant per matched pair · balanced explanation sources · response-time logging · expertise-stratified analysis.
          <div>
            <span class="tc-chip"><span class="tc-dot"></span>Correctness hidden from participants</span>
            <span class="tc-chip"><span class="tc-dot"></span>Matched stimulus pairs</span>
            <span class="tc-chip"><span class="tc-dot"></span>Append-only response store</span>
            <span class="tc-chip"><span class="tc-dot"></span>Protected researcher ground truth</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_access_choice(study)
