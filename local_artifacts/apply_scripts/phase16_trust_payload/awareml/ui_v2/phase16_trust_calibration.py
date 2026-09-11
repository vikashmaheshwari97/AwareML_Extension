from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
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
from .page_utils import phase_pills


PARTICIPANT_STATE_PREFIX = "p16_participant"


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


def _render_researcher_unlock() -> bool:
    if _researcher_authorized():
        return True
    configured_key = str(os.environ.get("AWAREML_STUDY_RESEARCHER_KEY", ""))
    st.markdown("#### Researcher access")
    if not configured_key:
        st.info(
            "Researcher labels are protected. Set AWAREML_STUDY_RESEARCHER_KEY before launching Streamlit, "
            "or use the legacy AWAREML_STUDY_RESEARCHER_MODE=1 development flag."
        )
        return False
    with st.form("p16_researcher_unlock_form", clear_on_submit=True):
        key = st.text_input("Researcher key", type="password")
        unlock = st.form_submit_button("Unlock researcher console", type="primary")
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
    cols[0].metric("Correct pool", bank["correct"])
    cols[1].metric("Incorrect pool", bank["incorrect"])
    cols[2].metric("Pairs", bank["pairs"])
    cols[3].metric("Trust measure", str(trust.get("status", "unknown")).upper())
    required = power.get("required_completed_participants")
    cols[4].metric("Power target", "Pending" if required is None else str(required))


def _participant_registration(study: TrustCalibrationStudy, collection_mode: str) -> Optional[str]:
    participant_hash_key = "{}_hash".format(PARTICIPANT_STATE_PREFIX)
    mode_key = "{}_mode".format(PARTICIPANT_STATE_PREFIX)
    if st.session_state.get(participant_hash_key) and st.session_state.get(mode_key) == collection_mode:
        return str(st.session_state[participant_hash_key])

    protocol = study.protocol
    materials = protocol.get("participant_materials") or {}
    st.markdown("### Before you begin")
    st.write(materials.get("instructions") or "You will evaluate a sequence of explanations one at a time.")
    with st.expander("Study information and consent", expanded=False):
        st.write(
            materials.get("consent_text")
            or "Pilot consent text has not yet been finalized. Do not use this draft for final recruitment."
        )

    with st.form("p16_registration_form"):
        participant_code = st.text_input(
            "Participant/session code",
            help="Use the anonymous code provided by the study team. The raw code is not stored in the study database.",
        )
        expertise_group = st.selectbox("Expertise group", list(ALLOWED_EXPERTISE_GROUPS))
        expertise_self_rating = st.slider(
            "Self-rated ML/AutoML expertise",
            1,
            5,
            3,
            help="1 = very limited; 5 = highly experienced",
        )
        track2_separate = st.checkbox(
            "I confirm that I did NOT author the Track-1 scenarios used to construct this study's explanation stimuli."
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


def render_phase16_participant_study(
    collection_mode: Optional[str] = None,
    isolated: bool = False,
) -> None:
    try:
        study = TrustCalibrationStudy()
    except Exception as exc:
        st.error("Phase-16 study could not initialize: {}".format(exc))
        st.stop()

    mode = str(collection_mode or study.collection_mode()).lower()
    if mode == "final":
        try:
            study.assert_final_collection_ready()
        except ProtocolGateError as exc:
            st.error("FINAL collection is locked: {}".format(exc))
            st.stop()

    if isolated:
        hero(
            "HUMAN STUDY · CALIBRATED RELIANCE",
            "Trust Calibration",
            "Evaluate explanation evidence one item at a time. Correctness labels and researcher ground truth are hidden.",
            pills=["Phase 16", "Track 2", "Blinded", mode.upper()],
        )
    else:
        st.info(
            "Participant preview only. For real data collection use the isolated `phase16_participant_app.py` entry point "
            "so participants cannot navigate to Phase-15 researcher evidence."
        )

    if mode == "pilot":
        st.warning(
            "PILOT mode — the trust instrument/power calculation may still be provisional. Pilot records are stored "
            "separately and are never mixed into FINAL analysis."
        )
    else:
        st.success("FINAL study mode · frozen design · participant blinding enabled")

    participant_hash = _participant_registration(study, mode)
    if participant_hash is None:
        return

    progress = study.participant_progress(participant_hash, mode)
    trial = study.current_trial(participant_hash, mode)
    if trial is None:
        st.success("Study complete. Thank you for your participation.")
        st.progress(1.0)
        st.caption("Your responses have been stored under a pseudonymous participant identifier.")
        if not isolated and st.button("Clear local participant session", key="p16_clear_completed"):
            _reset_participant_session()
            st.rerun()
        return

    completed = int(progress["completed"])
    total = int(progress["total"])
    st.progress(float(completed) / float(max(total, 1)))
    st.caption("Completed {} of {} · current item {}".format(completed, total, trial["order"]))

    st.markdown(
        """
        <div class="r9-callout" style="border-left:4px solid #2563eb;padding:18px 20px">
          <div style="font-size:.76rem;letter-spacing:.08em;text-transform:uppercase;opacity:.65">Item {item_id}</div>
          <div style="font-size:.92rem;margin-top:8px"><b>Context</b> · {source}</div>
          <div style="font-size:1.02rem;line-height:1.55;margin-top:10px"><b>Prompt</b><br>{prompt}</div>
          <div style="font-size:1.05rem;line-height:1.62;margin-top:14px"><b>Explanation</b><br>{explanation}</div>
        </div>
        """.format(
            item_id=trial["item_id"],
            source=trial["source"],
            prompt=trial["prompt"],
            explanation=trial["explanation"],
        ),
        unsafe_allow_html=True,
    )
    st.caption("The study does not reveal whether this explanation is correct or incorrect.")

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

    with st.form("p16_trial_form_{}".format(trial["item_id"])):
        st.markdown("#### Your response")
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
                "How confident / authoritative does the explanation sound?",
                1,
                7,
                4,
                key="p16_confidence_{}".format(trial["item_id"]),
            )

        decision_action = st.radio(
            "What would you do with this recommendation/explanation?",
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


def _render_researcher_console(study: TrustCalibrationStudy) -> None:
    if not _render_researcher_unlock():
        return

    st.success("Researcher console unlocked. Correctness labels below must not be shown to participants.")
    protocol = study.protocol
    bank_summary = study.bank.validate_design_pool(
        int((protocol.get("design") or {}).get("shared_pool_min_per_condition", 20))
    )

    section("Design gate", "Final collection stays locked until the protocol, trust measure, power calculation and ethics fields are finalized and frozen.")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Protocol", str(protocol.get("status", "unknown")).upper())
    p2.metric("Trust measure", str((protocol.get("trust_measure") or {}).get("status", "unknown")).upper())
    p3.metric("Power", str((protocol.get("power_calculation") or {}).get("status", "unknown")).upper())
    p4.metric("Design frozen", "YES" if FINAL_DESIGN_MANIFEST.exists() else "NO")

    with st.expander("Protocol JSON", expanded=False):
        st.json(protocol, expanded=False)
    with st.expander("Verified Phase-15 stimulus-bank summary", expanded=False):
        st.json(bank_summary, expanded=False)

    section("Participant-facing blinding preview", "This preview never shows condition labels and does not write responses.")
    preview = study.build_preview_assignment("advanced-labs-preview")
    preview_index = st.slider("Preview order", 1, len(preview), 1, key="p16_preview_order")
    item = preview[preview_index - 1]
    st.markdown("**{} · {}**".format(item["item_id"], item["source"]))
    st.write(item["prompt"])
    st.info(item["explanation"])

    with st.expander("Researcher-only stimulus ground truth", expanded=False):
        stimulus_ids = sorted(study.bank.stimuli)
        selected = st.selectbox("Stimulus", stimulus_ids, key="p16_researcher_stimulus")
        stimulus = study.bank.get(selected)
        st.json(
            {
                "stimulus_id": stimulus.stimulus_id,
                "pair_id": stimulus.pair_id,
                "source_stage": stimulus.source_stage,
                "condition": stimulus.correctness_condition,
                "researcher_label": stimulus.researcher_label,
                "error_type": stimulus.error_type,
                "explanation_sha256": stimulus.explanation_sha256,
            }
        )

    section("Collection status", "PILOT and FINAL records are physically separated by collection_mode in the append-only store.")
    mode = st.radio("Researcher dataset", ["pilot", "final"], horizontal=True, key="p16_research_mode")
    summary = study.store.summary(mode)
    m1, m2, m3 = st.columns(3)
    m1.metric("Registered", summary["registered_participants"])
    m2.metric("Completed", summary["completed_participants"])
    m3.metric("Responses", summary["responses"])
    st.caption("Completed by expertise: {}".format(summary["completed_by_expertise"] or "none"))

    participants = pd.DataFrame(study.store.participant_rows(mode))
    assignments = pd.DataFrame(study.store.all_assignment_rows(mode))
    responses = pd.DataFrame(study.store.response_rows(mode))
    d1, d2, d3 = st.columns(3)
    with d1:
        _download_frame("Download participant summary", participants, "phase16_{}_participants.csv".format(mode), "p16_dl_participants")
    with d2:
        _download_frame("Download assignments", assignments, "phase16_{}_assignments.csv".format(mode), "p16_dl_assignments")
    with d3:
        _download_frame("Download responses", responses, "phase16_{}_responses.csv".format(mode), "p16_dl_responses")

    if st.button("Run descriptive calibration analysis", key="p16_run_analysis", use_container_width=True):
        result = analyze_phase16_store(study.store, mode)
        st.session_state.p16_analysis_result = result
    if st.session_state.get("p16_analysis_result"):
        st.json(st.session_state.p16_analysis_result, expanded=False)

    section("Finalization commands", "Freezing remains command-line gated so a UI click cannot accidentally create journal-facing final evidence.")
    st.code(
        "python scripts/freeze_phase16_design.py\n"
        "python scripts/analyze_phase16_trust.py --mode final\n"
        "python scripts/freeze_phase16_results.py",
        language="powershell",
    )
    st.caption(
        "Do not run the design freeze until Morten has finalized the validated trust measure and power calculation, "
        "and your ethics/participant materials are finalized."
    )


def phase16_trust_calibration_page() -> None:
    hero(
        "HUMAN STUDY · CALIBRATED RELIANCE",
        "Trust Calibration",
        (
            "Phase 16 tests whether participant trust tracks actual explanation correctness or instead follows "
            "surface fluency and confident-sounding language. The previous top/second/worst utility scaffold is retired."
        ),
        pills=phase_pills(),
    )
    try:
        study = TrustCalibrationStudy()
    except Exception as exc:
        st.error("Phase-16 study could not initialize: {}".format(exc))
        return

    _protocol_status_cards(study)
    st.markdown(
        """
        <div class="r9-callout">
          <b>Phase-16 design.</b> Within-subject · randomized · blinded · correct vs incorrect explanation stimuli ·
          one variant per underlying pair · balanced source stages · response-time logging · expertise-stratified analysis.
          Track-2 participants must not be people who authored Track-1 scenarios.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_preview, tab_researcher = st.tabs(["Participant preview", "Researcher console"])
    with tab_preview:
        st.info(
            "This is a no-save preview inside Advanced Labs. Actual participants should use the isolated "
            "`phase16_participant_app.py` entry point so they cannot navigate into Phase-15 researcher ground truth."
        )
        preview = study.build_preview_assignment("advanced-labs-blinded-preview")
        idx = st.slider("Preview item", 1, len(preview), 1, key="p16_public_preview")
        item = preview[idx - 1]
        st.markdown("#### {} · {}".format(item["item_id"], item["source"]))
        st.write(item["prompt"])
        st.info(item["explanation"])
        st.caption("Correctness condition intentionally hidden in this preview.")
        st.code("streamlit run phase16_participant_app.py", language="powershell")
    with tab_researcher:
        _render_researcher_console(study)
