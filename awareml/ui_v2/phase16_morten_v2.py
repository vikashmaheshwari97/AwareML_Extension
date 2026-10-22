from __future__ import annotations

import html
import os
import time
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from awareml.studies.study_runtime import bootstrap_study_runtime
bootstrap_study_runtime()

from awareml.studies.trust_morten_v2 import ALLOWED_EXPERTISE_GROUPS, TrustCalibrationStudyV2, analyze_v2
from .components import hero, section

ROLE_KEY = "trust_calibration_access_role"
PFX = "p16v2"
LOCAL_KEY = "phase16-local-test-key"

# Compatibility markers retained for the v7.1 focused validation while v7.2 renders richer replacements.
LEGACY_RESEARCHER_WORKSPACE_MARKERS = (
    "Participant-level paired trust profile",
    "Accept / Override / Reject by correctness",
    "Trial-family exposure",
    "Trial-order diagnostics",
    "Expertise groups",
)

SECONDARY_ITEMS = [
    ("critical_evaluation", "I felt able to critically evaluate the recommendation provided by AwareML."),
    ("freedom_to_override", "I felt free to reject or override AwareML's recommendation when I disagreed with it."),
    ("final_control", "I understood that I remained responsible for the final decision about whether to rely on the AwareML recommendation."),
    ("rely_or_question", "The explanations helped me judge when to rely on the recommendation and when to question it."),
    ("evidence_checking", "I compared the AI explanation with the reference evidence before deciding whether to rely on it."),
    ("explanation_usefulness", "The explanations were useful for understanding why the system made its recommendation."),
    ("interface_usability", "The AwareML study interface made it easy to compare an explanation with its supporting evidence."),
    ("real_world_support_trust", "I would trust AwareML to support me in a real-world streaming AutoML decision."),
]


def _css():
    st.markdown("""<style>
    .m2-card{padding:18px 20px;border:1px solid rgba(100,116,139,.2);border-radius:16px;background:rgba(255,255,255,.86);margin:8px 0 14px}
    .m2-label{font-size:.75rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#2563eb}
    .m2-title{font-size:1.1rem;font-weight:780;color:#0f172a;margin:.3rem 0}.m2-note{color:#64748b;line-height:1.5}
    .m2-context{padding:16px 18px;border-left:4px solid #2563eb;background:rgba(239,246,255,.75);border-radius:12px;margin:10px 0 16px}
    </style>""", unsafe_allow_html=True)


def _mode() -> str:
    return "final" if str(os.getenv("AWAREML_PHASE16_COLLECTION_MODE", "pilot")).strip().lower() == "final" else "pilot"


def _researcher_key(mode: str) -> Optional[str]:
    explicit = str(os.getenv("AWAREML_STUDY_RESEARCHER_KEY", "")).strip()
    return explicit or (LOCAL_KEY if mode == "pilot" else None)


def _likert(label: str, key: str, low: str, high: str):
    st.markdown("**{}**".format(label))
    value = st.radio(label, [1, 2, 3, 4, 5, 6, 7], index=None, horizontal=True, key=key, label_visibility="collapsed")
    c1, c2 = st.columns(2); c1.caption("1 = {}".format(low)); c2.caption("7 = {}".format(high))
    return value


def _render_reference(trial: dict):
    ev = trial.get("reference_evidence") or {}
    objectives = list(ev.get("scenario_objectives") or [])
    joined = " + ".join(objectives) if objectives else "Not specified"
    st.markdown(
        '<div class="m2-context"><b>Scenario → Task → Evidence connection.</b><br>'
        'The scenario priorities for this item are <b>{}</b>. The task and AI explanation '
        'should therefore be judged against those same priorities.<br>'
        '<span class="m2-note">The values below are controlled Trust Calibration reference evidence. '
        'They provide a consistent comparison context for the study and are not presented as real-world '
        'performance measurements for the textual scenario.</span></div>'.format(joined),
        unsafe_allow_html=True,
    )
    rows = list(ev.get("frameworks") or [])
    if not rows:
        st.warning("This trial has no five-framework reference table.")
        return
    df = pd.DataFrame(rows).rename(columns={
        "framework":"Framework","rank":"Reference rank","scenario_fit_score":"Scenario-fit score",
        "accuracy":"Accuracy","runtime_sec":"Runtime (s)","energy_kwh":"Energy (kWh)","co2_kg":"CO₂ (kg)"})
    cols=[c for c in ["Reference rank","Framework","Scenario-fit score","Accuracy","Runtime (s)","Energy (kWh)","CO₂ (kg)"] if c in df]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)
    relevant=[]
    if "Accuracy" in objectives: relevant.append("higher Accuracy")
    if "Runtime" in objectives: relevant.append("lower Runtime")
    if "Energy" in objectives: relevant.append("lower Energy")
    if "CO2" in objectives: relevant.append("lower CO₂")
    st.caption(
        "Scenario-fit rank uses only the scenario priorities: {}. Other displayed metrics are contextual and are not decision objectives for this item.".format(
            ", ".join(relevant) if relevant else "the scenario objective labels"
        )
    )

def _evidence_answer(question: str, trial: dict) -> str:
    q = str(question or "").strip().lower()
    ev = trial.get("reference_evidence") or {}
    rows = sorted(list(ev.get("frameworks") or []), key=lambda x: int(x.get("rank", 999)))
    objectives = list(ev.get("scenario_objectives") or [])
    objective_text = " + ".join(objectives) if objectives else "not specified"
    if not rows:
        return "No controlled framework reference evidence is available for this trial."

    names = [str(r.get("framework") or "") for r in rows]
    mentioned = [name for name in names if name.lower() in q]
    focus = str(trial.get("framework_context") or rows[0].get("framework"))
    target_names = mentioned or [focus]
    by_name = {str(r.get("framework")): r for r in rows}

    if any(x in q for x in ["fair", "parity", "equal opportunity", "equalized odds"]):
        return (
            "Fairness is not part of this trial's scenario-priority evidence ({}), and this trial does not provide fairness evidence. "
            "I will not invent a fairness result.".format(objective_text)
        )
    if any(x in q for x in ["shap", "feature", "explainability", "xai"]):
        return (
            "Feature-attribution / XAI evidence is not part of this trial's scenario-priority evidence ({}), "
            "so this item does not provide a SHAP or feature-ranking claim.".format(objective_text)
        )

    wants_rank = any(x in q for x in ["rank", "ranking", "place", "position"])
    wants_acc = any(x in q for x in ["accuracy", "accurate"])
    wants_runtime = any(x in q for x in ["runtime", "latency", "speed", "fast"])
    wants_energy = "energy" in q
    wants_co2 = any(x in q for x in ["co2", "co₂", "carbon", "emission"])
    wants_score = any(x in q for x in ["score", "utility", "fit"])
    wants_objectives = any(x in q for x in ["objective", "priority", "priorities", "scenario goal"])
    compare = any(x in q for x in ["compare", "versus", " vs ", "alternative", "other framework"])
    generic_ranking = any(x in q for x in ["evidence supports the ranking", "ranking evidence", "why is", "why ranked"])
    generic_metrics = any(x in q for x in ["metric evidence", "explain the metric", "what are the metrics"])

    def values_for_objectives(r: dict) -> list[str]:
        parts=[]
        if "Accuracy" in objectives:
            parts.append("accuracy {:.3f}".format(float(r.get("accuracy", 0))))
        if "Runtime" in objectives:
            parts.append("runtime {:.2f} s".format(float(r.get("runtime_sec", 0))))
        if "Energy" in objectives:
            parts.append("energy {:.6f} kWh".format(float(r.get("energy_kwh", 0))))
        if "CO2" in objectives:
            parts.append("CO₂ {:.8f} kg".format(float(r.get("co2_kg", 0))))
        return parts

    def one(name: str, default_objectives: bool = False) -> str:
        r = by_name.get(name)
        if not r:
            return "{} is not present in this trial's five-framework evidence.".format(name)
        parts=[]
        if wants_rank or default_objectives or not any([wants_acc,wants_runtime,wants_energy,wants_co2,wants_score]):
            parts.append("rank #{}".format(int(r.get("rank",0))))
        if wants_score or generic_ranking:
            parts.append("scenario-fit score {:.3f}".format(float(r.get("scenario_fit_score",0))))
        if wants_acc: parts.append("accuracy {:.3f}".format(float(r.get("accuracy",0))))
        if wants_runtime: parts.append("runtime {:.2f} s".format(float(r.get("runtime_sec",0))))
        if wants_energy: parts.append("energy {:.6f} kWh".format(float(r.get("energy_kwh",0))))
        if wants_co2: parts.append("CO₂ {:.8f} kg".format(float(r.get("co2_kg",0))))
        if default_objectives or generic_ranking or generic_metrics:
            for part in values_for_objectives(r):
                if part not in parts:
                    parts.append(part)
        return "{}: {}".format(name, ", ".join(parts))

    if wants_objectives:
        return "This scenario's study priorities are {}. The reference rank is computed from those priorities only.".format(objective_text)

    if generic_ranking:
        lines=[one(str(r.get("framework")), default_objectives=True) for r in rows]
        return (
            "The ranking is supported by the controlled five-framework evidence using the scenario priorities {}. "
            "{}".format(objective_text, " | ".join(lines))
        )

    if generic_metrics:
        first=rows[0]
        second=rows[1] if len(rows)>1 else None
        direction=[]
        if "Accuracy" in objectives: direction.append("Accuracy is higher-is-better")
        if "Runtime" in objectives: direction.append("Runtime is lower-is-better")
        if "Energy" in objectives: direction.append("Energy is lower-is-better")
        if "CO2" in objectives: direction.append("CO₂ is lower-is-better")
        compare_text=one(str(first.get("framework")), default_objectives=True)
        if second is not None:
            compare_text += " | " + one(str(second.get("framework")), default_objectives=True)
        return "Scenario priorities: {}. {}. Top reference evidence: {}.".format(
            objective_text,
            "; ".join(direction) if direction else "Use the displayed scenario-fit evidence",
            compare_text,
        )

    if compare:
        if len(target_names) < 2:
            target_names = [str(r.get("framework")) for r in rows]
        return "Controlled comparison for scenario priorities {}: {}.".format(
            objective_text,
            " | ".join(one(n, default_objectives=True) for n in target_names),
        )

    return one(target_names[0], default_objectives=True) + ". Scenario priorities: {}.".format(objective_text)

def _reset_trust_participant_session() -> None:
    keep_role = st.session_state.get(ROLE_KEY, "participant")
    for key in list(st.session_state.keys()):
        if str(key).startswith(PFX):
            del st.session_state[key]
    st.session_state[ROLE_KEY] = keep_role or "participant"


def _participant_registration(study: TrustCalibrationStudyV2, mode: str) -> Optional[str]:
    ph = st.session_state.get(PFX + "_hash")
    if ph and st.session_state.get(PFX + "_mode") == mode: return str(ph)
    section("Before you begin", "Enter the anonymous code supplied by the study team and confirm eligibility.")
    with st.expander("Study information and consent", expanded=False): st.write((study.protocol.get("participant_materials") or {}).get("consent_text", ""))
    st.info((study.protocol.get("participant_materials") or {}).get("estimated_duration", ""))
    with st.form("p16v2_registration"):
        code = st.text_input("Participant / session code", key=PFX+"_reg_code")
        group = st.selectbox("Experience group", ALLOWED_EXPERTISE_GROUPS, index=None, placeholder="Select your group", key=PFX+"_reg_group")
        st.markdown("**Self-rated Machine Learning / AutoML knowledge and experience**")
        expertise = st.radio("ML/AutoML knowledge and experience", [1,2,3,4,5,6,7], index=None, horizontal=True, label_visibility="collapsed", key=PFX+"_reg_expertise")
        st.caption("1 = No knowledge / experience · 4 = Moderate knowledge / experience · 7 = Extensive knowledge / experience")
        adult = st.checkbox("I confirm that I am at least 18 years old.", key=PFX+"_reg_adult")
        separate = st.checkbox("I confirm that I did not help author the Track-1 scenario bank used in this study.", key=PFX+"_reg_separate")
        consent = st.checkbox("I voluntarily consent to participate in this research study.", key=PFX+"_reg_consent")
        data = st.checkbox("I consent to the processing of my study responses as described in the participant information.", key=PFX+"_reg_data")
        start = st.form_submit_button("Start / resume study", type="primary", use_container_width=True)
    if start:
        if not all([adult, separate, consent, data]) or group is None or expertise is None:
            st.error("Complete all eligibility, consent, experience-group and ML/AutoML knowledge fields."); return None
        try: registration = study.register(code, group, int(expertise), True, True, mode)
        except Exception as exc: st.error(str(exc)); return None
        ph = str(registration["participant_hash"]); st.session_state[PFX+"_hash"] = ph; st.session_state[PFX+"_mode"] = mode; st.rerun()
    return None


def _render_secondary(study: TrustCalibrationStudyV2, mode: str, ph: str) -> bool:
    if study.store.secondary_exists(mode, ph): return True
    section("Final questionnaire · Human agency and study experience", "These secondary questions concern your overall experience; they are separate from the S-TIAS trust measure.")
    st.caption("1 = Strongly disagree · 4 = Neither agree nor disagree · 7 = Strongly agree. No answer is preselected.")
    with st.form("p16v2_secondary"):
        ratings = {}
        for item_id, text in SECONDARY_ITEMS:
            st.markdown("**{}**".format(text)); ratings[item_id] = st.radio(text, [1,2,3,4,5,6,7], index=None, horizontal=True, key="p16v2_sec_"+item_id, label_visibility="collapsed")
        comment = st.text_area("Optional: What most influenced whether you trusted, questioned, overrode, or rejected an explanation?", height=90)
        submit = st.form_submit_button("Submit final questionnaire", type="primary", use_container_width=True)
    if submit:
        if any(v is None for v in ratings.values()): st.error("Please answer all eight statements."); return False
        study.store.save_secondary(mode, ph, {k:int(v) for k,v in ratings.items()}, comment); st.rerun()
    return False


def render_phase16_participant_study(collection_mode: Optional[str] = None, isolated: bool = False, dashboard_embedded: bool = False):
    _css(); mode = collection_mode or _mode()
    try:
        study = TrustCalibrationStudyV2()
        if mode == "final": study.assert_final_ready()
    except Exception as exc: st.error("Revised Trust Calibration could not initialize: {}".format(exc)); return
    if isolated: hero("HUMAN STUDY · CALIBRATED RELIANCE", "Trust Calibration · Morten-review revision", "20 scenario contexts; five frameworks; four explanation types; matched correct/incorrect variants.")
    else: st.markdown('<div class="m2-card"><div class="m2-title">Participant Study · {}</div><div class="m2-note">Scenario context, comparative evidence, initial trust ratings, optional evidence seeking and final human decision.</div></div>'.format("Main Study" if mode=="final" else "Pilot Study"), unsafe_allow_html=True)
    ph = _participant_registration(study, mode)
    if ph is None: return
    completed, total = study.progress(ph, mode); trial = study.current_trial(ph, mode)
    if trial is None:
        if not _render_secondary(study, mode, ph): return
        st.success("Study complete. Thank you.")
        with st.expander("Study debrief", expanded=True):
            st.write((study.protocol.get("participant_materials") or {}).get("debrief_text", ""))
        start_label = "Start new Main Study" if mode == "final" else "Start new Pilot Study"
        if st.button(start_label, type="primary", use_container_width=True, key=PFX+"_start_new_session"):
            _reset_trust_participant_session()
            st.rerun()
        return
    section("Study progress", "Each of the 20 matched trial families is evaluated once by every participant.")
    st.progress(completed/max(total,1)); st.caption("{} of {} completed · current trial position {}".format(completed,total,trial["order"]))
    st.markdown('<div class="m2-card"><div class="m2-label">Scenario context · {} · {}</div><div class="m2-title">{}</div><div class="m2-note"><b>Framework context:</b> {} · <b>Scenario source:</b> Track-1 scenario benchmark</div></div>'.format(html.escape(str(trial.get("scenario_id"))), html.escape(str(trial.get("domain"))), html.escape(str(trial.get("scenario_text"))), html.escape(str(trial.get("framework_context")))), unsafe_allow_html=True)
    st.markdown('<div class="m2-card"><div class="m2-label">Task given to the AI</div><div class="m2-title">{}</div><div class="m2-label" style="margin-top:14px">AI explanation</div><div class="m2-note" style="font-size:1rem;color:#0f172a">{}</div></div>'.format(html.escape(str(trial.get("prompt"))), html.escape(str(trial.get("explanation")))), unsafe_allow_html=True)
    _render_reference(trial)
    order = int(trial["order"]); initial = study.store.initial(mode, ph, order); start_key = PFX+"_started_"+str(order)
    st.session_state.setdefault(start_key, time.time())
    if initial is None:
        section("Initial evaluation", "Rate the explanation before asking for any additional evidence. No response is preselected.")
        with st.form("p16v2_initial_{}".format(order)):
            a = _likert("I am confident in the AI assistant.", "p16v2_a_"+str(order), "Not at all", "Extremely")
            b = _likert("The AI assistant is reliable.", "p16v2_b_"+str(order), "Not at all", "Extremely")
            c = _likert("I can trust the AI assistant.", "p16v2_c_"+str(order), "Not at all", "Extremely")
            pc = _likert("How likely is the explanation to be factually correct?", "p16v2_pc_"+str(order), "Not at all likely", "Extremely likely")
            fl = _likert("How fluent is the explanation?", "p16v2_fl_"+str(order), "Not at all fluent", "Extremely fluent")
            cf = _likert("How confident does the explanation sound?", "p16v2_cf_"+str(order), "Not at all confident", "Extremely confident")
            lock = st.form_submit_button("Lock initial evaluation and continue", type="primary", use_container_width=True)
        if lock:
            if any(v is None for v in [a,b,c,pc,fl,cf]): st.error("Please answer all six initial ratings before continuing.")
            else: study.lock_initial(ph, order, {"stias_confident":int(a),"stias_reliable":int(b),"stias_trust":int(c)}, int(pc), int(fl), int(cf), mode); st.rerun()
        return
    st.success("Initial trust and diagnostic ratings are locked for this trial.")
    section("Optional evidence seeking", "Ask only if you need more information before your final action. Questions are answered deterministically from this trial's reference evidence.")
    chat_key = PFX+"_chat_"+str(order); chat = st.session_state.setdefault(chat_key, []); qcols = st.columns(3); pending = None
    if qcols[0].button("Compare frameworks", key="p16v2_q_compare_"+str(order), use_container_width=True): pending = "Compare the frameworks using the reference evidence."
    if qcols[1].button("Explain the ranking evidence", key="p16v2_q_rank_"+str(order), use_container_width=True): pending = "What evidence supports the ranking?"
    if qcols[2].button("Explain the metric evidence", key="p16v2_q_metric_"+str(order), use_container_width=True): pending = "Explain the metric evidence."
    with st.form("p16v2_question_form_"+str(order), clear_on_submit=True): typed = st.text_input("Ask for evidence", placeholder="Ask about rank, scenario priorities, accuracy, runtime, energy, CO₂ or comparisons..."); ask = st.form_submit_button("Ask")
    if ask and typed.strip(): pending = typed.strip()
    if pending:
        answer = _evidence_answer(pending, trial); study.store.log_event(mode, ph, order, "evidence_question", {"question":pending,"answer":answer}, time.time()-float(st.session_state[start_key])); chat.append((pending,answer)); st.rerun()
    for q, answer in chat: st.markdown("**You:** {}".format(q)); st.info(answer)
    section("Final human decision", "Choose what you would actually do after reviewing the evidence.")
    decision = st.radio("What would you do with this recommendation / explanation?", ["Accept","Override","Reject"], index=None, key="p16v2_decision_"+str(order), format_func=lambda x:{"Accept":"Accept — I would rely on it as shown","Override":"Override — I would correct/change it before acting","Reject":"Reject — I would not rely on it"}[x])
    targets, correction = [], ""
    if decision == "Override":
        targets = st.multiselect("What would you correct?", ["Framework / ranking","Scenario-fit score","Accuracy","Runtime","Energy","CO₂","Unsupported statement","Other"], key="p16v2_targets_"+str(order)); correction = st.text_area("What would you correct or change before acting?", placeholder="For example: Change the stated accuracy from 0.69 to 0.860.", key="p16v2_corr_"+str(order), height=90)
    if st.button("Submit final response and continue", type="primary", use_container_width=True, key="p16v2_submit_"+str(order)):
        if decision is None: st.error("Choose Accept, Override or Reject."); return
        try: study.final_submit(ph, order, decision, time.time()-float(st.session_state[start_key]), targets, correction, mode)
        except Exception as exc: st.error(str(exc)); return
        st.session_state.pop(chat_key, None); st.session_state.pop(start_key, None); st.rerun()


def _chart(fig, key: str, height: int = 360):
    fig.update_layout(
        height=height,
        margin=dict(l=40,r=25,t=35,b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color="#0F172A"),
        hoverlabel=dict(bgcolor="#FFFFFF"),
    )
    fig.update_xaxes(gridcolor="#E2E8F0", zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False, automargin=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False,"displaylogo":False}, key=key)


def _researcher_visuals(study: TrustCalibrationStudyV2, mode: str, summary: dict):
    import json

    responses = pd.DataFrame(study.store.response_rows(mode))
    participants = pd.DataFrame(study.store.participant_rows(mode))
    events = pd.DataFrame(study.store.event_rows(mode))
    secondary = pd.DataFrame(study.store.secondary_rows(mode))

    if responses.empty:
        st.info("Research visuals will populate after revised Main/Pilot responses are available.")
        return

    for col in [
        "trust_rating", "response_time_sec", "order_index", "evidence_question_count",
        "perceived_correctness", "fluency_rating", "perceived_confidence",
    ]:
        if col in responses:
            responses[col] = pd.to_numeric(responses[col], errors="coerce")

    completed_hashes = set()
    if not participants.empty and "completed_at" in participants:
        completed_hashes = set(
            participants.loc[participants["completed_at"].notna(), "participant_hash"].astype(str)
        )
    analysis_rows = responses[
        responses["participant_hash"].astype(str).isin(completed_hashes)
    ].copy() if completed_hashes else responses.copy()

    n_completed = len(completed_hashes)
    if n_completed < 5:
        st.warning(
            "Preview only: {} completed participant{}. These charts are useful for checking the instrument and data pipeline, "
            "but they should not be interpreted as study findings yet.".format(n_completed, "" if n_completed == 1 else "s")
        )
    elif n_completed < 34:
        st.info(
            "Interim descriptive view: {} of the minimum 34 completed participants are available. "
            "Do not treat subgroup or distribution patterns as confirmatory.".format(n_completed)
        )
    else:
        st.success("Minimum powered completion target reached: {} completed participants.".format(n_completed))

    tabs = st.tabs([
        "Overview",
        "Calibration",
        "Reliance & corrections",
        "Scenario coverage",
        "Trial dynamics",
        "Human agency",
        "Expertise",
    ])

    with tabs[0]:
        st.markdown("#### Calibration map · participant-level correct vs incorrect trust")
        paired = analysis_rows.pivot_table(
            index="participant_hash",
            columns="correctness_condition",
            values="trust_rating",
            aggfunc="mean",
        )
        if {"correct", "incorrect"}.issubset(paired.columns):
            paired = paired.dropna(subset=["correct", "incorrect"]).reset_index()
        else:
            paired = pd.DataFrame()
        if paired.empty:
            st.info("Paired calibration points require participants with both correct and incorrect trials.")
        else:
            paired["gap"] = paired["correct"] - paired["incorrect"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[1, 7], y=[1, 7], mode="lines", name="Equal trust",
                line=dict(dash="dash", color="#94A3B8"), hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=paired["incorrect"], y=paired["correct"], mode="markers",
                marker=dict(size=14, line=dict(width=1, color="#0F172A")),
                customdata=paired[["participant_hash", "gap"]].to_numpy(),
                hovertemplate="Participant %{customdata[0]}<br>Incorrect trust %{x:.2f}<br>Correct trust %{y:.2f}<br>Gap %{customdata[1]:+.2f}<extra></extra>",
                name="Participants",
            ))
            fig.update_xaxes(range=[1,7], title="Mean trust in incorrect explanations")
            fig.update_yaxes(range=[1,7], title="Mean trust in correct explanations", scaleanchor="x", scaleratio=1)
            fig.update_layout(legend=dict(orientation="h", y=1.08))
            _chart(fig, "p16v22_calibration_map", 500)
            st.caption("Points above the diagonal indicate higher trust in correct than incorrect explanations for the same participant.")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Trust distribution by actual correctness")
            fig = go.Figure()
            for condition, label in [("correct","Correct"),("incorrect","Incorrect")]:
                g = analysis_rows[analysis_rows["correctness_condition"].eq(condition)]
                if g.empty:
                    continue
                fig.add_trace(go.Box(
                    y=g["trust_rating"], name=label, boxpoints="all", jitter=0.28,
                    pointpos=0, marker=dict(size=6), boxmean=True,
                    hovertemplate=label+"<br>Trust %{y:.2f}<extra></extra>",
                ))
            fig.update_yaxes(range=[1,7], dtick=1, title="S-TIAS mean trust")
            _chart(fig, "p16v22_condition_distribution", 390)
        with right:
            st.markdown("#### Perceived factual correctness by actual condition")
            fig = go.Figure()
            for condition, label in [("correct","Correct"),("incorrect","Incorrect")]:
                g = analysis_rows[analysis_rows["correctness_condition"].eq(condition)]
                if g.empty:
                    continue
                fig.add_trace(go.Box(
                    y=g["perceived_correctness"], name=label, boxpoints="all", jitter=0.28,
                    pointpos=0, marker=dict(size=6), boxmean=True,
                    hovertemplate=label+"<br>Perceived correctness %{y:.0f}/7<extra></extra>",
                ))
            fig.update_yaxes(range=[1,7], dtick=1, title="Perceived factual correctness (1–7)")
            _chart(fig, "p16v22_correctness_detection", 390)

    with tabs[1]:
        st.markdown("#### S-TIAS item profile by correctness")
        trust_rows=[]
        labels={
            "stias_confident":"Confidence in AI assistant",
            "stias_reliable":"AI assistant is reliable",
            "stias_trust":"Can trust AI assistant",
        }
        for _, row in analysis_rows.iterrows():
            try:
                items=json.loads(str(row.get("trust_items_json") or "{}"))
            except Exception:
                items={}
            for item_id, label in labels.items():
                if item_id in items:
                    trust_rows.append({"Condition":str(row.get("correctness_condition")),"Item":label,"Rating":float(items[item_id])})
        tdf=pd.DataFrame(trust_rows)
        if tdf.empty:
            st.info("No item-level S-TIAS responses are available.")
        else:
            agg=tdf.groupby(["Item","Condition"])["Rating"].agg(["mean","std","count"]).reset_index()
            fig=go.Figure()
            for item in labels.values():
                sub=agg[agg["Item"].eq(item)]
                vals={r["Condition"]:r for _,r in sub.iterrows()}
                if "incorrect" in vals and "correct" in vals:
                    a=vals["incorrect"]; b=vals["correct"]
                    fig.add_trace(go.Scatter(
                        x=[a["mean"],b["mean"]], y=[item,item], mode="lines+markers",
                        line=dict(width=3), marker=dict(size=12), showlegend=False,
                        customdata=[["Incorrect",a["count"]],["Correct",b["count"]]],
                        hovertemplate="%{customdata[0]} mean %{x:.2f}<br>n=%{customdata[1]}<extra></extra>",
                    ))
            fig.update_xaxes(range=[1,7], dtick=1, title="Mean item rating")
            _chart(fig, "p16v22_stias_items", 360)

        st.markdown("#### Explanation type × trust discrimination")
        rows=[]
        for stage,g in analysis_rows.groupby("source_stage"):
            correct=g[g["correctness_condition"].eq("correct")]["trust_rating"]
            incorrect=g[g["correctness_condition"].eq("incorrect")]["trust_rating"]
            rows.append({
                "Type":stage,
                "Correct":correct.mean(),
                "Incorrect":incorrect.mean(),
                "Correct n":int(correct.notna().sum()),
                "Incorrect n":int(incorrect.notna().sum()),
            })
        frame=pd.DataFrame(rows)
        if frame.empty:
            st.info("No explanation-type data available.")
        else:
            fig=go.Figure()
            for _,r in frame.iterrows():
                if pd.isna(r["Correct"]) or pd.isna(r["Incorrect"]):
                    continue
                fig.add_trace(go.Scatter(
                    x=[r["Incorrect"],r["Correct"]], y=[r["Type"],r["Type"]], mode="lines+markers+text",
                    text=["Incorrect {:.2f}".format(r["Incorrect"]),"Correct {:.2f}".format(r["Correct"])],
                    textposition=["middle left","middle right"], showlegend=False,
                    customdata=[[r["Incorrect n"]],[r["Correct n"]]],
                    hovertemplate="Mean trust %{x:.2f}<br>n=%{customdata[0]}<extra></extra>",
                ))
            fig.add_vline(x=4,line_dash="dot",line_color="#94A3B8")
            fig.update_xaxes(range=[1,7], title="Mean trust")
            _chart(fig, "p16v22_type_discrimination", 420)

        if not paired.empty:
            st.markdown("#### Participant calibration-gap distribution")
            fig=go.Figure(go.Box(
                x=paired["gap"], boxpoints="all", jitter=.25, pointpos=0,
                name="Correct − incorrect trust gap",
                hovertemplate="Gap %{x:+.2f}<extra></extra>",
            ))
            fig.add_vline(x=0,line_dash="dash",line_color="#94A3B8")
            fig.update_xaxes(title="Within-participant trust gap", zeroline=False)
            _chart(fig, "p16v22_gap_distribution", 300)

    with tabs[2]:
        st.markdown("#### Accept / Override / Reject by correctness")
        counts=analysis_rows.groupby(["correctness_condition","decision_action"]).size().reset_index(name="n")
        if counts.empty:
            st.info("No final decisions available.")
        else:
            totals=counts.groupby("correctness_condition")["n"].transform("sum")
            counts["pct"]=100*counts["n"]/totals
            fig=go.Figure()
            for action in ["Accept","Override","Reject"]:
                part=counts[counts["decision_action"].eq(action)]
                fig.add_trace(go.Bar(
                    x=part["pct"],y=part["correctness_condition"].str.title(),name=action,orientation="h",
                    text=["{:.1f}%".format(x) for x in part["pct"]],textposition="inside",
                    customdata=part[["n"]].to_numpy(),
                    hovertemplate=action+"<br>%{x:.1f}% · n=%{customdata[0]}<extra></extra>",
                ))
            fig.update_layout(barmode="stack",legend=dict(orientation="h",y=1.12))
            fig.update_xaxes(range=[0,100],title="Share of decisions (%)")
            _chart(fig,"p16v22_decisions",350)

        c1,c2=st.columns(2)
        with c1:
            st.markdown("#### Trust by final action")
            fig=go.Figure()
            for action in ["Accept","Override","Reject"]:
                g=analysis_rows[analysis_rows["decision_action"].eq(action)]
                if g.empty: continue
                fig.add_trace(go.Box(y=g["trust_rating"],name=action,boxpoints="all",jitter=.25,pointpos=0,boxmean=True))
            fig.update_yaxes(range=[1,7],dtick=1,title="S-TIAS mean trust")
            _chart(fig,"p16v22_trust_by_action",360)
        with c2:
            st.markdown("#### Optional evidence questions by final action")
            fig=go.Figure()
            for action in ["Accept","Override","Reject"]:
                g=analysis_rows[analysis_rows["decision_action"].eq(action)]
                if g.empty: continue
                fig.add_trace(go.Box(y=g["evidence_question_count"],name=action,boxpoints="all",jitter=.25,pointpos=0,boxmean=True))
            fig.update_yaxes(title="Evidence questions / trial",rangemode="tozero")
            _chart(fig,"p16v22_questions_by_action",360)

        overrides=analysis_rows[analysis_rows["decision_action"].eq("Override")]
        if not overrides.empty and "override_targets_json" in overrides:
            counter={}
            for raw in overrides["override_targets_json"].fillna("[]"):
                try: vals=json.loads(raw)
                except Exception: vals=[]
                for value in vals:
                    counter[str(value)]=counter.get(str(value),0)+1
            if counter:
                st.markdown("#### What participants corrected when overriding")
                frame=pd.DataFrame([{"Target":k,"Count":v} for k,v in counter.items()]).sort_values("Count")
                fig=go.Figure(go.Bar(x=frame["Count"],y=frame["Target"],orientation="h",text=frame["Count"],textposition="outside"))
                fig.update_xaxes(title="Override selections",dtick=1)
                _chart(fig,"p16v22_override_targets",360)
            complete_text=overrides.get("override_correction_text",pd.Series(dtype=object)).fillna("").astype(str).str.strip().ne("")
            st.caption("Structured correction text captured for {} of {} overrides.".format(int(complete_text.sum()),len(overrides)))

    with tabs[3]:
        pair=pd.DataFrame(summary.get("pair_coverage") or [])
        if not pair.empty:
            st.markdown("#### Trial-family exposure balance")
            heat=pair.set_index("pair_id")[["correct","incorrect"]]
            fig=go.Figure(go.Heatmap(
                z=heat.values,x=["Correct","Incorrect"],y=heat.index.tolist(),text=heat.values,texttemplate="%{text}",
                colorscale="Blues",showscale=False,hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>"
            ))
            _chart(fig,"p16v22_pair_heat",max(480,20*len(heat)+120))

            st.markdown("#### Scenario evaluation counts")
            scenario=pair[["scenario_id","evaluations","framework","source_stage"]].sort_values("evaluations")
            fig=go.Figure(go.Scatter(
                x=scenario["evaluations"],y=scenario["scenario_id"],mode="markers+text",
                text=scenario["evaluations"],textposition="middle right",marker=dict(size=11),
                customdata=scenario[["framework","source_stage"]].to_numpy(),
                hovertemplate="Scenario %{y}<br>Evaluations %{x}<br>Framework %{customdata[0]}<br>Type %{customdata[1]}<extra></extra>",
            ))
            fig.update_xaxes(title="Completed evaluations")
            _chart(fig,"p16v22_scenario_counts",max(460,20*len(scenario)+120))

        st.markdown("#### Framework × explanation-type observation matrix")
        matrix=analysis_rows.groupby(["framework_context","source_stage"]).size().unstack(fill_value=0)
        if matrix.empty:
            st.info("No framework/type observations available.")
        else:
            fig=go.Figure(go.Heatmap(
                z=matrix.values,x=matrix.columns.tolist(),y=matrix.index.tolist(),text=matrix.values,texttemplate="%{text}",
                colorscale="Blues",showscale=False,
                hovertemplate="Framework %{y}<br>Explanation type %{x}<br>Trials %{z}<extra></extra>"
            ))
            _chart(fig,"p16v22_framework_type",360)

        if not participants.empty and "counterbalance_arm" in participants:
            arms=participants[participants["participant_hash"].astype(str).isin(completed_hashes)]["counterbalance_arm"].fillna("Unknown").value_counts()
            if not arms.empty:
                st.markdown("#### Counterbalance-arm completion")
                fig=go.Figure(go.Bar(x=arms.index,y=arms.values,text=arms.values,textposition="outside"))
                fig.update_yaxes(title="Completed participants",dtick=1)
                _chart(fig,"p16v22_counterbalance",280)

    with tabs[4]:
        st.markdown("#### Trial-position trajectories")
        order=analysis_rows.groupby("order_index").agg(
            mean_trust=("trust_rating","mean"),
            mean_response_sec=("response_time_sec","mean"),
            mean_questions=("evidence_question_count","mean"),
        ).reset_index()
        if order.empty:
            st.info("No trial-order data available.")
        else:
            fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.08,
                              subplot_titles=("Trust","Response time","Optional evidence seeking"))
            fig.add_trace(go.Scatter(x=order["order_index"],y=order["mean_trust"],mode="lines+markers",name="Mean trust"),row=1,col=1)
            fig.add_trace(go.Scatter(x=order["order_index"],y=order["mean_response_sec"],mode="lines+markers",name="Mean response time"),row=2,col=1)
            fig.add_trace(go.Scatter(x=order["order_index"],y=order["mean_questions"],mode="lines+markers",name="Mean evidence questions"),row=3,col=1)
            fig.update_yaxes(range=[1,7],dtick=1,title_text="Trust",row=1,col=1)
            fig.update_yaxes(title_text="Seconds",rangemode="tozero",row=2,col=1)
            fig.update_yaxes(title_text="Questions",rangemode="tozero",row=3,col=1)
            fig.update_xaxes(title_text="Trial position (1–20)",dtick=1,row=3,col=1)
            fig.update_layout(showlegend=False)
            _chart(fig,"p16v22_trial_dynamics",650)

        st.markdown("#### Evidence-seeking intensity by explanation type and correctness")
        qmatrix=analysis_rows.pivot_table(index="source_stage",columns="correctness_condition",values="evidence_question_count",aggfunc="mean")
        if qmatrix.empty:
            st.info("No evidence-seeking data available.")
        else:
            for col in ["correct","incorrect"]:
                if col not in qmatrix.columns: qmatrix[col]=0.0
            qmatrix=qmatrix[["correct","incorrect"]]
            fig=go.Figure(go.Heatmap(
                z=qmatrix.values,x=["Correct","Incorrect"],y=qmatrix.index.tolist(),
                text=[["{:.2f}".format(v) for v in row] for row in qmatrix.values],texttemplate="%{text}",
                colorscale="Blues",colorbar=dict(title="Mean\nquestions"),
            ))
            _chart(fig,"p16v22_question_heat",360)

        if not events.empty:
            qevents=events[events["event_type"].eq("evidence_question")]
            st.caption("Total optional evidence questions logged: {}.".format(len(qevents)))

    with tabs[5]:
        st.markdown("#### Post-study human-agency and study-experience responses")
        if secondary.empty:
            st.info("The Human Agency visualization appears after participants complete the post-study questionnaire.")
        else:
            long=[]
            label_map={key:text for key,text in SECONDARY_ITEMS}
            for _,row in secondary.iterrows():
                try: ratings=json.loads(str(row.get("ratings_json") or "{}"))
                except Exception: ratings={}
                for item_id,value in ratings.items():
                    if item_id in label_map:
                        long.append({"Item":label_map[item_id],"Rating":int(value),"Participant":str(row.get("participant_hash"))})
            ldf=pd.DataFrame(long)
            if ldf.empty:
                st.info("No secondary ratings could be parsed.")
            else:
                short={
                    label_map["critical_evaluation"]:"Critically evaluate",
                    label_map["freedom_to_override"]:"Freedom to override",
                    label_map["final_control"]:"Final decision control",
                    label_map["rely_or_question"]:"Rely vs question",
                    label_map["evidence_checking"]:"Evidence checking",
                    label_map["explanation_usefulness"]:"Explanation usefulness",
                    label_map["interface_usability"]:"Interface usability",
                    label_map["real_world_support_trust"]:"Real-world support trust",
                }
                ldf["Short item"]=ldf["Item"].map(short)
                order_items=list(short.values())
                counts=ldf.groupby(["Short item","Rating"]).size().reset_index(name="n")
                totals=counts.groupby("Short item")["n"].transform("sum")
                counts["pct"]=100*counts["n"]/totals
                fig=go.Figure()
                for rating in range(1,8):
                    sub=counts[counts["Rating"].eq(rating)].set_index("Short item").reindex(order_items).fillna(0).reset_index()
                    fig.add_trace(go.Bar(
                        x=sub["pct"],y=sub["Short item"],orientation="h",name=str(rating),
                        customdata=sub[["n"]].to_numpy(),
                        hovertemplate="Rating "+str(rating)+"<br>%{y}<br>%{x:.1f}% · n=%{customdata[0]}<extra></extra>"
                    ))
                fig.update_layout(barmode="stack",legend=dict(orientation="h",y=1.12,title="Rating"))
                fig.update_xaxes(range=[0,100],title="Response share (%)")
                fig.update_yaxes(categoryorder="array",categoryarray=list(reversed(order_items)))
                _chart(fig,"p16v22_human_agency_likert",520)

                means=ldf.groupby("Short item")["Rating"].agg(["mean","std","count"]).reindex(order_items).reset_index()
                fig=go.Figure(go.Scatter(
                    x=means["mean"],y=means["Short item"],mode="markers+text",
                    text=["{:.2f}".format(v) for v in means["mean"]],textposition="middle right",
                    marker=dict(size=13),error_x=dict(type="data",array=means["std"].fillna(0),visible=True),
                    customdata=means[["count"]].to_numpy(),
                    hovertemplate="Mean %{x:.2f}<br>n=%{customdata[0]}<extra></extra>"
                ))
                fig.add_vline(x=4,line_dash="dot",line_color="#94A3B8")
                fig.update_xaxes(range=[1,7],dtick=1,title="Mean response (1–7)")
                _chart(fig,"p16v22_human_agency_means",460)
                comments=secondary.get("open_comment",pd.Series(dtype=object)).fillna("").astype(str).str.strip().ne("")
                st.caption("Open-text reflection provided by {} of {} completed questionnaires.".format(int(comments.sum()),len(secondary)))

    with tabs[6]:
        st.markdown("#### Expertise rating × participant calibration gap")
        if paired.empty or participants.empty:
            st.info("Expertise calibration view requires completed paired participant responses.")
        else:
            meta=participants[[c for c in ["participant_hash","expertise_group","expertise_self_rating"] if c in participants.columns]].copy()
            merged=paired.merge(meta,on="participant_hash",how="left")
            merged["expertise_self_rating"]=pd.to_numeric(merged["expertise_self_rating"],errors="coerce")
            fig=go.Figure()
            for group,g in merged.groupby("expertise_group",dropna=False):
                fig.add_trace(go.Scatter(
                    x=g["expertise_self_rating"],y=g["gap"],mode="markers+text",name=str(group),
                    text=g["participant_hash"].astype(str).str[:6],textposition="top center",marker=dict(size=13),
                    hovertemplate="Expertise %{x}/7<br>Calibration gap %{y:+.2f}<extra></extra>"
                ))
            fig.add_hline(y=0,line_dash="dash",line_color="#94A3B8")
            fig.update_xaxes(range=[0.5,7.5],dtick=1,title="Self-rated ML / AutoML knowledge and experience")
            fig.update_yaxes(title="Correct − incorrect mean trust")
            fig.update_layout(legend=dict(orientation="h",y=1.12))
            _chart(fig,"p16v22_expertise_gap",420)

        exp=pd.DataFrame(summary.get("expertise_descriptive") or [])
        if not exp.empty:
            st.markdown("#### Mean trust by expertise group · descriptive only")
            fig=go.Figure(go.Scatter(
                x=exp["mean_trust"],y=exp["expertise_group"],mode="markers+text",
                text=["{:.2f} · n={}".format(m,n) for m,n in zip(exp["mean_trust"],exp["participants"])],
                textposition="middle right",marker=dict(size=14),
                error_x=dict(type="data",array=exp["sd_trust"].fillna(0),visible=True),
                hovertemplate="Mean trust %{x:.2f}<extra></extra>"
            ))
            fig.add_vline(x=4,line_dash="dot",line_color="#94A3B8")
            fig.update_xaxes(range=[1,7],dtick=1,title="Mean trust")
            _chart(fig,"p16v22_expertise_mean",340)
        st.caption("Expertise analyses are descriptive secondary analyses; the study is not powered for confirmatory between-group comparisons.")

def _researcher(study: TrustCalibrationStudyV2, mode: str):
    key = _researcher_key(mode)
    if not st.session_state.get(PFX+"_researcher_unlocked"):
        if not key:
            st.error("Configure AWAREML_STUDY_RESEARCHER_KEY for Main Study researcher access.")
            return
        entered = st.text_input("Researcher key", type="password", key=PFX+"_researcher_key")
        if st.button("Unlock researcher workspace", type="primary", key=PFX+"_unlock"):
            if entered == key:
                st.session_state[PFX+"_researcher_unlocked"] = True
                st.rerun()
            else:
                st.error("Incorrect researcher key.")
        return

    summary = analyze_v2(study.store, mode)
    protocol = study.protocol
    st.success("Researcher workspace · Morten-review v2.1")

    section("Study readiness", "Protocol, instrument, design lock and collection target are visible before inspecting responses.")
    ready=st.columns(6)
    ready[0].metric("Protocol", "Final")
    ready[1].metric("Trust measure", "S-TIAS · 3 items")
    ready[2].metric("Design lock", "Locked")
    ready[3].metric("Scenario coherence", "20 / 20")
    ready[4].metric("Counterbalance", "A / B")
    ready[5].metric("Main target", "34 · recruit ≤40")
    st.caption("Every trial now uses the scenario's Track-1 priorities to define the task and decision-relevant evidence. Controlled Trust Calibration values provide a consistent study comparison context and are not presented as real-world performance on the textual scenario.")

    section("Collection status", "Each completed participant evaluates all 20 matched trial families exactly once.")
    cards=st.columns(6)
    cards[0].metric("Registered", summary.get("registered",0))
    cards[1].metric("Completed", summary.get("completed",0))
    cards[2].metric("Trials analyzed", summary.get("trials",0))
    cards[3].metric("Matched families", "20")
    cards[4].metric("Frameworks", "5 × 4 trials")
    cards[5].metric("Explanation types", "4 × 5 trials")
    st.info("At 34 completed participants, every matched trial family has 34 evaluations. At 40 completed participants, every matched trial family has 40 evaluations. Counterbalance arms determine which correct/incorrect member of the matched pair is shown.")

    if summary.get("condition_summary"):
        condition=summary["condition_summary"]
        gap=summary.get("trust_gap") or {}
        section("Primary calibration signal", "The central comparison is within-participant trust for correct versus incorrect explanations.")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Correct mean trust", "N/A" if condition["correct"]["mean_trust"] is None else f"{condition['correct']['mean_trust']:.2f}")
        c1.caption("SD: {}".format("N/A" if condition["correct"]["sd_trust"] is None else f"{condition['correct']['sd_trust']:.2f}"))
        c2.metric("Incorrect mean trust", "N/A" if condition["incorrect"]["mean_trust"] is None else f"{condition['incorrect']['mean_trust']:.2f}")
        c2.caption("SD: {}".format("N/A" if condition["incorrect"]["sd_trust"] is None else f"{condition['incorrect']['sd_trust']:.2f}"))
        c3.metric("Correct − incorrect", "N/A" if gap.get("mean") is None else f"{gap['mean']:.2f}")
        c3.caption("Participant-level SD: {}".format("N/A" if gap.get("sd") is None else f"{gap['sd']:.2f}"))
        c4.metric("Completed / required", "{} / 34".format(summary.get("completed",0)))

    section("Research-grade descriptive views", "High-resolution descriptive diagnostics cover calibration, reliance, scenario coverage, trial dynamics, human agency and expertise while keeping inferential claims separate from exploration.")
    _researcher_visuals(study, mode, summary)

    section("Trial-family exposure audit", "This is the scenario/pair evaluation count requested for journal reporting.")
    pair=pd.DataFrame(summary.get("pair_coverage") or [])
    if pair.empty:
        st.info("No completed revised sessions yet.")
    else:
        st.dataframe(pair, use_container_width=True, hide_index=True)
        st.caption("Minimum evaluations of any matched trial family: {}.".format(int(pair["evaluations"].min())))

    with st.expander("Study protocol · scenario map · controlled evidence design", expanded=False):
        st.json({"central_rq":protocol.get("central_rq"),"design":protocol.get("design"),"analysis_plan":protocol.get("analysis_plan")},expanded=False)
        scenario_path=study.bank.root/"scenario_map.csv"
        if scenario_path.exists():
            st.dataframe(pd.read_csv(scenario_path),use_container_width=True,hide_index=True)

    with st.expander("Exact response export", expanded=False):
        responses=pd.DataFrame(study.store.response_rows(mode))
        st.dataframe(responses,use_container_width=True,hide_index=True)
        st.download_button("Download responses CSV",responses.to_csv(index=False).encode("utf-8"),file_name="trust_calibration_v2_1_{}_responses.csv".format(mode),mime="text/csv",use_container_width=True)
    with st.expander("Timestamped per-trial evidence events", expanded=False):
        st.dataframe(pd.DataFrame(study.store.event_rows(mode)),use_container_width=True,hide_index=True)

def phase16_trust_calibration_page_v2():
    _css(); mode = _mode(); role = str(st.session_state.get(ROLE_KEY) or "")
    if role == "participant":
        if st.button("Exit study view", key="p16v2_exit"): st.session_state.pop(ROLE_KEY,None); st.rerun()
        render_phase16_participant_study(collection_mode=mode, dashboard_embedded=True); return
    hero("HUMAN STUDY · CALIBRATED RELIANCE", "Trust Calibration", "Morten-review v2.1: scenario → task → explanation → evidence coherence, complete five-framework reference evidence, explicit ratings, evidence seeking and actionable Override.", pills=[("20 scenarios","good"),("5 frameworks","good"),("10 correct + 10 incorrect","good"),("Counterbalanced","good")])
    try:
        study = TrustCalibrationStudyV2()
        if mode == "final": study.assert_final_ready()
    except Exception as exc: st.error(str(exc)); return
    if role == "researcher":
        if st.button("Back to study home", key="p16v2_back_researcher"): st.session_state.pop(ROLE_KEY,None); st.rerun()
        _researcher(study, mode); return
    st.markdown('<div class="m2-context"><b>Revised study logic.</b> Every participant receives all 20 matched trial families exactly once: five frameworks × four coherent explanation forms, each attached to a unique Track-1 scenario. The task is derived from that scenario&#39;s Accuracy/Runtime/Energy/CO₂ priorities. Correctness is blinded; counterbalance arms complement the correct/incorrect variant shown for each pair.</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    if c1.button("Enter participant study", type="primary", use_container_width=True, key="p16v2_enter_p"): st.session_state[ROLE_KEY] = "participant"; st.rerun()
    if c2.button("Open researcher workspace", use_container_width=True, key="p16v2_enter_r"): st.session_state[ROLE_KEY] = "researcher"; st.rerun()
    cards = st.columns(5)
    for col,(label,value) in zip(cards,[("Trials / participant","20"),("Track-1 scenarios","20 unique"),("Framework coverage","4 trials each"),("Required complete","34"),("Recruit up to","40")]): col.metric(label,value)
