from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from awareml.studies.study_runtime import bootstrap_study_runtime
bootstrap_study_runtime()

from awareml.studies.store import StudyStore
from awareml.studies.information_seeking import CATEGORY_LABELS, classify_follow_up
from awareml.studies.information_seeking_grounding import grounded_information_answer
from awareml.studies.information_seeking_morten_v2 import (
    analyze,
    coding_scheme,
    context_path,
    interaction_sequence,
    load_context,
    manual_coding,
    protocol,
    save_context,
    scenario_catalog,
    session_summaries,
)
from .components import empty_state, hero, section
from .state import ensure_research_state, result_dicts


# Compatibility markers retained for the v7.1 focused validation while v7.2 renders richer replacements.
LEGACY_INFORMATION_RESEARCHER_MARKERS = (
    "Behavior overview",
    "Evidence use",
    "Decision & timing",
    "Sequences",
    "Clusters",
    "Qualitative coding",
)

PFX = "p17v2"
LOCAL_KEY = "phase16-local-test-key"
MANIFEST = Path("data/journal/information_seeking_v2/frozen/design_manifest.json")


def _state():
    return ensure_research_state()


def _mode() -> str:
    requested = str(os.getenv("AWAREML_INFORMATION_SEEKING_COLLECTION_MODE", "pilot")).strip().lower()
    ready = MANIFEST.exists() and str(os.getenv("AWAREML_INFORMATION_SEEKING_FINAL_ARMED", "")).strip().upper() == "YES"
    return "final" if requested == "final" and ready else "pilot"


def _study_name(mode: str) -> str:
    return "information_seeking_v2_{}".format(mode)


def _researcher_key(mode: str) -> Optional[str]:
    explicit = str(os.getenv("AWAREML_STUDY_RESEARCHER_KEY", "")).strip()
    if explicit:
        return explicit
    return LOCAL_KEY if mode == "pilot" else None


def _top(ranking: List[Dict[str, Any]]) -> Optional[dict]:
    rows = sorted(
        list(ranking or []),
        key=lambda r: (
            float(r.get("rank", 999)),
            -float(r.get("utility", 0) or 0),
        ),
    )
    return rows[0] if rows else None


def _run(results: List[dict], framework: str) -> dict:
    return next((row for row in results if str(row.get("framework")) == str(framework)), {})


def _chart(fig: go.Figure, key: str, height: int = 360) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=25, t=40, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color="#0F172A"),
        hoverlabel=dict(bgcolor="#FFFFFF"),
    )
    fig.update_xaxes(gridcolor="#E2E8F0", zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False, automargin=True)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "displaylogo": False},
        key=key,
    )


def _scenario_card(scenario: dict) -> None:
    objectives = str(scenario.get("ground_truth") or "N/A").replace("|", " + ")
    st.markdown(
        '<div style="padding:18px 20px;border:1px solid rgba(37,99,235,.18);border-left:4px solid #2563eb;'
        'border-radius:14px;background:rgba(239,246,255,.72);margin:8px 0 16px">'
        '<div style="font-size:.75rem;font-weight:800;letter-spacing:.07em;color:#2563eb">SCENARIO CONTEXT · {}</div>'
        '<div style="font-size:1.05rem;font-weight:700;margin:.45rem 0">{}</div>'
        '<div style="color:#64748b">Domain: {} · Scenario priorities: <b>{}</b> · Source: Track-1 scenario benchmark</div>'
        '</div>'.format(
            scenario.get("benchmark_id", "N/A"),
            scenario.get("scenario", "N/A"),
            scenario.get("domain") or "N/A",
            objectives,
        ),
        unsafe_allow_html=True,
    )

def _evidence_contract(ctx: dict) -> None:
    scenario = ctx.get("scenario") or {}
    priorities = str(scenario.get("ground_truth") or "N/A").replace("|", " + ")
    attested = bool(ctx.get("scenario_alignment_attested"))
    tone = "#047857" if attested else "#B45309"
    st.markdown(
        '<div style="padding:14px 17px;border-left:4px solid {tone};background:rgba(248,250,252,.9);border-radius:12px;margin:.6rem 0 1rem">'
        '<b>Scenario → measured evidence contract.</b> The scenario supplies the decision priorities (<b>{priorities}</b>). '
        'The saved AwareML context supplies measured benchmark evidence from the standardized five-framework run. '
        'The study does <b>not</b> claim that the benchmark dataset is literally the textual scenario.<br>'
        '<span style="color:#64748b">Researcher alignment attestation: <b>{status}</b>.</span>'
        '</div>'.format(
            tone=tone,
            priorities=priorities,
            status="confirmed" if attested else "not yet confirmed",
        ),
        unsafe_allow_html=True,
    )

def _ranking_frame(results: List[dict], ranking: List[dict]) -> pd.DataFrame:
    rows = []
    for rr in sorted(ranking, key=lambda r: float(r.get("rank", 999))):
        fw = str(rr.get("framework") or rr.get("Framework") or "")
        run = _run(results, fw)
        rows.append({
            "Rank": rr.get("rank"),
            "Framework": fw,
            "Utility": rr.get("utility"),
            "Accuracy": run.get("accuracy", rr.get("accuracy")),
            "Runtime (s)": run.get("runtime_sec", rr.get("runtime_sec")),
            "Energy (kWh)": run.get("energy_kwh", rr.get("energy_kwh")),
            "CO₂ (kg)": run.get("co2_kg", rr.get("co2_kg")),
        })
    return pd.DataFrame(rows)


def _ranking_table(results: List[dict], ranking: List[dict]) -> None:
    frame = _ranking_frame(results, ranking)
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _evidence_table(area: str, results: List[dict], ranking: List[dict]) -> None:
    rows = []
    for rr in sorted(ranking, key=lambda r: float(r.get("rank", 999))):
        fw = str(rr.get("framework") or "")
        run = _run(results, fw)
        if area == "performance":
            rows.append({
                "Framework": fw,
                "Rank": rr.get("rank"),
                "Accuracy": run.get("accuracy"),
                "Macro-F1": run.get("f1_macro"),
                "Runtime (s)": run.get("runtime_sec"),
            })
        elif area == "sustainability":
            rows.append({
                "Framework": fw,
                "Runtime (s)": run.get("runtime_sec"),
                "Energy (kWh)": run.get("energy_kwh"),
                "CO₂ (kg)": run.get("co2_kg"),
            })
        elif area == "fairness":
            fairness = run.get("fairness") or {}
            rows.append({
                "Framework": fw,
                "DP gap": fairness.get("dp_diff"),
                "EO gap": fairness.get("equal_opportunity_diff"),
                "EOdds gap": fairness.get("equalized_odds_gap"),
                "Brier gap": fairness.get("group_brier_score_gap"),
                "ECE gap": fairness.get("group_ece_gap"),
            })
        elif area == "drift":
            drift = run.get("drift_summary") or {}
            rows.append({
                "Framework": fw,
                "Drift events": len(run.get("drift_events") or []),
                "Recovery rate": drift.get("recovery_rate"),
                "Median recovery samples": drift.get("median_recovery_samples"),
                "Mean accuracy drop": drift.get("mean_accuracy_drop"),
            })
        else:
            xai = run.get("explainability") or {}
            rows.append({
                "Framework": fw,
                "Method": xai.get("method"),
                "Status": xai.get("status"),
                "Fidelity": xai.get("fidelity"),
                "Stability": xai.get("stability"),
                "Consistency": xai.get("consistency"),
                "Top feature": xai.get("top_feature"),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _reset_information_participant_session() -> None:
    current_view = st.session_state.get(PFX + "_view", "participant")
    for key in list(st.session_state.keys()):
        if str(key).startswith(PFX) and key != PFX + "_view":
            del st.session_state[key]
    st.session_state[PFX + "_view"] = current_view or "participant"


def _registration(mode: str, ctx: dict) -> bool:
    if st.session_state.get(PFX + "_session_code"):
        return True
    p = protocol()
    section(
        "Before you begin",
        "Use the anonymous/session code supplied by the study team. If this follows Trust Calibration, use the same code.",
    )
    st.info((p.get("participant_materials") or {}).get("estimated_duration", ""))
    with st.expander("Study information and consent", expanded=False):
        st.write((p.get("participant_materials") or {}).get("consent_text", ""))
    groups = (p.get("eligibility") or {}).get("groups") or [
        "University / CS student", "Practitioner", "ML / AutoML expert"
    ]
    with st.form("p17v2_registration"):
        code = st.text_input("Participant / session code", key=PFX+"_reg_code")
        group = st.selectbox("Experience group", groups, index=None, placeholder="Select your group", key=PFX+"_reg_group")
        st.markdown("**Self-rated Machine Learning / AutoML knowledge and experience**")
        expertise = st.radio(
            "ML/AutoML knowledge and experience",
            [1, 2, 3, 4, 5, 6, 7],
            index=None,
            horizontal=True,
            label_visibility="collapsed",
            key=PFX+"_reg_expertise",
        )
        st.caption("1 = No knowledge / experience · 4 = Moderate knowledge / experience · 7 = Extensive knowledge / experience")
        ml_ready = st.checkbox("I confirm that I have enough ML / AutoML familiarity to compare framework evidence in this activity.", key=PFX+"_reg_ml_ready")
        adult = st.checkbox("I confirm that I am at least 18 years old.", key=PFX+"_reg_adult")
        consent = st.checkbox("I voluntarily consent to participate in the Information-Seeking activity.", key=PFX+"_reg_consent")
        data = st.checkbox("I consent to the processing of my study responses as described in the study information.", key=PFX+"_reg_data")
        start = st.form_submit_button("Start information-seeking study", type="primary", use_container_width=True)
    if start:
        if not all([code.strip(), group, expertise, ml_ready, adult, consent, data]):
            st.error("Complete all registration, expertise, eligibility and consent fields.")
            return False
        st.session_state[PFX + "_session_code"] = code.strip()
        st.session_state[PFX + "_group"] = group
        st.session_state[PFX + "_expertise"] = int(expertise)
        st.session_state[PFX + "_started_epoch"] = time.time()
        st.session_state[PFX + "_chat"] = []
        st.session_state[PFX + "_evidence_events"] = []
        StudyStore().log(
            _study_name(mode),
            code.strip(),
            "session_start",
            {
                "experience_group": group,
                "expertise_rating": int(expertise),
                "scenario_id": (ctx.get("scenario") or {}).get("benchmark_id"),
                "context_id": ctx.get("context_id"),
            },
        )
        st.rerun()
    return False


def _log_view(mode: str, area: str) -> None:
    code = st.session_state[PFX + "_session_code"]
    started = float(st.session_state.get(PFX + "_started_epoch") or time.time())
    events = st.session_state.setdefault(PFX + "_evidence_events", [])
    events.append(area)
    StudyStore().log(
        _study_name(mode),
        code,
        "evidence_view",
        {
            "evidence_area": area,
            "view_index": len(events),
            "area_visit_number": events.count(area),
            "time_since_session_start_sec": max(0, time.time() - started),
        },
    )


def _rank_of(framework: str, ranking: List[dict]) -> Optional[int]:
    for row in ranking:
        if str(row.get("framework") or row.get("Framework") or "") == str(framework):
            try:
                return int(row.get("rank"))
            except Exception:
                return None
    return None


def _ask(mode: str, question: str, results: List[dict], ranking: List[dict]) -> None:
    q = str(question or "").strip()
    if not q:
        return
    category = classify_follow_up(q)
    answer, topics, targets = grounded_information_answer(q, category, results, ranking)
    ql = q.lower()
    if any(token in ql for token in ["rank", "ranking", "position"]):
        target = targets[0] if targets else None
        if target:
            rank = _rank_of(target, ranking)
            if rank is not None and "rank #" not in answer.lower():
                answer = "**Rank — {}:** #{} in the standardized observed ranking.\n\n{}".format(target, rank, answer)
    started = float(st.session_state.get(PFX + "_started_epoch") or time.time())
    StudyStore().log(
        _study_name(mode),
        st.session_state[PFX + "_session_code"],
        "chat_turn",
        {
            "question": q,
            "answer": answer,
            "category": category,
            "requested_topics": topics,
            "target_frameworks": targets,
            "time_since_session_start_sec": max(0, time.time() - started),
        },
    )
    st.session_state.setdefault(PFX + "_chat", []).append({"q": q, "a": answer})
    st.rerun()


def _participant(mode: str, ctx: dict) -> None:
    if not _registration(mode, ctx):
        return
    if st.session_state.get(PFX + "_completed"):
        st.success("Information-Seeking session complete. Thank you.")
        start_label = "Start new Main Study" if mode == "final" else "Start new Pilot Study"
        if st.button(start_label, type="primary", use_container_width=True, key=PFX+"_start_new_session"):
            _reset_information_participant_session()
            st.rerun()
        return

    scenario = ctx.get("scenario") or {}
    results = list(ctx.get("results") or [])
    ranking = list(ctx.get("ranking") or [])
    top = _top(ranking)
    if top is None:
        empty_state("Observed ranking required", "The standardized context does not contain an observed ranking.")
        return

    _scenario_card(scenario)
    _evidence_contract(ctx)

    section(
        "Initial AwareML recommendation",
        "The same standardized scenario and measured five-framework evidence context is used across Main Study participants for comparability.",
    )
    framework = str(top.get("framework") or "N/A")
    st.markdown("### {}".format(framework))
    st.write("{} is ranked first in the standardized observed ranking with utility {}.".format(framework, top.get("utility")))
    with st.expander("See the five-framework ranking context", expanded=True):
        _ranking_table(results, ranking)

    section(
        "Explore evidence",
        "Opening and reopening evidence areas is logged. Repeated access is part of the behavioral outcome.",
    )
    areas = [
        ("performance", "Performance"),
        ("fairness", "Fairness"),
        ("sustainability", "Sustainability"),
        ("drift", "Drift"),
        ("explainability", "Explainability"),
    ]
    cols = st.columns(5)
    for col, (key, label) in zip(cols, areas):
        if col.button("View " + label, key=PFX + "_ev_" + key, use_container_width=True):
            _log_view(mode, key)
            st.session_state[PFX + "_active_evidence"] = key
            st.rerun()
    active = st.session_state.get(PFX + "_active_evidence")
    if active:
        st.markdown("#### {} evidence".format(dict(areas)[active]))
        _evidence_table(active, results, ranking)

    section(
        "Ask for more information",
        "Ask only what you genuinely need. Answers are deterministic and grounded in the standardized measured evidence.",
    )
    quick = [
        ("Show evidence", "Show me the measured evidence supporting the top recommendation."),
        ("Why this choice?", "Why is this framework ranked first?"),
        ("Compare top 2", "Compare the top two ranked frameworks using the measured evidence."),
        ("Challenge", "Are you sure this recommendation is the best choice? What evidence could challenge it?"),
        ("Clarify", "Explain the recommendation in simpler terms and define the most important metric."),
    ]
    qcols = st.columns(5)
    for col, (label, q) in zip(qcols, quick):
        if col.button(label, key=PFX + "_q_" + label, use_container_width=True):
            _ask(mode, q, results, ranking)
    with st.form(PFX + "_freeq", clear_on_submit=True):
        q = st.text_input(
            "Ask your own question",
            placeholder="For example: What is AutoClass's rank and accuracy? Compare the top two on energy.",
        )
        ask = st.form_submit_button("Ask")
    if ask and q.strip():
        _ask(mode, q, results, ranking)
    for item in st.session_state.get(PFX + "_chat", []):
        st.markdown("**You:** " + item["q"])
        st.info(item["a"])

    section(
        "Your final decision",
        "Finish when you feel you have enough information. Zero follow-up questions is valid.",
    )
    decision = st.radio(
        "What would you do with the current recommendation?",
        ["Accept", "Override", "Reject"],
        index=None,
        key=PFX + "_decision",
        format_func=lambda x: {
            "Accept": "Accept — I would rely on it as shown",
            "Override": "Override — I would change/correct it before acting",
            "Reject": "Reject — I would not rely on it",
        }[x],
    )
    targets: List[str] = []
    correction = ""
    if decision == "Override":
        targets = st.multiselect(
            "What would you change?",
            ["Framework / ranking", "Accuracy", "Runtime", "Energy", "CO₂", "Fairness", "Explainability", "Other"],
            key=PFX + "_targets",
        )
        correction = st.text_area(
            "Briefly state what you would change before acting.",
            height=80,
            key=PFX + "_correction",
        )
    confidence = st.radio(
        "How confident are you in this final decision?",
        [1, 2, 3, 4, 5, 6, 7],
        index=None,
        horizontal=True,
        key=PFX + "_confidence",
    )
    st.caption("1 = Not at all confident · 7 = Extremely confident")
    rationale = st.text_area(
        "What made you ask for more information — or decide not to ask?",
        height=90,
        key=PFX + "_rationale",
    )
    if st.button("Finish and save study session", type="primary", use_container_width=True, key=PFX + "_finish"):
        if decision is None or confidence is None:
            st.error("Choose a final action and confidence rating.")
            return
        if decision == "Override" and (not targets or not correction.strip()):
            st.error("For Override, identify what you would change and state the correction.")
            return
        started = float(st.session_state.get(PFX + "_started_epoch") or time.time())
        views = list(st.session_state.get(PFX + "_evidence_events", []))
        StudyStore().log(
            _study_name(mode),
            st.session_state[PFX + "_session_code"],
            "session_end",
            {
                "recommendation_decision": decision.lower(),
                "final_confidence": int(confidence),
                "participant_rationale": rationale or None,
                "override_targets": targets,
                "override_correction": correction or None,
                "total_time_to_final_decision_sec": max(0, time.time() - started),
                "total_evidence_views": len(views),
                "unique_evidence_areas": list(dict.fromkeys(views)),
                "repeat_evidence_access_count": len(views) - len(set(views)),
                "follow_up_count": len(st.session_state.get(PFX + "_chat", [])),
            },
        )
        st.session_state[PFX + "_completed"] = True
        st.rerun()


def _category_label(key: str) -> str:
    return CATEGORY_LABELS.get(str(key), str(key).replace("_", " ").title())


def _researcher_visuals(events: pd.DataFrame, summaries: pd.DataFrame, analysis: dict) -> None:
    completed = summaries[summaries["completed"] == True].copy() if not summaries.empty else summaries
    n_completed = int(len(completed)) if completed is not None else 0

    if n_completed < 4:
        st.warning(
            "Preview only: {} completed Information-Seeking session{}. The workspace is ready, but patterns, clusters and subgroup differences should not yet be interpreted as study findings.".format(
                n_completed, "" if n_completed == 1 else "s"
            )
        )
    elif n_completed < 12:
        st.info("Interim descriptive view: {} of the planned 12 completed sessions are available.".format(n_completed))
    else:
        st.success("Planned Information-Seeking target reached: {} completed sessions.".format(n_completed))

    tabs = st.tabs([
        "Overview",
        "Evidence navigation",
        "Question behavior",
        "Decision journey",
        "Interaction sequences",
        "Clusters & expertise",
        "Qualitative coding",
    ])

    with tabs[0]:
        st.markdown("#### Information-seeking funnel")
        if completed.empty:
            st.info("Completed sessions are required for the behavioral funnel.")
        else:
            total=len(completed)
            evidence_users=int((pd.to_numeric(completed["evidence_view_count"],errors="coerce").fillna(0)>0).sum())
            follow_users=int((pd.to_numeric(completed["follow_up_count"],errors="coerce").fillna(0)>0).sum())
            revisit_users=int((pd.to_numeric(completed["repeat_evidence_access_count"],errors="coerce").fillna(0)>0).sum())
            changed=int(completed["recommendation_decision"].isin(["override","reject"]).sum())
            fig=go.Figure(go.Funnel(
                y=["Completed session","Viewed evidence","Asked follow-up","Revisited evidence","Override / Reject"],
                x=[total,evidence_users,follow_users,revisit_users,changed],
                textinfo="value+percent initial",
            ))
            _chart(fig,"p17v22_funnel",420)

        st.markdown("#### Participant behavior map")
        if completed.empty:
            st.info("No completed participant behavior is available.")
        else:
            completed=completed.copy()
            completed["evidence_view_count"]=pd.to_numeric(completed["evidence_view_count"],errors="coerce").fillna(0)
            completed["follow_up_count"]=pd.to_numeric(completed["follow_up_count"],errors="coerce").fillna(0)
            completed["repeat_evidence_access_count"]=pd.to_numeric(completed["repeat_evidence_access_count"],errors="coerce").fillna(0)
            fig=go.Figure()
            for decision,group in completed.groupby("recommendation_decision",dropna=False):
                fig.add_trace(go.Scatter(
                    x=group["evidence_view_count"],y=group["follow_up_count"],mode="markers+text",
                    name=str(decision).title(),text=group["session_hash"].astype(str).str[:6],textposition="top center",
                    marker=dict(size=12+3*group["repeat_evidence_access_count"].clip(upper=6)),
                    customdata=group[["repeat_evidence_access_count","total_time_to_final_decision_sec","final_confidence"]].to_numpy(),
                    hovertemplate="Evidence views %{x}<br>Follow-ups %{y}<br>Repeat views %{customdata[0]}<br>Decision time %{customdata[1]:.1f}s<br>Confidence %{customdata[2]}/7<extra></extra>",
                ))
            fig.update_xaxes(title="Evidence views")
            fig.update_yaxes(title="Follow-up questions")
            fig.update_layout(legend=dict(orientation="h",y=1.12))
            _chart(fig,"p17v22_behavior_map",430)
            st.caption("Marker size increases with repeated evidence access.")

    with tabs[1]:
        views=analysis.get("evidence_view_counts") or {}
        revisits=analysis.get("evidence_revisit_counts") or {}
        areas=sorted(set(views)|set(revisits))
        st.markdown("#### Evidence access and revisiting")
        if not areas:
            st.info("No evidence views have been recorded.")
        else:
            frame=pd.DataFrame({
                "Area":[a.replace("_"," ").title() for a in areas],
                "Total views":[views.get(a,0) for a in areas],
                "Repeat views":[revisits.get(a,0) for a in areas],
            })
            fig=go.Figure()
            fig.add_trace(go.Bar(x=frame["Area"],y=frame["Total views"],name="Total views",text=frame["Total views"],textposition="outside"))
            fig.add_trace(go.Bar(x=frame["Area"],y=frame["Repeat views"],name="Repeat views",text=frame["Repeat views"],textposition="outside"))
            fig.update_layout(barmode="group",legend=dict(orientation="h",y=1.12))
            fig.update_yaxes(title="Logged accesses",rangemode="tozero")
            _chart(fig,"p17v22_evidence_access",400)

        if events is not None and not events.empty:
            evidence_events=events[events["event_type"].eq("evidence_view")].copy()
            if not evidence_events.empty:
                rows=[]
                transitions={}
                for session,group in evidence_events.groupby("session_hash"):
                    seq=[]
                    for _,event in group.sort_values("id").iterrows():
                        try: payload=json.loads(str(event.get("payload_json") or "{}"))
                        except Exception: payload={}
                        area=str(payload.get("evidence_area") or "unknown")
                        rows.append({"Session":str(session)[:8],"Area":area})
                        seq.append(area)
                    for a,b in zip(seq,seq[1:]):
                        transitions[(a,b)]=transitions.get((a,b),0)+1
                access=pd.DataFrame(rows)
                if not access.empty:
                    st.markdown("#### Participant × evidence-area access matrix")
                    mat=pd.crosstab(access["Session"],access["Area"])
                    fig=go.Figure(go.Heatmap(
                        z=mat.values,x=[c.replace("_"," ").title() for c in mat.columns],y=mat.index.tolist(),
                        text=mat.values,texttemplate="%{text}",colorscale="Blues",showscale=False,
                        hovertemplate="Session %{y}<br>%{x}: %{z} views<extra></extra>"
                    ))
                    _chart(fig,"p17v22_access_matrix",max(340,32*len(mat)+130))
                if transitions:
                    labels=sorted(set([a for a,_ in transitions]+[b for _,b in transitions]))
                    z=[[transitions.get((a,b),0) for b in labels] for a in labels]
                    st.markdown("#### Evidence-navigation transition matrix")
                    fig=go.Figure(go.Heatmap(
                        z=z,x=[x.replace("_"," ").title() for x in labels],y=[x.replace("_"," ").title() for x in labels],
                        text=z,texttemplate="%{text}",colorscale="Blues",showscale=False,
                        hovertemplate="From %{y}<br>to %{x}<br>Transitions %{z}<extra></extra>"
                    ))
                    fig.update_xaxes(title="Next evidence area")
                    fig.update_yaxes(title="Previous evidence area")
                    _chart(fig,"p17v22_transition_matrix",390)

    with tabs[2]:
        left,right=st.columns(2)
        with left:
            st.markdown("#### Follow-up categories")
            categories=analysis.get("category_counts") or {}
            if categories:
                frame=pd.DataFrame([{"Category":_category_label(k),"Count":v} for k,v in categories.items()]).sort_values("Count")
                fig=go.Figure(go.Bar(x=frame["Count"],y=frame["Category"],orientation="h",text=frame["Count"],textposition="outside"))
                _chart(fig,"p17v22_categories",360)
            else:
                st.info("No follow-up questions have been recorded.")
        with right:
            st.markdown("#### Requested evidence topics")
            topics=analysis.get("requested_topic_counts") or {}
            if topics:
                frame=pd.DataFrame([{"Topic":str(k).replace("_"," ").title(),"Count":v} for k,v in topics.items()]).sort_values("Count")
                fig=go.Figure(go.Bar(x=frame["Count"],y=frame["Topic"],orientation="h",text=frame["Count"],textposition="outside"))
                _chart(fig,"p17v22_topics",360)
            else:
                st.info("No requested topics have been recorded.")

        if not completed.empty:
            st.markdown("#### Time to first follow-up")
            first=pd.to_numeric(completed["time_before_first_follow_up_sec"],errors="coerce").dropna()
            if first.empty:
                st.info("No participant asked a follow-up question yet.")
            else:
                fig=go.Figure(go.Box(x=first,boxpoints="all",jitter=.25,pointpos=0,name="Time to first follow-up"))
                fig.update_xaxes(title="Seconds from session start")
                _chart(fig,"p17v22_first_followup_time",300)

        if events is not None and not events.empty:
            chats=events[events["event_type"].eq("chat_turn")].copy()
            if not chats.empty:
                timeline=[]
                for _,event in chats.iterrows():
                    try: payload=json.loads(str(event.get("payload_json") or "{}"))
                    except Exception: payload={}
                    timeline.append({
                        "Session":str(event.get("session_hash"))[:8],
                        "Seconds":payload.get("time_since_session_start_sec"),
                        "Category":_category_label(payload.get("category") or "other_follow_up"),
                    })
                tdf=pd.DataFrame(timeline)
                tdf["Seconds"]=pd.to_numeric(tdf["Seconds"],errors="coerce")
                tdf=tdf.dropna(subset=["Seconds"])
                if not tdf.empty:
                    st.markdown("#### Follow-up timing by category")
                    fig=go.Figure()
                    for category,group in tdf.groupby("Category"):
                        fig.add_trace(go.Scatter(x=group["Seconds"],y=group["Session"],mode="markers",name=category,marker=dict(size=11)))
                    fig.update_xaxes(title="Seconds from session start")
                    fig.update_yaxes(title="Participant session")
                    fig.update_layout(legend=dict(orientation="h",y=1.12))
                    _chart(fig,"p17v22_followup_timing",420)

    with tabs[3]:
        left,right=st.columns(2)
        with left:
            st.markdown("#### Final recommendation decisions")
            decisions=analysis.get("decision_counts") or {}
            if decisions:
                frame=pd.DataFrame([{"Decision":k.title(),"Count":v} for k,v in decisions.items()])
                fig=go.Figure(go.Bar(x=frame["Decision"],y=frame["Count"],text=frame["Count"],textposition="outside"))
                fig.update_yaxes(title="Completed participants",dtick=1)
                _chart(fig,"p17v22_decisions",330)
            else:
                st.info("No completed decisions yet.")
        with right:
            st.markdown("#### Decision confidence")
            if completed.empty:
                st.info("No completed participants yet.")
            else:
                conf=pd.to_numeric(completed["final_confidence"],errors="coerce").dropna()
                if conf.empty:
                    st.info("No confidence ratings yet.")
                else:
                    counts=conf.value_counts().sort_index()
                    fig=go.Figure(go.Bar(x=counts.index.astype(int),y=counts.values,text=counts.values,textposition="outside"))
                    fig.update_xaxes(range=[.5,7.5],dtick=1,title="Confidence (1–7)")
                    fig.update_yaxes(title="Participants",dtick=1)
                    _chart(fig,"p17v22_confidence",330)

        if not completed.empty:
            st.markdown("#### Decision time × information-seeking depth")
            frame=completed.copy()
            frame["Decision time"]=pd.to_numeric(frame["total_time_to_final_decision_sec"],errors="coerce")
            frame["Follow-ups"]=pd.to_numeric(frame["follow_up_count"],errors="coerce").fillna(0)
            frame["Evidence views"]=pd.to_numeric(frame["evidence_view_count"],errors="coerce").fillna(0)
            fig=go.Figure()
            for decision,group in frame.groupby("recommendation_decision",dropna=False):
                fig.add_trace(go.Scatter(
                    x=group["Decision time"],y=group["Follow-ups"],mode="markers+text",name=str(decision).title(),
                    text=group["session_hash"].astype(str).str[:6],textposition="top center",
                    marker=dict(size=10+2*group["Evidence views"].clip(upper=8)),
                    customdata=group[["Evidence views","final_confidence"]].to_numpy(),
                    hovertemplate="Decision time %{x:.1f}s<br>Follow-ups %{y}<br>Evidence views %{customdata[0]}<br>Confidence %{customdata[1]}/7<extra></extra>"
                ))
            fig.update_xaxes(title="Seconds to final decision")
            fig.update_yaxes(title="Follow-up questions",rangemode="tozero")
            fig.update_layout(legend=dict(orientation="h",y=1.12))
            _chart(fig,"p17v22_decision_depth",410)

            st.markdown("#### Time to final decision distribution")
            times=pd.to_numeric(completed["total_time_to_final_decision_sec"],errors="coerce").dropna()
            if not times.empty:
                fig=go.Figure(go.Box(x=times,boxpoints="all",jitter=.25,pointpos=0,name="Completed sessions"))
                fig.update_xaxes(title="Seconds from session start to final decision")
                _chart(fig,"p17v22_decision_time",290)

    with tabs[4]:
        if summaries.empty:
            st.info("No participant sessions yet.")
        else:
            session=st.selectbox("Participant session",summaries["session_hash"].astype(str).tolist(),key=PFX+"_sequence_session")
            raw=events[events["session_hash"].astype(str).eq(str(session))].copy().sort_values("id") if not events.empty else pd.DataFrame()
            if raw.empty:
                st.info("No timestamped events for this session.")
            else:
                rows=[]
                for idx,(_,event) in enumerate(raw.iterrows(),1):
                    try: payload=json.loads(str(event.get("payload_json") or "{}"))
                    except Exception: payload={}
                    elapsed=payload.get("time_since_session_start_sec")
                    if elapsed is None and str(event.get("event_type"))=="session_end":
                        elapsed=payload.get("total_time_to_final_decision_sec")
                    rows.append({
                        "Step":idx,
                        "Event":str(event.get("event_type")),
                        "Elapsed":elapsed,
                        "Created":event.get("created_at"),
                        "Payload":event.get("payload_json"),
                    })
                seq=pd.DataFrame(rows)
                seq["Elapsed"]=pd.to_numeric(seq["Elapsed"],errors="coerce")
                st.markdown("#### Timestamped interaction path")
                x=seq["Elapsed"] if seq["Elapsed"].notna().any() else seq["Step"]
                xlabel="Seconds from session start" if seq["Elapsed"].notna().any() else "Interaction sequence index"
                fig=go.Figure(go.Scatter(
                    x=x,y=seq["Event"],mode="lines+markers+text",text=seq["Step"],textposition="top center",
                    customdata=seq[["Created"]].to_numpy(),marker=dict(size=12),
                    hovertemplate="Step %{text}<br>%{y}<br>Time %{x}<br>%{customdata[0]}<extra></extra>"
                ))
                fig.update_xaxes(title=xlabel)
                fig.update_yaxes(title=None)
                _chart(fig,"p17v22_sequence",max(390,70*seq["Event"].nunique()+160))
                with st.expander("Exact timestamped event payloads",expanded=False):
                    st.dataframe(seq,use_container_width=True,hide_index=True)

        if events is not None and not events.empty:
            composition=events.groupby(["session_hash","event_type"]).size().unstack(fill_value=0)
            if not composition.empty:
                st.markdown("#### Session × interaction-event composition")
                composition.index=composition.index.astype(str).str[:8]
                fig=go.Figure(go.Heatmap(
                    z=composition.values,x=composition.columns.tolist(),y=composition.index.tolist(),
                    text=composition.values,texttemplate="%{text}",colorscale="Blues",showscale=False,
                    hovertemplate="Session %{y}<br>%{x}: %{z}<extra></extra>"
                ))
                _chart(fig,"p17v22_event_composition",max(330,32*len(composition)+130))

    with tabs[5]:
        clusters=pd.DataFrame(analysis.get("exploratory_clusters") or [])
        if clusters.empty:
            st.info("Exploratory clustering becomes available after at least four completed sessions.")
        else:
            st.markdown("#### Exploratory interaction-pattern profiles")
            metric_cols=[c for c in clusters.columns if c.endswith("_mean")]
            heat=clusters.set_index("cluster")[metric_cols]
            norm=heat.copy()
            for col in metric_cols:
                lo,hi=heat[col].min(),heat[col].max()
                norm[col]=0.5 if hi==lo else (heat[col]-lo)/(hi-lo)
            fig=go.Figure(go.Heatmap(
                z=norm.values,x=[c.replace("_mean","").replace("_"," ").title() for c in metric_cols],
                y=["Cluster {}".format(x) for x in norm.index],
                text=[["{:.2f}".format(v) for v in row] for row in heat.values],texttemplate="%{text}",
                colorscale="Blues",colorbar=dict(title="Relative\nprofile")
            ))
            _chart(fig,"p17v22_cluster_profiles",360)

            cross=pd.DataFrame(analysis.get("cluster_by_expertise") or [])
            if not cross.empty and "cluster" in cross:
                st.markdown("#### Cluster × expertise group")
                matrix=cross.set_index("cluster")
                fig=go.Figure(go.Heatmap(
                    z=matrix.values,x=matrix.columns.tolist(),y=["Cluster {}".format(x) for x in matrix.index],
                    text=matrix.values,texttemplate="%{text}",colorscale="Blues",showscale=False
                ))
                _chart(fig,"p17v22_cluster_expertise",330)
            st.caption("Clusters are exploratory descriptive aids only; they are not confirmatory participant types.")

        if not completed.empty:
            st.markdown("#### Expertise-group behavior profile · descriptive only")
            metrics=["follow_up_count","evidence_view_count","repeat_evidence_access_count","total_time_to_final_decision_sec"]
            group=completed.groupby("experience_group")[metrics].mean(numeric_only=True)
            if not group.empty:
                norm=group.copy()
                for col in metrics:
                    lo,hi=group[col].min(),group[col].max()
                    norm[col]=0.5 if hi==lo else (group[col]-lo)/(hi-lo)
                fig=go.Figure(go.Heatmap(
                    z=norm.values,x=[x.replace("_"," ").title() for x in metrics],y=norm.index.tolist(),
                    text=[["{:.2f}".format(v) for v in row] for row in group.values],texttemplate="%{text}",
                    colorscale="Blues",colorbar=dict(title="Relative\nlevel")
                ))
                _chart(fig,"p17v22_expertise_behavior",360)
                st.caption("Cell text shows the exact group mean; color is normalized within each metric for visual comparison only.")

    with tabs[6]:
        themes=analysis.get("manual_theme_counts") or {}
        if themes:
            scheme=coding_scheme()
            labels={str(x.get("id")):str(x.get("label")) for x in scheme.get("codes") or []}
            frame=pd.DataFrame([{"Theme":labels.get(k,k),"Coded sessions":v} for k,v in themes.items()]).sort_values("Coded sessions")
            st.markdown("#### Manual qualitative theme counts")
            fig=go.Figure(go.Bar(x=frame["Coded sessions"],y=frame["Theme"],orientation="h",text=frame["Coded sessions"],textposition="outside"))
            _chart(fig,"p17v22_theme_counts",390)

            codes=manual_coding(events)
            if not codes.empty and "themes" in codes:
                theme_sets=[set(x or []) for x in codes["themes"].tolist()]
                keys=sorted(set().union(*theme_sets)) if theme_sets else []
                if len(keys)>=2:
                    z=[]
                    for a in keys:
                        z.append([sum(1 for s in theme_sets if a in s and b in s) for b in keys])
                    st.markdown("#### Qualitative-theme co-occurrence")
                    fig=go.Figure(go.Heatmap(
                        z=z,x=[labels.get(k,k) for k in keys],y=[labels.get(k,k) for k in keys],
                        text=z,texttemplate="%{text}",colorscale="Blues",showscale=False,
                        hovertemplate="%{y} + %{x}<br>Coded together %{z} times<extra></extra>"
                    ))
                    _chart(fig,"p17v22_theme_cooccurrence",420)
        else:
            st.info("No manually coded sessions yet.")

def _researcher(mode: str, current_results: List[dict], current_ranking: List[dict]) -> None:
    key = _researcher_key(mode)
    if not st.session_state.get(PFX + "_researcher_unlocked"):
        if not key:
            st.error("Configure AWAREML_STUDY_RESEARCHER_KEY for Main Study researcher access.")
            return
        entered = st.text_input("Researcher key", type="password", key=PFX + "_rk")
        if st.button("Unlock researcher workspace", type="primary", key=PFX + "_unlock"):
            if entered == key:
                st.session_state[PFX + "_researcher_unlocked"] = True
                st.rerun()
            else:
                st.error("Incorrect researcher key.")
        return

    p = protocol()
    scheme = coding_scheme()
    st.success("Researcher workspace · Information-Seeking v2.1")

    section("Study readiness", "The richer workspace separates protocol readiness, standardized context and participant behavior.")
    ready = st.columns(6)
    ready[0].metric("Protocol", "Final")
    ready[1].metric("Main target", "12")
    ready[2].metric("Coding scheme", "Final")
    ready[3].metric("Manual coding", "Authoritative")
    ready[4].metric("Response mode", "Grounded")
    ready[5].metric("Design lock", "Locked")

    section(
        "Standardized scenario + measured evidence context",
        "The scenario defines what matters; the AwareML run supplies measured evidence. Researcher alignment is explicitly confirmed before Main Study use.",
    )
    catalog = scenario_catalog()
    options = catalog["benchmark_id"].astype(str).tolist()
    chosen = st.selectbox("Standardized Track-1 scenario", options, key=PFX + "_scenario")
    row = catalog[catalog["benchmark_id"].astype(str) == chosen].iloc[0].to_dict()
    _scenario_card(row)

    saved = load_context(mode)
    if saved:
        _evidence_contract(saved)
        st.success(
            "Saved standardized {} context · Scenario {} · Context ID {}".format(
                "Main Study" if mode == "final" else "Pilot",
                (saved.get("scenario") or {}).get("benchmark_id"),
                str(saved.get("context_id", ""))[:12],
            )
        )
        if not saved.get("scenario_alignment_attested"):
            st.warning("This saved context predates the v2.1 scenario-alignment confirmation. Confirm it before collecting revised Main Study participants.")
            confirm_old = st.checkbox(
                "I confirm that the saved Decision Lab objectives/weights operationalize the saved scenario's Track-1 scenario priorities.",
                key=PFX + "_confirm_saved",
            )
            if st.button("Confirm alignment for saved context", disabled=not confirm_old, use_container_width=True, key=PFX + "_attest_saved"):
                save_context(mode, saved.get("results") or [], saved.get("ranking") or [], saved.get("scenario") or {}, alignment_attested=True)
                st.success("Scenario alignment confirmed for the saved context.")
                st.rerun()
    else:
        st.warning("No standardized {} v2.1 context is saved.".format("Main Study" if mode == "final" else "Pilot"))

    expected = {"AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha"}
    have = {str(r.get("framework")) for r in current_results}
    ready_context = bool(current_ranking) and expected.issubset(have)
    if ready_context:
        align = st.checkbox(
            "I confirm that the current Decision Lab objectives/weights operationalize this selected scenario's Track-1 scenario priorities: {}.".format(
                str(row.get("ground_truth") or "N/A").replace("|", " + ")
            ),
            key=PFX + "_alignment_confirm",
        )
        if st.button(
            "Initialize / replace standardized {} context".format("Main Study" if mode == "final" else "Pilot"),
            type="primary",
            use_container_width=True,
            disabled=not align,
            key=PFX + "_savectx",
        ):
            scenario = {k: row.get(k) for k in ["benchmark_id", "scenario_id", "scenario", "ground_truth", "domain", "source"]}
            save_context(mode, current_results, current_ranking, scenario, alignment_attested=True)
            st.success("Standardized v2.1 context saved and alignment attested.")
            st.rerun()
    else:
        st.caption("To replace the context: run all five frameworks and open Decision Lab once, then return here.")

    context_for_plot = load_context(mode)
    if context_for_plot:
        with st.expander("Standardized five-framework context", expanded=False):
            _ranking_table(context_for_plot.get("results") or [], context_for_plot.get("ranking") or [])
            frame = _ranking_frame(context_for_plot.get("results") or [], context_for_plot.get("ranking") or [])
            if not frame.empty and frame["Utility"].notna().any():
                fig = go.Figure(go.Bar(
                    x=frame["Utility"], y=frame["Framework"], orientation="h",
                    text=["#{}".format(int(x)) if pd.notna(x) else "" for x in frame["Rank"]], textposition="outside",
                ))
                fig.update_yaxes(autorange="reversed")
                fig.update_xaxes(title="Observed Decision Lab utility")
                _chart(fig, "p17v21_context_ranking", 330)

    events = StudyStore().export(_study_name(mode))
    summaries = session_summaries(events)
    analysis = analyze(events)

    section("Collection overview", "Pilot and Main Study events remain separate. Means and SDs are descriptive.")
    cards = st.columns(7)
    cards[0].metric("Registered", analysis.get("registered_sessions", 0))
    cards[1].metric("Completed", analysis.get("completed_sessions", 0))
    follow_rate = analysis.get("follow_up_rate")
    cards[2].metric("Asked follow-up", "N/A" if follow_rate is None else "{:.1%}".format(follow_rate))
    for card, key, label in zip(
        cards[3:],
        ["follow_up_count", "evidence_view_count", "repeat_evidence_access_count", "total_time_to_final_decision_sec"],
        ["Follow-ups", "Evidence views", "Repeat views", "Decision time (s)"],
    ):
        rec = analysis.get(key) or {}
        card.metric(label + " mean", "N/A" if rec.get("mean") is None else "{:.2f}".format(rec["mean"]))
        card.caption("SD: {}".format("N/A" if rec.get("sd") is None else "{:.2f}".format(rec["sd"])))

    section("Research-grade behavior visualization", "Evidence navigation, question behavior, decisions, timing, interaction sequences, exploratory clusters and qualitative coding are presented as separate descriptive views.")
    _researcher_visuals(events, summaries, analysis)

    section("Manual qualitative coding", "Manual coding remains authoritative; automated categories and exploratory clusters are analysis aids only.")
    if summaries.empty:
        st.info("No participant sessions are available for coding.")
    else:
        session = st.selectbox("Session to code", summaries["session_hash"].astype(str).tolist(), key=PFX + "_codesess")
        labels = {str(x.get("id")): str(x.get("label")) for x in scheme.get("codes") or []}
        themes = st.multiselect("Manual qualitative themes", list(labels), format_func=lambda x: labels.get(x, x), key=PFX + "_themes")
        note = st.text_area("Coder note", height=80, key=PFX + "_note")
        if st.button("Save qualitative coding", key=PFX + "_savecode", type="primary"):
            StudyStore().log(
                _study_name(mode),
                "researcher-coding",
                "qualitative_code",
                {
                    "coded_session_hash": session,
                    "themes": themes,
                    "coder_note": note or None,
                    "coding_scheme": "information_seeking_qualitative_coding_scheme_v2",
                },
            )
            st.success("Coding saved.")

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Download event log CSV",
        data=events.to_csv(index=False).encode("utf-8"),
        file_name="information_seeking_v2_{}_events.csv".format(mode),
        mime="text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "Download session summaries JSON",
        data=summaries.to_json(orient="records", indent=2).encode("utf-8") if not summaries.empty else b"[]",
        file_name="information_seeking_v2_{}_sessions.json".format(mode),
        mime="application/json",
        use_container_width=True,
    )
    d3.download_button(
        "Download analysis JSON",
        data=json.dumps(analysis, indent=2, default=str).encode("utf-8"),
        file_name="information_seeking_v2_{}_analysis.json".format(mode),
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("Study protocol and coding scheme", expanded=False):
        st.json({"protocol": p, "coding_scheme": scheme}, expanded=False)
    with st.expander("Verbalization prompts for facilitated sessions", expanded=False):
        for q in [
            "Tell me what you are looking at before deciding whether to trust the recommendation.",
            "What information would you need before acting on this recommendation?",
            "What makes you ask a follow-up question — or decide not to ask one?",
            "Which visual or explanation detail changed your decision most?",
            "Was there a point where you felt you had enough information? Why?",
        ]:
            st.markdown("- " + q)


def information_seeking_research_page_v2() -> None:
    mode = _mode()
    hero(
        "HUMAN STUDY · INFORMATION SEEKING",
        "Information-Seeking Study",
        "Morten-review v2.1: explicit scenario-to-decision connection, grounded multi-topic answers, repeat-access logging, full interaction sequences and research-grade researcher visualization.",
        pills=[
            ("Target 12", "good"),
            ("Scenario-linked", "good"),
            ("Repeat access logged", "good"),
            ("Manual coding", "good"),
        ],
    )

    current = result_dicts()
    ranking = _state().get("ranking") or []
    saved = load_context(mode)
    ctx = saved
    if mode == "pilot" and ctx is None and current and ranking:
        row = scenario_catalog().iloc[0].to_dict()
        ctx = {
            "schema_version": "information_seeking_context_v2_1",
            "collection_mode": "pilot",
            "scenario": {k: row.get(k) for k in ["benchmark_id", "scenario_id", "scenario", "ground_truth", "domain", "source"]},
            "results": current,
            "ranking": ranking,
            "context_id": "current-session-pilot",
            "scenario_alignment_attested": False,
        }

    view = st.session_state.get(PFX + "_view", "home")
    if view == "home":
        section("Choose study access", "Participants inspect evidence; researchers manage the standardized scenario context, behavior analysis, sequences and coding.")
        c1, c2 = st.columns(2)
        if c1.button("Enter participant study", type="primary", use_container_width=True, key=PFX + "_enterp"):
            st.session_state[PFX + "_view"] = "participant"
            st.rerun()
        if c2.button("Open researcher workspace", use_container_width=True, key=PFX + "_enterr"):
            st.session_state[PFX + "_view"] = "researcher"
            st.rerun()
        cards = st.columns(5)
        for card, (label, value) in zip(cards, [
            ("Main Study target", "12"),
            ("Expertise scale", "1–7"),
            ("Evidence revisits", "Logged"),
            ("Interaction sequence", "Timestamped"),
            ("Researcher visuals", "Enhanced"),
        ]):
            card.metric(label, value)
        if mode == "final" and saved is None:
            st.warning("Main Study v2.1 context must be initialized in Researcher Workspace before participants enter.")
        elif ctx is not None:
            if mode == "final" and not ctx.get("scenario_alignment_attested"):
                st.warning("The saved Main Study context needs one researcher alignment confirmation before revised participant collection.")
            else:
                st.success("Participant context ready · Scenario {}.".format((ctx.get("scenario") or {}).get("benchmark_id", "N/A")))
        return

    if st.button("Back to study home", key=PFX + "_back"):
        st.session_state[PFX + "_view"] = "home"
        st.rerun()

    if view == "researcher":
        _researcher(mode, current, ranking)
        return

    if ctx is None:
        empty_state("Standardized scenario/evidence context required", "Researcher must initialize the revised Main Study context first.")
        return
    if mode == "final" and not ctx.get("scenario_alignment_attested"):
        empty_state(
            "Researcher scenario-alignment confirmation required",
            "Open Researcher Workspace and confirm that the saved Decision Lab objectives/weights operationalize the selected Track-1 scenario priorities.",
        )
        return
    _participant(mode, ctx)
