from __future__ import annotations

# Read-only snapshots of existing chart renderers. No StudyStore or production analysis writes.
import json
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PFX = "awareml_demo17"
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "illustrative_demo" / "awareml_34_cases.json"
# parents[2] is the repository root for awareml/ui_v2/module.py.

SECONDARY_ITEMS = [('critical_evaluation', 'I felt able to critically evaluate the recommendation provided by AwareML.'), ('freedom_to_override', "I felt free to reject or override AwareML's recommendation when I disagreed with it."), ('final_control', 'I understood that I remained responsible for the final decision about whether to rely on the AwareML recommendation.'), ('rely_or_question', 'The explanations helped me judge when to rely on the recommendation and when to question it.'), ('evidence_checking', 'I compared the AI explanation with the reference evidence before deciding whether to rely on it.'), ('explanation_usefulness', 'The explanations were useful for understanding why the system made its recommendation.'), ('interface_usability', 'The AwareML study interface made it easy to compare an explanation with its supporting evidence.'), ('real_world_support_trust', 'I would trust AwareML to support me in a real-world streaming AutoML decision.')]

def _category_label(key):
    return str(key).replace('_', ' ').title()


def _chart(fig, key, height=360):
    if key in {'demo17_first_followup_time', 'demo17_followup_timing',
               'demo17_decision_depth', 'demo17_decision_time'}:
        st.info('Timing visualization unavailable: generated cases contain no observed timestamps or durations.')
        return
    if key == 'demo16_trial_dynamics':
        fig.add_annotation(text='Response time not recorded', x=.5, y=.5,
                           xref='paper', yref='paper', showarrow=False)
    fig.add_annotation(text='ILLUSTRATIVE DEMO — GENERATED DATA', x=1, y=1.14,
                       xref='paper', yref='paper', xanchor='right', showarrow=False,
                       font=dict(size=10, color='#64748b'))
    fig.update_layout(template='plotly_white', height=height,
                      margin=dict(l=70, r=40, t=90, b=70))
    st.plotly_chart(fig, use_container_width=True, key=key,
                    config={'displaylogo':False})


def load_demo(path=None):
    raw = json.loads(Path(path or DATA_PATH).read_text(encoding='utf-8'))
    required = ['participants', 'trust_trials', 'post_study', 'information_seeking', 'scripted_events']
    if raw.get('is_synthetic') is not True or raw.get('eligible_for_main_study') is not False:
        raise ValueError('Only explicitly generated demonstration data are allowed.')
    for name in required:
        if not isinstance(raw.get(name), list) or not raw[name]:
            raise ValueError('Missing demo records: '+name)
        if any(r.get('is_synthetic') is not True or r.get('eligible_for_main_study') is not False for r in raw[name]):
            raise ValueError('Unlabeled or real records rejected: '+name)
    people = raw['participants']; trials = raw['trust_trials']
    codes = [p['participant_code'] for p in people]
    if len(codes) != 34 or len(set(codes)) != 34 or len(trials) != 680:
        raise ValueError('Expected 34 distinct demo cases and 680 trials.')
    for code in codes:
        rows = [t for t in trials if t['participant_code']==code]
        if len(rows)!=20 or len({r['pair_id'] for r in rows})!=20:
            raise ValueError('Invalid matched-family assignment for '+code)
        if sum(r['condition']=='known_correct' for r in rows)!=10:
            raise ValueError('Invalid condition balance for '+code)
        for r in rows:
            if any(not 1 <= float(r[k]) <= 7 for k in ['trust_1','trust_2','trust_3','factual_correctness','fluency','confident_language']):
                raise ValueError('Rating outside 1–7')
            if abs(sum(r[k] for k in ['trust_1','trust_2','trust_3'])/3-r['trust_mean'])>1e-8:
                raise ValueError('Trust mean mismatch')
    for name in ['post_study','information_seeking']:
        if Counter(r['participant_code'] for r in raw[name]) != Counter(codes):
            raise ValueError('Demo case linkage mismatch: '+name)
    return raw


class ReadOnlyDemoStore:
    def __init__(self, responses, participants, events, secondary):
        self.rows = (responses, participants, events, secondary)
    def response_rows(self, mode): return list(self.rows[0])
    def participant_rows(self, mode): return list(self.rows[1])
    def event_rows(self, mode): return list(self.rows[2])
    def secondary_rows(self, mode): return list(self.rows[3])


def trust_adapter(raw):
    ids={p['participant_code']:'D%03d_demo'%(i+1) for i,p in enumerate(raw['participants'])}
    response=[]
    for r in raw['trust_trials']:
        response.append(dict(participant_hash=ids[r['participant_code']],
          source_demo_code=r['participant_code'], correctness_condition='correct' if r['condition']=='known_correct' else 'incorrect',
          trust_rating=r['trust_mean'], response_time_sec=float('nan'), order_index=r['trial_position'],
          evidence_question_count=r['evidence_question_count'], perceived_correctness=r['factual_correctness'],
          fluency_rating=r['fluency'], perceived_confidence=r['confident_language'],
          trust_items_json=json.dumps(dict(zip(['stias_confident','stias_reliable','stias_trust'],[r['trust_1'],r['trust_2'],r['trust_3']]))),
          source_stage=r['pair_id'].split('_',2)[2], framework_context=r['framework'],
          decision_action=r['recommendation_action'].split(' — ')[0],
          override_targets_json=json.dumps(['Required evidence'] if r['recommendation_action'].startswith('Override') else []),
          override_correction_text=r['override_text'], explanation_action=r['explanation_action'],
          correction_text=r['correction_text'], pair_id=r['pair_id'], scenario_id=r['scenario_id']))
    participants=[dict(participant_hash=ids[p['participant_code']], completed_at='demonstration_completion_marker_not_a_timestamp',
                       counterbalance_arm=p['arm'], expertise_group=p['experience_group'],
                       expertise_self_rating=p['expertise_rating']) for p in raw['participants']]
    secondary=[dict(participant_hash=ids[p['participant_code']], ratings_json=json.dumps({k:p['item_'+str(i+1)] for i,(k,_) in enumerate(SECONDARY_ITEMS)}),open_comment=p['comment']) for p in raw['post_study']]
    events=[dict(event_type='evidence_question') for r in response for _ in range(r['evidence_question_count'])]
    df=pd.DataFrame(response); cover=[]
    for pair,g in df.groupby('pair_id'):
        cover.append(dict(pair_id=pair, scenario_id=g.iloc[0]['scenario_id'], framework=g.iloc[0]['framework_context'],
                          source_stage=g.iloc[0]['source_stage'], evaluations=len(g),correct=int(g['correctness_condition'].eq('correct').sum()),incorrect=int(g['correctness_condition'].eq('incorrect').sum())))
    joined=df.merge(pd.DataFrame(participants),on='participant_hash')
    expertise=[]
    for group,g in joined.groupby('expertise_group'):
        per=g.groupby('participant_hash')['trust_rating'].mean()
        expertise.append(dict(expertise_group=group,mean_trust=per.mean(),sd_trust=per.std(),participants=len(per)))
    return SimpleNamespace(store=ReadOnlyDemoStore(response,participants,events,secondary)),dict(pair_coverage=cover,expertise_descriptive=expertise)


def information_adapter(raw):
    ids={p['participant_code']:'D%03d_demo'%(i+1) for i,p in enumerate(raw['participants'])}
    meta={p['participant_code']:p for p in raw['participants']}
    summaries=[];events=[];views=Counter();revisits=Counter();categories=Counter();topics=Counter();decisions=Counter()
    for r in raw['information_seeking']:
        code=r['participant_code']; decision=r['recommendation_action'].split(' — ')[0].lower();decisions[decision]+=1
        summaries.append(dict(session_hash=ids[code],completed=True,evidence_view_count=r['evidence_view_count'],follow_up_count=r['question_count'],repeat_evidence_access_count=r['revisit_count'],
          total_time_to_final_decision_sec=float('nan'),time_before_first_follow_up_sec=float('nan'),
          final_confidence=r['decision_confidence'],recommendation_decision=decision,
          experience_group=meta[code]['experience_group']))
        counts=Counter(r['evidence_sequence']); views.update(counts);revisits.update({k:max(0,v-1) for k,v in counts.items()})
    for i,e in enumerate(raw['scripted_events']):
        if e['track']!='information_seeking': continue
        payload=dict(e['payload']); typ=e['event_type']
        if typ=='scripted_evidence_view':typ='evidence_view';payload['evidence_area']=payload['area']
        elif typ=='scripted_question':
            typ='chat_turn';q=payload.get('question','').lower()
            cat='explanation_probe' if 'why' in q else 'evidence_request' if 'fairness' in q else 'clarification'
            payload['category']=cat; categories[cat]+=1
            for topic in ['fairness','drift','framework']:
                if topic in q:topics[topic]+=1
        elif typ=='scripted_final_decision':typ='session_end'
        events.append(dict(id=i,session_hash=ids[e['participant_code']],event_type=typ,created_at=None,payload_json=json.dumps(payload),event_source='scripted_not_observed'))
    df=pd.DataFrame(summaries);groups=[];cross=[]
    # Explicit demonstration grouping rules; no fitted or validated participant types.
    df['demo_group']=['No questions' if r['follow_up_count']==0 else 'Questions and revisits' if r['repeat_evidence_access_count']>0 else 'Questions without revisits' for _,r in df.iterrows()]
    for group,g in df.groupby('demo_group'):
        groups.append(dict(cluster=group,follow_ups_mean=g['follow_up_count'].mean(),evidence_views_mean=g['evidence_view_count'].mean(),revisits_mean=g['repeat_evidence_access_count'].mean()))
        cross.append(dict(cluster=group,**{x:int(g['experience_group'].eq(x).sum()) for x in sorted(df['experience_group'].unique())}))
    return pd.DataFrame(events),df,dict(evidence_view_counts=dict(views),evidence_revisit_counts=dict(revisits),category_counts=dict(categories),requested_topic_counts=dict(topics),decision_counts=dict(decisions),exploratory_clusters=groups,cluster_by_expertise=cross,manual_theme_counts={})


def _banner(raw):
    st.subheader('Illustrative demo results')
    st.warning('Generated examples only. The trust contrast was built into the data. These are not collected participant results or evidence of study effectiveness.')
    st.caption('Read-only demo files. Main Study databases, assessor reviews, readiness gates and inferential analyses are not used by this view.')
    c=st.columns(4); ps=raw['participants']
    correct=sum(x['mean_trust_correct'] for x in ps)/len(ps);incorrect=sum(x['mean_trust_incorrect'] for x in ps)/len(ps)
    for col,label,value in zip(c,['Demo cases','Correct trust','Incorrect trust','Illustrative difference'],[str(len(ps)),f'{correct:.2f}',f'{incorrect:.2f}',f'{correct-incorrect:+.2f}']):col.metric(label,value)


def _downloads(raw, track):
    with st.expander('Demo records and downloads'):
        names=['trust_trials','post_study','participants'] if track=='trust' else ['information_seeking','scripted_events']
        for name in names:
            frame=pd.DataFrame(raw[name]);st.dataframe(frame,use_container_width=True,hide_index=True)
            st.download_button('Download illustrative '+name,frame.to_csv(index=False).encode('utf-8'),file_name='ILLUSTRATIVE_'+name+'.csv',mime='text/csv',key='demo_download_'+track+'_'+name)
        st.download_button('Download complete illustrative JSON',json.dumps(raw,indent=2).encode('utf-8'),file_name='ILLUSTRATIVE_34_cases.json',mime='application/json',key='demo_json_'+track)


def _researcher_access():
    # Prevent participant access to condition labels and the answer-bearing exports.
    allowed = ['p16_task_context_r1_researcher_unlocked',
               'p17_task_context_r1_researcher_unlocked']
    if any(st.session_state.get(k) for k in allowed):
        return True
    st.info('Researcher access required. Open Study workspace, choose Open researcher workspace, and unlock it using your existing researcher key. Then return to Illustrative demo results.')
    return False


def render_trust_demo():
    if not _researcher_access(): return
    try: raw=load_demo()
    except (OSError,ValueError,KeyError,TypeError) as e: st.error('Demo data could not be loaded: '+str(e));return
    _banner(raw)
    study,summary=trust_adapter(raw);_trust_visuals(study,'illustrative_demo',summary)
    st.subheader('Explanation handling and correction examples')
    frame=pd.DataFrame(raw['trust_trials']);counts=frame.groupby(['condition','explanation_action']).size().unstack(fill_value=0)
    fig=go.Figure([go.Bar(x=counts.index,y=counts[c],name=c) for c in counts.columns]);fig.update_layout(barmode='stack');_chart(fig,'demo16_explanation_handling',420)
    st.subheader('Fluency and confident-sounding language')
    fig=go.Figure()
    for metric in ['fluency','confident_language']:
        for condition in ['known_correct','known_incorrect']:
            g=frame[frame['condition'].eq(condition)]
            fig.add_trace(go.Box(y=g[metric],name=metric.replace('_',' ').title()+' / '+condition.replace('known_',''),boxmean=True))
    fig.update_yaxes(range=[1,7],dtick=1,title='Generated rating (1–7)')
    _chart(fig,'demo16_language_diagnostics',400)
    st.subheader('Perceived critical-evaluation ability and optional evidence seeking')
    fractions=frame.groupby('participant_code')['evidence_question_count'].apply(lambda x:float((x>0).mean())).to_dict()
    post=raw['post_study']
    fig=go.Figure(go.Scatter(x=[r['item_1'] for r in post],y=[fractions[r['participant_code']] for r in post],mode='markers',text=[r['participant_code'] for r in post],marker=dict(size=11),hovertemplate='%{text}<br>Critical evaluation %{x}/7<br>Fraction seeking evidence %{y:.2f}<extra></extra>'))
    fig.update_xaxes(range=[.5,7.5],dtick=1,title='Generated critical-evaluation rating')
    fig.update_yaxes(range=[0,1],title='Fraction of trials with scripted questions')
    _chart(fig,'demo16_critical_evaluation_seeking',400)
    st.caption('This association is generated, not evidence of actual critical-evaluation ability.')
    _downloads(raw,'trust')


def render_information_demo():
    if not _researcher_access(): return
    try: raw=load_demo()
    except (OSError,ValueError,KeyError,TypeError) as e: st.error('Demo data could not be loaded: '+str(e));return
    _banner(raw)
    st.info('All 34 cases include this activity for demonstration. Timing and human qualitative coding are unavailable. Scripted sequences are not observed behavior.')
    with st.expander('Illustrative context and evidence'):
        ctx=raw['information_seeking_context'];st.write(ctx['scenario']);st.write(ctx['explanation']);st.dataframe(pd.DataFrame(ctx['ranking']),use_container_width=True,hide_index=True);st.json(ctx['supplementary_evidence'])
    events,summaries,analysis=information_adapter(raw);_information_visuals(events,summaries,analysis)
    _downloads(raw,'information')


def _trust_visuals(study: TrustCalibrationStudyV2, mode: str, summary: dict):
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

    st.info('34 illustrative cases; this is not a Main Study completion count.')

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
            _chart(fig, "demo16_calibration_map", 500)
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
            _chart(fig, "demo16_condition_distribution", 390)
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
            _chart(fig, "demo16_correctness_detection", 390)

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
            _chart(fig, "demo16_stias_items", 360)

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
            _chart(fig, "demo16_type_discrimination", 420)

        if not paired.empty:
            st.markdown("#### Participant calibration-gap distribution")
            fig=go.Figure(go.Box(
                x=paired["gap"], boxpoints="all", jitter=.25, pointpos=0,
                name="Correct − incorrect trust gap",
                hovertemplate="Gap %{x:+.2f}<extra></extra>",
            ))
            fig.add_vline(x=0,line_dash="dash",line_color="#94A3B8")
            fig.update_xaxes(title="Within-participant trust gap", zeroline=False)
            _chart(fig, "demo16_gap_distribution", 300)

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
            _chart(fig,"demo16_decisions",350)

        c1,c2=st.columns(2)
        with c1:
            st.markdown("#### Trust by final action")
            fig=go.Figure()
            for action in ["Accept","Override","Reject"]:
                g=analysis_rows[analysis_rows["decision_action"].eq(action)]
                if g.empty: continue
                fig.add_trace(go.Box(y=g["trust_rating"],name=action,boxpoints="all",jitter=.25,pointpos=0,boxmean=True))
            fig.update_yaxes(range=[1,7],dtick=1,title="S-TIAS mean trust")
            _chart(fig,"demo16_trust_by_action",360)
        with c2:
            st.markdown("#### Optional evidence questions by final action")
            fig=go.Figure()
            for action in ["Accept","Override","Reject"]:
                g=analysis_rows[analysis_rows["decision_action"].eq(action)]
                if g.empty: continue
                fig.add_trace(go.Box(y=g["evidence_question_count"],name=action,boxpoints="all",jitter=.25,pointpos=0,boxmean=True))
            fig.update_yaxes(title="Evidence questions / trial",rangemode="tozero")
            _chart(fig,"demo16_questions_by_action",360)

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
                _chart(fig,"demo16_override_targets",360)
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
            _chart(fig,"demo16_pair_heat",max(480,20*len(heat)+120))

            st.markdown("#### Scenario evaluation counts")
            scenario=pair[["scenario_id","evaluations","framework","source_stage"]].sort_values("evaluations")
            fig=go.Figure(go.Scatter(
                x=scenario["evaluations"],y=scenario["scenario_id"],mode="markers+text",
                text=scenario["evaluations"],textposition="middle right",marker=dict(size=11),
                customdata=scenario[["framework","source_stage"]].to_numpy(),
                hovertemplate="Scenario %{y}<br>Evaluations %{x}<br>Framework %{customdata[0]}<br>Type %{customdata[1]}<extra></extra>",
            ))
            fig.update_xaxes(title="Completed evaluations")
            _chart(fig,"demo16_scenario_counts",max(460,20*len(scenario)+120))

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
            _chart(fig,"demo16_framework_type",360)

        if not participants.empty and "counterbalance_arm" in participants:
            arms=participants[participants["participant_hash"].astype(str).isin(completed_hashes)]["counterbalance_arm"].fillna("Unknown").value_counts()
            if not arms.empty:
                st.markdown("#### Counterbalance-arm completion")
                fig=go.Figure(go.Bar(x=arms.index,y=arms.values,text=arms.values,textposition="outside"))
                fig.update_yaxes(title="Completed participants",dtick=1)
                _chart(fig,"demo16_counterbalance",280)

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
            _chart(fig,"demo16_trial_dynamics",650)

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
            _chart(fig,"demo16_question_heat",360)

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
                _chart(fig,"demo16_human_agency_likert",520)

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
                _chart(fig,"demo16_human_agency_means",460)
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
            _chart(fig,"demo16_expertise_gap",420)

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
            _chart(fig,"demo16_expertise_mean",340)
        st.caption("Expertise analyses are descriptive secondary analyses; the study is not powered for confirmatory between-group comparisons.")

def _information_visuals(events: pd.DataFrame, summaries: pd.DataFrame, analysis: dict) -> None:
    completed = summaries[summaries["completed"] == True].copy() if not summaries.empty else summaries
    n_completed = int(len(completed)) if completed is not None else 0

    st.info('34 scripted demonstration sessions, not 34 observed sessions. The real Track 3 target is unchanged.')

    tabs = st.tabs([
        "Overview",
        "Evidence navigation",
        "Question behavior",
        "Decision journey",
        "Interaction sequences",
        "Demo groups & expertise",
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
            _chart(fig,"demo17_funnel",420)

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
            _chart(fig,"demo17_behavior_map",430)
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
            _chart(fig,"demo17_evidence_access",400)

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
                    _chart(fig,"demo17_access_matrix",max(340,32*len(mat)+130))
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
                    _chart(fig,"demo17_transition_matrix",390)

    with tabs[2]:
        left,right=st.columns(2)
        with left:
            st.markdown("#### Follow-up categories")
            categories=analysis.get("category_counts") or {}
            if categories:
                frame=pd.DataFrame([{"Category":_category_label(k),"Count":v} for k,v in categories.items()]).sort_values("Count")
                fig=go.Figure(go.Bar(x=frame["Count"],y=frame["Category"],orientation="h",text=frame["Count"],textposition="outside"))
                _chart(fig,"demo17_categories",360)
            else:
                st.info("No follow-up questions have been recorded.")
        with right:
            st.markdown("#### Requested evidence topics")
            topics=analysis.get("requested_topic_counts") or {}
            if topics:
                frame=pd.DataFrame([{"Topic":str(k).replace("_"," ").title(),"Count":v} for k,v in topics.items()]).sort_values("Count")
                fig=go.Figure(go.Bar(x=frame["Count"],y=frame["Topic"],orientation="h",text=frame["Count"],textposition="outside"))
                _chart(fig,"demo17_topics",360)
            else:
                st.info("No requested topics have been recorded.")

        if not completed.empty:
            st.markdown("#### Time to first follow-up")
            first=pd.to_numeric(completed["time_before_first_follow_up_sec"],errors="coerce").dropna()
            if first.empty:
                st.info("Follow-ups exist, but their timings were not recorded in this generated dataset.")
            else:
                fig=go.Figure(go.Box(x=first,boxpoints="all",jitter=.25,pointpos=0,name="Time to first follow-up"))
                fig.update_xaxes(title="Seconds from session start")
                _chart(fig,"demo17_first_followup_time",300)

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
                    _chart(fig,"demo17_followup_timing",420)

    with tabs[3]:
        left,right=st.columns(2)
        with left:
            st.markdown("#### Final recommendation decisions")
            decisions=analysis.get("decision_counts") or {}
            if decisions:
                frame=pd.DataFrame([{"Decision":k.title(),"Count":v} for k,v in decisions.items()])
                fig=go.Figure(go.Bar(x=frame["Decision"],y=frame["Count"],text=frame["Count"],textposition="outside"))
                fig.update_yaxes(title="Completed participants",dtick=1)
                _chart(fig,"demo17_decisions",330)
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
                    _chart(fig,"demo17_confidence",330)

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
            _chart(fig,"demo17_decision_depth",410)

            st.markdown("#### Time to final decision distribution")
            times=pd.to_numeric(completed["total_time_to_final_decision_sec"],errors="coerce").dropna()
            if not times.empty:
                fig=go.Figure(go.Box(x=times,boxpoints="all",jitter=.25,pointpos=0,name="Completed sessions"))
                fig.update_xaxes(title="Seconds from session start to final decision")
                _chart(fig,"demo17_decision_time",290)

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
                st.markdown("#### Scripted interaction sequence")
                x=seq["Elapsed"] if seq["Elapsed"].notna().any() else seq["Step"]
                xlabel="Seconds from session start" if seq["Elapsed"].notna().any() else "Interaction sequence index"
                fig=go.Figure(go.Scatter(
                    x=x,y=seq["Event"],mode="lines+markers+text",text=seq["Step"],textposition="top center",
                    customdata=seq[["Created"]].to_numpy(),marker=dict(size=12),
                    hovertemplate="Step %{text}<br>%{y}<br>Time %{x}<br>%{customdata[0]}<extra></extra>"
                ))
                fig.update_xaxes(title=xlabel)
                fig.update_yaxes(title=None)
                _chart(fig,"demo17_sequence",max(390,70*seq["Event"].nunique()+160))
                with st.expander("Scripted event payloads without observed timestamps",expanded=False):
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
                _chart(fig,"demo17_event_composition",max(330,32*len(composition)+130))

    with tabs[5]:
        clusters=pd.DataFrame(analysis.get("exploratory_clusters") or [])
        if clusters.empty:
            st.info("Exploratory clustering becomes available after at least four completed sessions.")
        else:
            st.markdown("#### Rule-based demonstration profiles")
            metric_cols=[c for c in clusters.columns if c.endswith("_mean")]
            heat=clusters.set_index("cluster")[metric_cols]
            norm=heat.copy()
            for col in metric_cols:
                lo,hi=heat[col].min(),heat[col].max()
                norm[col]=0.5 if hi==lo else (heat[col]-lo)/(hi-lo)
            fig=go.Figure(go.Heatmap(
                z=norm.values,x=[c.replace("_mean","").replace("_"," ").title() for c in metric_cols],
                y=["Group {}".format(x) for x in norm.index],
                text=[["{:.2f}".format(v) for v in row] for row in heat.values],texttemplate="%{text}",
                colorscale="Blues",colorbar=dict(title="Relative\nprofile")
            ))
            _chart(fig,"demo17_cluster_profiles",360)

            cross=pd.DataFrame(analysis.get("cluster_by_expertise") or [])
            if not cross.empty and "cluster" in cross:
                st.markdown("#### Demonstration group × expertise")
                matrix=cross.set_index("cluster")
                fig=go.Figure(go.Heatmap(
                    z=matrix.values,x=matrix.columns.tolist(),y=["Group {}".format(x) for x in matrix.index],
                    text=matrix.values,texttemplate="%{text}",colorscale="Blues",showscale=False
                ))
                _chart(fig,"demo17_cluster_expertise",330)
            st.caption("Groups are assigned by explicit scripted activity rules, not discovered participant types.")

        if not completed.empty:
            st.markdown("#### Expertise-group behavior profile · descriptive only")
            metrics=["follow_up_count","evidence_view_count","repeat_evidence_access_count"]
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
                _chart(fig,"demo17_expertise_behavior",360)
                st.caption("Cell text shows the exact group mean; color is normalized within each metric for visual comparison only.")

    with tabs[6]:
        themes=analysis.get("manual_theme_counts") or {}
        if themes:
            scheme=coding_scheme()
            labels={str(x.get("id")):str(x.get("label")) for x in scheme.get("codes") or []}
            frame=pd.DataFrame([{"Theme":labels.get(k,k),"Coded sessions":v} for k,v in themes.items()]).sort_values("Coded sessions")
            st.markdown("#### Manual qualitative theme counts")
            fig=go.Figure(go.Bar(x=frame["Coded sessions"],y=frame["Theme"],orientation="h",text=frame["Coded sessions"],textposition="outside"))
            _chart(fig,"demo17_theme_counts",390)

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
                    _chart(fig,"demo17_theme_cooccurrence",420)
        else:
            st.info("Manual qualitative coding is not available for these generated cases. No human coding is implied.")
