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
    # awareml_generated_case_group_summary_7_12_15
    group_counts = Counter(p['experience_group'] for p in raw['participants'])
    st.caption('Generated case groups: {0} students | {1} practitioners | {2} ML / AutoML experts'.format(group_counts['University / CS student'], group_counts['Practitioner'], group_counts['ML / AutoML expert']))
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


def _condition_table(frame, column, order):
    counts=pd.crosstab(frame['condition'],frame[column]).reindex(index=['known_correct','known_incorrect'],columns=order,fill_value=0)
    pct=counts.div(counts.sum(axis=1),axis=0)*100
    pct.index=['Correct','Incorrect']
    return pct


def _stacked_table(frame, column, order, short, key):
    pct=_condition_table(frame,column,order)
    colors=['#2563eb','#e89d27','#c84f5d','#8c94a6']
    fig=go.Figure()
    for j,(name,label) in enumerate(zip(order,short)):
        fig.add_trace(go.Bar(x=pct[name],y=pct.index,orientation='h',name=label,marker_color=colors[j],text=[f'{x:.1f}%' for x in pct[name]],textposition='inside'))
    fig.update_layout(barmode='stack',legend=dict(orientation='h',y=-.2))
    fig.update_xaxes(range=[0,100],title='Share of generated trials (%)')
    _chart(fig,key,370)
    st.dataframe(pct.rename(columns=dict(zip(order,short))).round(2),use_container_width=True)


def render_trust_demo():
    if not _researcher_access():return
    try:raw=load_demo()
    except (OSError,ValueError,KeyError,TypeError) as e:st.error('Demo data could not be loaded: '+str(e));return
    _banner(raw)
    st.caption('Five selected charts. Generated policy settings shape these displays; they are not observed effects.')
    frame=pd.DataFrame(raw['trust_trials']);people=pd.DataFrame(raw['participants'])
    st.subheader('1  Paired trust by explanation condition')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=[1,7],y=[1,7],mode='lines',name='Equal trust',line=dict(color='#a5adba',dash='dash')))
    fig.add_trace(go.Scatter(x=people['mean_trust_incorrect'],y=people['mean_trust_correct'],mode='markers',text=people['participant_code'],name='Illustrative cases',marker=dict(size=10,color='#2563eb'),hovertemplate='%{text}<br>Incorrect %{x:.2f}<br>Correct %{y:.2f}<extra></extra>'))
    fig.update_xaxes(range=[1,7],title='Mean trust with incorrect explanations');fig.update_yaxes(range=[1,7],title='Mean trust with correct explanations')
    _chart(fig,'compact_demo_trust',440)
    ex=['Use the explanation as written to support my decision','Flag an inaccurate or unsupported claim and request a revised explanation before relying on it','Do not rely on this explanation to support my decision','I am unsure how I would use this explanation']
    st.subheader('2  Explanation handling')
    _stacked_table(frame,'explanation_action',ex,['Use as written','Flag and request revision','Do not rely','Unsure'],'compact_demo_handling')
    st.subheader('3  Recommendation decisions')
    st.info('Scripted behavior example: accepting the recommendation can be justified in both conditions because the framework choice remains supported. Lower acceptance is a chosen demo policy, not a necessary sign of calibration.')
    rec=['Accept — use the recommended framework','Override — propose a different framework choice or decision policy','Reject — do not act on the framework recommendation']
    _stacked_table(frame,'recommendation_action',rec,['Accept','Override','Reject'],'compact_demo_decisions')
    st.subheader('4  Explanation ratings')
    metrics=['factual_correctness','fluency','confident_language'];labels=['Factual correctness','Fluency','Confident language'];fig=go.Figure()
    for condition,label,color in [('known_correct','Correct','#2563eb'),('known_incorrect','Incorrect','#d58033')]:
        g=frame[frame['condition']==condition];fig.add_trace(go.Bar(x=labels,y=[g[m].mean() for m in metrics],name=label,marker_color=color,text=[f'{g[m].mean():.2f}' for m in metrics],textposition='outside'))
    fig.update_layout(barmode='group');fig.update_yaxes(range=[0,7.6],title='Generated mean rating (1–7)',dtick=1)
    _chart(fig,'compact_demo_diagnostics',400)
    st.caption('Factual ratings distinguish correctness. Fluency and confident language remain similar because numerical errors do not necessarily change presentation.')
    st.subheader('5  Post-study human agency and study experience')
    short=['Critical evaluation','Freedom to override','Final responsibility','Rely or question','Evidence checking','Explanation usefulness','Interface usability','Real-world support trust']
    post=raw['post_study'];fig=go.Figure()
    colors=['#a94c51','#c57073','#dfa4a5','#d8dce2','#a7c6e6','#568fc6','#225fa4']
    for rating in range(1,8):
        shares=[100*sum(r['item_'+str(j)]==rating for r in post)/len(post) for j in range(1,9)]
        fig.add_trace(go.Bar(x=shares,y=short,orientation='h',name=str(rating),marker_color=colors[rating-1]))
    fig.update_layout(barmode='stack',legend=dict(orientation='h',title='Rating',y=-.2));fig.update_xaxes(range=[0,100],title='Share of generated responses (%)')
    _chart(fig,'compact_demo_agency',470)
    st.caption('1 = Strongly disagree; 7 = Strongly agree. High values indicate reported agreement in the example, not established critical-evaluation skill. These secondary items are not a validated composite scale.')
    with st.expander('Exact post-study item means'):
        st.dataframe(pd.DataFrame([{'Statement':text,'Mean':sum(r['item_'+str(j+1)] for r in post)/len(post)} for j,(_,text) in enumerate(SECONDARY_ITEMS)]),use_container_width=True,hide_index=True)
    _downloads(raw,'trust')


def render_information_demo():
    if not _researcher_access():return
    try:raw=load_demo()
    except (OSError,ValueError,KeyError,TypeError) as e:st.error('Demo data could not be loaded: '+str(e));return
    st.subheader('Illustrative Information-Seeking results')
    st.warning('Generated examples, not collected behavior. All 34 cases demonstrate this optional activity; the real study sampling plan is unchanged.')
    # awareml_generated_information_group_summary_7_12_15
    group_counts = Counter(p['experience_group'] for p in raw['participants'])
    st.caption('Generated case groups: {0} students | {1} practitioners | {2} ML / AutoML experts'.format(group_counts['University / CS student'], group_counts['Practitioner'], group_counts['ML / AutoML expert']))
    rows=raw['information_seeking'];cards=st.columns(4)
    for col,label,value in zip(cards,['Demo cases','Scripted questions','Evidence accesses','Repeated accesses'],[len(rows),sum(r['question_count'] for r in rows),sum(r['evidence_view_count'] for r in rows),sum(r['revisit_count'] for r in rows)]):col.metric(label,value)
    st.subheader('1  Evidence access and revisits')
    views=Counter();repeat=Counter()
    for r in rows:
        counts=Counter(r['evidence_sequence']);views.update(counts);repeat.update({k:max(0,v-1) for k,v in counts.items()})
    areas=['Performance','Fairness','Sustainability','Drift','Explainability'];fig=go.Figure()
    fig.add_trace(go.Bar(x=areas,y=[views[a] for a in areas],name='All accesses',marker_color='#2563eb'))
    fig.add_trace(go.Bar(x=areas,y=[repeat[a] for a in areas],name='Repeat accesses',marker_color='#d58033'))
    fig.update_layout(barmode='group');fig.update_yaxes(title='Scripted accesses',dtick=5);_chart(fig,'compact_info_access',390)
    st.subheader('2  Evidence views and follow-up questions')
    fig=go.Figure()
    for action in ['Accept','Override','Reject']:
        group=[r for r in rows if r['recommendation_action'].startswith(action)]
        fig.add_trace(go.Scatter(x=[r['evidence_view_count'] for r in group],y=[r['question_count'] for r in group],mode='markers',name=action,text=[r['participant_code'] for r in group],marker=dict(size=12),hovertemplate='%{text}<br>Views %{x}<br>Questions %{y}<extra></extra>'))
    fig.update_xaxes(title='Scripted evidence views',dtick=1);fig.update_yaxes(title='Scripted questions',dtick=1);_chart(fig,'compact_info_depth',390)
    st.caption('Cases at the same coordinates may overlap. Exact records are available below. Zero questions is a valid demonstration path.')
    st.subheader('3  Final recommendation decisions')
    counts=Counter(r['recommendation_action'].split(' — ')[0] for r in rows);actions=['Accept','Override','Reject']
    fig=go.Figure(go.Bar(x=actions,y=[counts[a] for a in actions],text=[counts[a] for a in actions],textposition='outside',marker_color=['#2563eb','#d58033','#bd5d68']))
    fig.update_yaxes(title='Generated cases',dtick=5);_chart(fig,'compact_info_decisions',360)
    st.info('No observed timestamps, durations or human qualitative coding were supplied. Those charts are omitted rather than filled with invented observations.')
    with st.expander('Scripted sequence and final confidence by case'):
        code=st.selectbox('Demo case',[r['participant_code'] for r in rows],key='compact_demo_case')
        r=next(x for x in rows if x['participant_code']==code)
        st.write('Sequence: '+(' → '.join(r['evidence_sequence']) or 'No evidence views'))
        st.write('Generated decision confidence: '+str(r['decision_confidence'])+'/7');st.write(r['rationale']);st.json(r['questions'])
    with st.expander('Fictional context and evidence limitations'):
        ctx=raw['information_seeking_context'];st.write(ctx['scenario']);st.write(ctx['explanation']);st.dataframe(pd.DataFrame(ctx['ranking']),use_container_width=True,hide_index=True);st.json(ctx['supplementary_evidence'])
    _downloads(raw,'information')
