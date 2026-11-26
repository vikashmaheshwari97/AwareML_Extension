import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from awareml.studies.trust_morten_v2 import TrustCalibrationStudyV2, StudyV2Error, StimulusBankV2, analyze_v2
from awareml.studies import information_seeking_morten_v2 as info
from awareml.ui_v2.phase16_morten_v2 import PFX as TRUST_PFX
from awareml.ui_v2.phase17_morten_v2 import PFX as INFORMATION_PFX
from awareml.studies.morten_review_v3 import (
    bank_rows, validate_claims, review_template, review_status, review_identity, scenario_coverage,
    paired_inference, evidence_association, agency_evidence_association, study_ranking,
)


def study(tmp_path):
    return TrustCalibrationStudyV2(db_path=tmp_path/'v3.sqlite',id_salt='test-private-salt-0123456789')


def register(s,code='TEST001'):
    return s.register(code,'Practitioner',4,True,True,'pilot')['participant_hash']


def lock(s,ph,order,rating=4):
    s.lock_initial(ph,order,{k:rating for k in ['stias_confident','stias_reliable','stias_trust']},rating,4,4,'pilot')


def finish(s,ph,order,**kw):
    return s.final_submit(ph,order,'Accept',4,[],'','pilot',explanation_action=kw.pop('explanation_action','Use as shown'),scope_basis='Included study objectives',**kw)


def results():
    return [dict(framework=f,accuracy=.7+i*.02,runtime_sec=10-i,energy_kwh=.001+i*.0001,co2_kg=.0004+i*.00001) for i,f in enumerate(['AutoStreamML','AutoClass','EvoAutoML','OAML','ChaCha'])]


def test_exact_numeric_manipulation_and_fixed_recommendation():
    rows=bank_rows();assert len(rows)==40 and validate_claims()==[]
    for a,b in zip(rows[::2],rows[1::2]):
        assert a['recommendation']==b['recommendation']
        assert a['evidence_summary']==b['evidence_summary']
        assert a['prompt']==b['prompt']
        assert sum(not c['supported'] for c in b['claims'])==1
        # Each changed numerical claim is visible with the same precision used by the card.
        assert a['explanation']!=b['explanation']


def test_counterbalance_and_persistent_resume(tmp_path):
    s=study(tmp_path);a=register(s,'AAA');b=register(s,'BBB')
    aa=s.store.assignment_rows('pilot',a);bb=s.store.assignment_rows('pilot',b)
    assert len(aa)==20 and sum(x['correctness_condition']=='correct' for x in aa)==10
    da={x['pair_id']:x['correctness_condition'] for x in aa}
    assert all(da[x['pair_id']]!=x['correctness_condition'] for x in bb)
    assert register(s,'AAA')==a and s.store.assignment_rows('pilot',a)==aa
    for fw in {x['framework_context'] for x in aa}:
        assert sum(x['framework_context']==fw and x['correctness_condition']=='correct' for x in aa)==2


def test_initial_lock_required_immutable_and_final_judgments_separate(tmp_path):
    s=study(tmp_path);ph=register(s)
    with pytest.raises(StudyV2Error):finish(s,ph,1)
    lock(s,ph,1)
    with pytest.raises(StudyV2Error,match='already locked'):lock(s,ph,1)
    with pytest.raises(StudyV2Error):finish(s,ph,1,explanation_action=None)
    finish(s,ph,1,explanation_action='Correct before use',explanation_targets=['Runtime'],explanation_correction='Use the runtime in the supplied reference row.')
    r=s.store.response_rows('pilot')[0]
    assert r['decision_action']=='Accept' and r['explanation_action']=='Correct before use'
    assert r['decision_target']=='framework_recommendation' and r['recommendation_correct']==1
    with pytest.raises(StudyV2Error):finish(s,ph,1)


def test_no_fake_human_reviews_and_stale_hashes_rejected(tmp_path):
    assert not review_status(pd.DataFrame())['ready']
    frames=[]
    # Explicit synthetic test fixtures, never delivered as completed human reviews.
    mapping={review_identity(r):r for r in bank_rows()}
    for assessor in ['synthetic-test-A','synthetic-test-B']:
        f=review_template(assessor)
        f['recommendation_verdict']='Supported'
        f['explanation_verdict']=f.review_id.map(lambda i:'Supported' if mapping[i]['researcher_label']=='known_correct' else 'Incorrect')
        f['claim_or_reason']='Synthetic test fixture only';f['reference_location']='Synthetic test row';f['scope_clear']='Yes'
        frames.append(f)
    f=pd.concat(frames)
    assert review_status(f)['ready']
    f.iloc[0,f.columns.get_loc('explanation_verdict')]='Insufficient evidence'
    assert not review_status(f)['ready']
    f['bank_sha256']='stale'
    assert not review_status(f)['ready']


def test_final_collection_blocks_pending_exact_review(tmp_path,monkeypatch):
    monkeypatch.setenv('AWAREML_PHASE16_FINAL_ARMED','YES')
    monkeypatch.setattr('awareml.studies.morten_review_v3.REVIEWS',tmp_path/'missing.csv')
    s=study(tmp_path)
    with pytest.raises(StudyV2Error,match='review is pending'):s.register('FINAL001','Practitioner',4,True,True,'final')


def test_full_sixty_scenario_coverage_includes_zeroes():
    empty=scenario_coverage(pd.DataFrame());assert len(empty)==60
    assert sum(r['included_in_track2'] for r in empty)==20
    assert sum(r['expected_N34'] for r in empty)==680
    assert sum(r['expected_N40'] for r in empty)==800
    assert sum(r['expected_N34']==0 for r in empty)==40


def test_primary_inference_matches_hand_computed_gap_and_dz():
    a=np.array([1.,2.,3.]);out=paired_inference(a,iterations=1000)
    assert out['mean_gap']==2 and out['cohen_dz']==2
    assert out['p_two_sided']==.25
    assert out['ci95'][0]<=2<=out['ci95'][1]
    assert paired_inference([2])['cohen_dz'] is None


def test_evidence_and_agency_associations_use_correct_units():
    frame=pd.DataFrame([dict(participant_hash=p,evidence_question_count=q,decision_action=a,explanation_action=e,perceived_correctness=r) for p,q,a,e,r in [('a',1,'Accept','Correct before use',2),('a',0,'Accept','Use as shown',5),('b',0,'Reject','Do not use',3),('c',1,'Accept','Use as shown',6)]])
    out=evidence_association(frame,iterations=10)
    assert out['recommendation_accept']['risk_difference_evidence_minus_no_evidence']==.5
    sec=[dict(participant_hash=p,ratings_json=json.dumps({'critical_evaluation':v})) for p,v in [('a',4),('b',2),('c',7)]]
    result=agency_evidence_association(frame,sec,iterations=10)
    assert result['n_participants']==3 and result['spearman_rho']>.99


def test_information_score_ignores_unscored_objectives_and_missing_rejected():
    rows,meta=study_ranking(results(),{'ground_truth':'Accuracy|Runtime'})
    assert rows[0]['framework']=='ChaCha' and meta['objective_weights']=={'Accuracy':.5,'Runtime':.5}
    assert rows[0]['scenario_fit_score']==rows[0]['utility']
    with pytest.raises(ValueError):study_ranking(results(),{'ground_truth':'Privacy'})
    with pytest.raises(ValueError):study_ranking(results(),{'ground_truth':'Accuracy'},weights={'Accuracy':1,'Fairness':1})
    r=results();r[0]['accuracy']=None
    with pytest.raises(ValueError):study_ranking(r,{'ground_truth':'Accuracy'})


def test_information_context_hash_roundtrip_and_tampering(tmp_path,monkeypatch):
    monkeypatch.setattr(info,'CONTEXT_DIR',tmp_path)
    ctx=info.save_context('pilot',results(),[{'framework':'wrong','utility':999}],{'ground_truth':'Accuracy'},True)
    assert ctx['ranking'][0]['framework']=='ChaCha'
    assert info.load_context('pilot')['context_id']==ctx['context_id']
    path=info.context_path('pilot');d=json.loads(path.read_text());d['results'][0]['accuracy']=.99;path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='integrity'):info.load_context('pilot')


def test_final_information_context_cannot_change_after_start(tmp_path,monkeypatch):
    monkeypatch.setattr(info,'CONTEXT_DIR',tmp_path)
    from awareml.studies.store import StudyStore
    monkeypatch.setattr(StudyStore,'export',lambda self,name:pd.DataFrame([{'event_type':'session_start'}]))
    with pytest.raises(ValueError,match='cannot be replaced'):info.save_context('final',results(),[],{'ground_truth':'Accuracy'},True)


def test_complete_twenty_trial_analysis(tmp_path):
    s=study(tmp_path);ph=register(s)
    for i in range(1,21):
        trial=s.current_trial(ph,'pilot')
        a=s.store.assignment_rows('pilot',ph)[i-1]
        lock(s,ph,i,5 if a['correctness_condition']=='correct' else 2)
        finish(s,ph,i)
    assert s.store.participant('pilot',ph)['completed_at'] is not None
    out=analyze_v2(s.store,'pilot')
    assert out['completed']==1 and out['trust_gap']['mean']==3
    assert len(out['all_scenario_coverage'])==60


def test_shared_card_and_no_ambiguous_action_wording():
    trust=Path('awareml/ui_v2/phase16_morten_v2.py').read_text();assessor=Path('assessor_review_app.py').read_text()
    assert 'render_stimulus_card(trial)' in trust and 'render_stimulus_card(trial)' in assessor
    assert 'this recommendation / explanation' not in trust
    assert 'What would you do with this explanation?' in trust
    assert 'What would you do with this recommendation?' in trust
    text=Path('awareml/ui_v2/phase17_morten_v2.py').read_text()
    assert 'Initial AwareML recommendation' not in text and '"Utility"' not in text


def _isolate_ui_runtime(monkeypatch):
    # Synthetic UI tests must not read workstation Main Study settings or use live data.
    from awareml.studies import study_runtime
    monkeypatch.setattr(study_runtime, 'load_local_runtime', lambda *args, **kwargs: {})
    monkeypatch.setenv('AWAREML_PHASE16_COLLECTION_MODE', 'pilot')
    monkeypatch.setenv('AWAREML_INFORMATION_SEEKING_COLLECTION_MODE', 'pilot')
    monkeypatch.setenv('AWAREML_PHASE16_FINAL_ARMED', '')
    monkeypatch.setenv('AWAREML_INFORMATION_SEEKING_FINAL_ARMED', '')


def test_streamlit_participant_can_accept_choice_and_correct_explanation(tmp_path,monkeypatch):
    _isolate_ui_runtime(monkeypatch)
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('AWAREML_PHASE16_V3_DB',str(tmp_path/'ui.sqlite'))
    at=AppTest.from_string('from awareml.ui_v2.phase16_morten_v2 import render_phase16_participant_study\nrender_phase16_participant_study(collection_mode="pilot", isolated=True)').run(timeout=30)
    assert not at.exception
    at.text_input(key=TRUST_PFX + '_reg_code').set_value('UIV8TEST')
    at.selectbox(key=TRUST_PFX + '_reg_group').select('Practitioner')
    at.radio(key=TRUST_PFX + '_reg_expertise').set_value(4)
    for box in at.checkbox:box.check()
    at.button[0].click().run(timeout=30)
    assert len(at.radio)==6 and all(r.value is None for r in at.radio)
    assert not any(r.label=='What would you do with this recommendation?' for r in at.radio)
    for radio in at.radio:radio.set_value(4)
    at.button[0].click().run(timeout=30)
    assert not at.exception
    assert all(r.value is None for r in at.radio)
    at.radio(key='p16v3_explanation_action_1').set_value('Correct before use').run()
    at.multiselect(key='p16v3_ex_targets_1').set_value(['Runtime'])
    at.text_area(key='p16v3_ex_correction_1').set_value('Use the runtime recorded in the reference row.')
    at.radio(key='p16v3_decision_1').set_value('Accept')
    at.radio(key='p16v3_scope_basis_1').set_value('Included study objectives')
    at.button(key='p16v3_submit_1').click().run(timeout=30)
    # Streamlit 1.40 AppTest may retain elements from the preceding rerun in this
    # immediate tree. Verify the actual stored response and the specific next-trial
    # widgets instead of the total, which includes stale preceding-trial elements.
    assert not at.exception
    from awareml.studies.trust_morten_v2 import Phase16V2Store
    store=Phase16V2Store(tmp_path/'ui.sqlite')
    rows=store.response_rows('pilot')
    assert len(rows)==1
    row=rows[0]
    assert row['order_index']==1 and row['decision_action']=='Accept'
    assert row['explanation_action']=='Correct before use'
    assert row['explanation_correction']=='Use the runtime recorded in the reference row.'
    assert store.current_order('pilot',row['participant_hash'])==2
    assert store.initial('pilot',row['participant_hash'],2) is None
    for key in ['a','b','c','pc','fl','cf']:
        assert at.radio(key='p16v3_'+key+'_2').value is None
    assert not any(r.key=='p16v3_decision_2' for r in at.radio)


def test_information_participant_scope_and_fixed_score_ui(tmp_path,monkeypatch):
    _isolate_ui_runtime(monkeypatch)
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(info,'CONTEXT_DIR',tmp_path)
    from awareml.config import settings
    from dataclasses import replace
    monkeypatch.setattr('awareml.studies.store.settings',replace(settings,study_db=tmp_path/'info.sqlite'))
    info.save_context('pilot',results(),[],{'ground_truth':'Accuracy','scenario':'Plate recognition accuracy and privacy.','scenario_id':'DEMO','benchmark_id':'DEMO','domain':'Transport'},True)
    at=AppTest.from_string('from awareml.ui_v2.phase17_morten_v2 import information_seeking_research_page_v2\ninformation_seeking_research_page_v2()').run(timeout=30)
    at.button(key=INFORMATION_PFX + '_enterp').click().run(timeout=30)
    at.run(timeout=30)
    assert not at.exception
    at.text_input(key=INFORMATION_PFX + '_reg_code').set_value('INFOV8TEST')
    at.selectbox(key=INFORMATION_PFX + '_reg_group').select('Practitioner')
    at.radio(key=INFORMATION_PFX + '_reg_expertise').set_value(4)
    for box in at.checkbox:box.check()
    at.button(key='FormSubmitter:p17v3_registration-Start information-seeking study').click().run(timeout=30)
    assert not at.exception
    labels=[x.value for x in at.markdown]
    assert any('AwareML recommendation' in t for t in labels)
    assert not any('Initial AwareML recommendation' in t for t in labels)
    assert 'Scenario-fit score' in at.dataframe[0].value.columns
    assert all(r.value is None for r in at.radio)
