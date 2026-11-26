import json
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from awareml.studies.task_context_revision import (
 resolve_trial,require_information_context,presentation_sha,review_bank_sha,EXPLANATION_LABELS,TRUST_INSTRUCTION)
from awareml.studies.morten_review_v3 import bank_rows,review_identity,review_template,review_status
from awareml.studies.trust_morten_v2 import TrustCalibrationStudyV2

def test_all_40_cards_have_explicit_context_and_remain_blinded():
    for row in bank_rows():
        trial=resolve_trial(row['participant_version'])
        assert trial['scenario_text'].strip() and trial['prompt'].strip()
        assert 'researcher_label' not in trial and 'claims' not in trial

def test_missing_scenario_resolves_from_exact_bank_id_but_unknown_does_not():
    trial=dict(bank_rows()[0]['participant_version']);text=trial.pop('scenario_text')
    assert resolve_trial(trial)['scenario_text']==text
    trial['scenario_id']='NONEXISTENT'
    with pytest.raises(ValueError):resolve_trial(trial)

def test_shared_card_renders_scenario_before_recommendation():
    trial=bank_rows()[0]['participant_version']
    code='from awareml.ui_v2.study_card_v3 import render_stimulus_card\nrender_stimulus_card('+repr(trial)+')'
    app=AppTest.from_string(code).run(timeout=30)
    assert not app.exception
    text=' '.join(str(x.value) for x in app.markdown)
    assert trial['scenario_text'] in text
    assert text.index('Scenario and decision context')<text.index('AI recommendation')
    assert 'Your decision task' in text

def test_unknown_context_blocks_card_before_ratings():
    app=AppTest.from_string('from awareml.ui_v2.study_card_v3 import render_stimulus_card\nrender_stimulus_card({"scenario_id":"MISSING","prompt":"task"})\nimport streamlit as st\nst.radio("Rating",[1,2,3],index=None)').run(timeout=30)
    assert app.error and not app.radio

def test_review_identity_includes_presentation_and_old_hash_cannot_pass():
    from awareml.studies.morten_review_v3 import digest,BANK
    row=bank_rows()[0]
    assert review_identity(row)!='RV-'+digest(row['stimulus_id'])[:12].upper()
    assert len(presentation_sha())==64 and len(review_bank_sha(BANK))==64
    assert not review_status()['ready']

def test_new_registration_uses_separate_version_and_context_reaches_trial(tmp_path):
    s=TrustCalibrationStudyV2(db_path=tmp_path/'study.sqlite',id_salt='test-salt-only-0123456789')
    p=s.register('TEST-CONTEXT','Practitioner',4,True,True,'pilot')
    assert 'task_context_r1' in p['protocol_version']
    trial=s.current_trial(p['participant_hash'],'pilot')
    assert resolve_trial(trial)['scenario_text'].strip()
    assignments=s.store.assignment_rows('pilot',p['participant_hash'])
    assert len(assignments)==20 and sum(a['correctness_condition']=='correct' for a in assignments)==10

def test_information_task_requires_scenario_and_alignment():
    for ctx in [{},{'scenario':{'scenario':'situation'}}]:
        with pytest.raises(ValueError):require_information_context(ctx)
    assert require_information_context({'scenario':{'scenario':'situation'},'scenario_alignment_attested':True})['scenario']=='situation'

def test_information_task_is_visible_and_benchmark_scope_explicit():
    app=AppTest.from_string('from awareml.ui_v2.study_task_context import render_information_task\nrender_information_task({"scenario":{"scenario":"A hospital needs reliable predictions."},"scenario_alignment_attested":True})').run(timeout=30)
    assert not app.exception
    assert any('hospital' in str(x.value) for x in app.markdown)
    assert any('not measurements of deployment' in x.value for x in app.caption)

def test_trust_instructions_and_explanation_choice_have_separate_meanings():
    assert 'recommendation and supporting explanation' in TRUST_INSTRUCTION
    assert 'request a revised explanation' in EXPLANATION_LABELS['Correct before use']
    assert set(EXPLANATION_LABELS)=={'Use as shown','Correct before use','Do not use','Unsure'}

def test_installed_source_scopes_questions_and_preserves_s_tias():
    root=Path(__file__).resolve().parents[1]
    text=(root/'awareml/ui_v2/phase16_morten_v2.py').read_text()
    for item in ['I am confident in the AI assistant.','The AI assistant is reliable.','I can trust the AI assistant.']:
        assert item in text
    assert 'EXPLANATION_LABELS[value]' in text
    assert 'ASSISTANT_INSTRUCTION' in text and 'EXPLANATION_INSTRUCTION' in text
    assert 'SCOPE_LABELS[value]' in text
