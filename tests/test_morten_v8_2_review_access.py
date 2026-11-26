import pandas as pd
from streamlit.testing.v1 import AppTest
from awareml.studies.trust_morten_v2 import TrustCalibrationStudyV2,StudyV2Error
from awareml.studies import study_runtime
from awareml.ui_v2.phase16_morten_v2 import PFX


def test_pending_review_does_not_block_researcher_upload(tmp_path,monkeypatch):
    monkeypatch.setattr(study_runtime,'load_local_runtime',lambda *a,**k:{})
    monkeypatch.setenv('AWAREML_PHASE16_V3_DB',str(tmp_path/'review.sqlite'))
    monkeypatch.setenv('AWAREML_PHASE16_COLLECTION_MODE','final')
    monkeypatch.setenv('AWAREML_STUDY_RESEARCHER_KEY','test-only-review-key')
    def pending(self):raise StudyV2Error('Exact explanation review is pending')
    monkeypatch.setattr(TrustCalibrationStudyV2,'assert_final_ready',pending)
    at=AppTest.from_string('from awareml.ui_v2.phase16_morten_v2 import phase16_trust_calibration_page_v2\nphase16_trust_calibration_page_v2()').run(timeout=30)
    assert not at.exception
    assert at.button(key='p16v3_enter_p').disabled
    assert not at.button(key='p16v3_enter_r').disabled
    at.button(key='p16v3_enter_r').click().run(timeout=30)
    at.text_input(key=PFX + '_researcher_key').set_value('test-only-review-key')
    at.button(key=PFX + '_unlock').click().run(timeout=30)
    assert not at.exception
    assert any(u.label=='Import completed assessor CSV' for u in at.get('file_uploader'))


def test_main_participant_button_enabled_only_when_ready(tmp_path,monkeypatch):
    monkeypatch.setattr(study_runtime,'load_local_runtime',lambda *a,**k:{})
    monkeypatch.setenv('AWAREML_PHASE16_V3_DB',str(tmp_path/'ready.sqlite'))
    monkeypatch.setenv('AWAREML_PHASE16_COLLECTION_MODE','final')
    monkeypatch.setattr(TrustCalibrationStudyV2,'assert_final_ready',lambda self:None)
    at=AppTest.from_string('from awareml.ui_v2.phase16_morten_v2 import phase16_trust_calibration_page_v2\nphase16_trust_calibration_page_v2()').run(timeout=30)
    assert not at.exception
    assert not at.button(key='p16v3_enter_p').disabled


def test_direct_final_participant_entry_keeps_readiness_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(study_runtime,'load_local_runtime',lambda *a,**k:{})
    monkeypatch.setenv('AWAREML_PHASE16_V3_DB',str(tmp_path/'direct.sqlite'))
    def pending(self):raise StudyV2Error('Exact explanation review is pending')
    monkeypatch.setattr(TrustCalibrationStudyV2,'assert_final_ready',pending)
    at=AppTest.from_string('from awareml.ui_v2.phase16_morten_v2 import render_phase16_participant_study\nrender_phase16_participant_study(collection_mode="final",isolated=True)').run(timeout=30)
    assert not at.exception
    assert any('Exact explanation review is pending' in x.value for x in at.error)
    assert not any(x.key==PFX + '_reg_code' for x in at.text_input)
