"""Check genuine exported results separately from synthetic inference fixtures."""
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from awareml.ui_v2 import recommender_assets as assets, recommender_bridge as bridge
from awareml.protocol_v3.common import utilities
from test_copilot_current_protocol import make_bundle,frame

PROJECT=Path(__file__).resolve().parents[1]
RESULTS=PROJECT/'artifacts/copilot_results/20261025_five_frameworks'


@pytest.fixture
def real(monkeypatch,tmp_path):
    if not (RESULTS/'models/manifest.json').is_file():
        pytest.skip('Requires the original local experiment export; not included in a source checkout.')
    monkeypatch.setenv('AWAREML_RECOMMENDER_BUNDLE',str(RESULTS/'models'))
    monkeypatch.delenv('AWAREML_RECOMMENDER_V3_BUNDLE',raising=False)
    for kind in ('DEVELOPMENT','EXTERNAL'):
        monkeypatch.delenv('AWAREML_V3_'+kind+'_EVALUATION',raising=False)
    monkeypatch.setattr(assets,'POINTER',tmp_path/'absent.json')
    return RESULTS/'models'


def test_actual_development_results_work_without_model_binaries(real):
    status=bridge.historical_status()
    assert status['ready']
    assert (status['development_datasets'],status['training_rows'],status['underlying_runs'])==(47,235,1175)
    assert status['seeds']==[42,43,44,45,46]
    assert not list(real.glob('*.joblib'))
    ranked,_=bridge.historical_prior({'accuracy':1})
    assert len(ranked)==5 and ranked.support_datasets.eq(47).all()
    assert np.isfinite(ranked.historical_utility).all()


def test_actual_models_are_not_faked_from_saved_predictions(real):
    assert not bridge.bundle_status()['ready']
    with pytest.raises(RuntimeError):bridge.load_current_recommender()


def test_actual_lodo_and_external_reports_match_exact_model_manifest(real):
    dev=bridge.evaluation_evidence('development',evaluation_directory=RESULTS/'development')
    ext=bridge.evaluation_evidence('external',evaluation_directory=RESULTS/'external')
    assert dev['metadata']['n_datasets']==47
    assert ext['metadata']['n_datasets']==31
    assert ext['metadata']['n_preference_cases_per_method']==3100
    assert dev['metadata']['bundle_manifest_sha256']==ext['metadata']['bundle_manifest_sha256']


def test_corrupted_aggregate_still_rejected_without_models(real,tmp_path):
    copied=tmp_path/'corrupted';shutil.copytree(real,copied)
    with (copied/'development_aggregates.csv').open('a') as handle:handle.write('\nchanged\n')
    assert not bridge.historical_status(copied)['ready']


def test_actual_workspace_has_shared_header_and_available_historical_path(real):
    app=AppTest.from_string('from awareml.ui_v2.copilot_workspace_v5 import render_copilot_workspace_header\nrender_copilot_workspace_header({})').run(timeout=30)
    assert not app.exception
    text=' '.join(x.value for x in app.markdown)
    captions=' '.join(x.value for x in app.caption)
    assert 'r9-hero' in text and 'Goal and Recommendation Panel' in text
    assert '1,175' in captions and '5-seed' in captions
    assert 'Available' in text and 'training pending' not in text.lower()


def test_actual_historical_compare_is_usable_now(real):
    app=AppTest.from_string('from awareml.ui_v2.copilot_recommender import render_historical_preference_prior_tab\nrender_historical_preference_prior_tab()').run(timeout=30)
    assert not app.exception
    assert [m.value for m in app.metric][:4]==['1,175','47','5','5']
    app.button(key='three_hist_run').click().run()
    assert not app.exception
    table=app.dataframe[0].value
    assert len(table)==5 and 'Top-3 rate' in table and 'ChaCha-OVR' in table.Framework.tolist()


def test_actual_dataset_page_shows_results_before_model_transfer(real):
    app=AppTest.from_string('from awareml.ui_v2.copilot_recommender import render_dataset_aware_tab\nrender_dataset_aware_tab()').run(timeout=30)
    assert not app.exception
    assert [m.value for m in app.metric][:4]==['47','235','1,175','31']
    assert any('31 datasets' in c.value for c in app.caption)


def test_goal_comparison_cards_return_with_historical_results(real):
    app=AppTest.from_string('from awareml.ui_v2.copilot_recommender import render_current_goal_guidance\nrender_current_goal_guidance({}, {"primary_weights":{"accuracy":1}}, None)').run(timeout=30)
    assert not app.exception
    text=' '.join(x.value for x in app.markdown)
    assert 'GLOBAL HISTORICAL PRIOR' in text and 'EVIDENCE RELATIONSHIP' in text
    assert '1,175' in text


def test_model_inference_restores_four_diagnostics_tabs_and_human_review(tmp_path,monkeypatch):
    model=make_bundle(tmp_path/'model')
    monkeypatch.setenv('AWAREML_RECOMMENDER_BUNDLE',str(model))
    monkeypatch.setattr(assets,'POINTER',tmp_path/'absent.json')
    from awareml.ui_v2 import copilot_diagnostics as diagnostics
    monkeypatch.setattr(diagnostics,'ROOT',tmp_path)
    app=AppTest.from_string('from awareml.ui_v2.copilot_recommender import render_dataset_aware_tab\nrender_dataset_aware_tab()')
    app.session_state['awareml_state']={'dataset':frame(),'target':'target','dataset_name':'Synthetic test only'}
    app.run(timeout=30);app.button(key='three_dataset_run').click().run()
    assert not app.exception
    assert [t.label for t in app.tabs]==['Why this ranking?','Uncertainty','Sensitivity','Human decision']
    app.radio(key='cop_current_human_decision').set_value('Keep as alternative').run()
    app.button(key='cop_current_save_review').click().run()
    assert not app.exception
    records=(tmp_path/'artifacts/copilot/dataset_aware_recommender_reviews.jsonl').read_text().splitlines()
    saved=json.loads(records[-1]);assert saved['decision']=='Keep as alternative' and saved['evidence_id']


def test_sensitivity_uses_current_utility_and_preserves_predictions():
    from awareml.ui_v2.copilot_diagnostics import sensitivity_table
    ranked=pd.DataFrame({'framework':['ChaCha','OAML'],'accuracy':[.9,.8],'runtime':[20,2],'energy':[.02,.002],'co2':[.008,.0008]})
    original=ranked.copy(deep=True)
    table=sensitivity_table(ranked,{'Accuracy':{'accuracy':1},'Runtime':{'runtime':1}})
    assert table.loc[(table.Profile=='Accuracy')&(table.Rank==1),'Framework'].iloc[0]=='ChaCha'
    assert table.loc[(table.Profile=='Runtime')&(table.Rank==1),'Framework'].iloc[0]=='OAML'
    pd.testing.assert_frame_equal(original,ranked)


def test_patcher_repairs_workspace_entrypoint_without_rewriting_other_functions():
    spec=importlib.util.spec_from_file_location('patch_entry',PROJECT/'scripts/patch_dashboard_labels.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source='# Keep customization\ndef another_page():\n    return 9\ndef copilot_workspace_page():\n    render_goal_copilot_unified_page()\n'
    result=module.patch_source(source)
    assert 'render_copilot_workspace_header(ensure_research_state())' in result
    assert 'def another_page():\n    return 9' in result
    assert module.patch_source(result)==result


def importer():
    spec=importlib.util.spec_from_file_location('import_assets_test',PROJECT/'scripts/import_recommender_dashboard.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_model_package_import_validates_and_preserves_previous_assets(tmp_path):
    model=make_bundle(tmp_path/'synthetic_model')
    archive=tmp_path/'models.zip'
    with zipfile.ZipFile(archive,'w') as package:
        for p in model.iterdir():package.write(p,'models/'+p.name)
    target=tmp_path/'local';(target/'awareml/ui_v2').mkdir(parents=True)
    previous=target/'artifacts/dashboard_evidence';previous.mkdir(parents=True)
    (previous/'keep.csv').write_text('previous local data\n')
    importer().install(archive,target)
    assert bridge.bundle_status(target/'artifacts/dashboard_evidence/models')['ready']
    pointer=json.loads((target/'artifacts/recommender/dashboard_assets.json').read_text())
    assert Path(pointer['bundle'])==target/'artifacts/dashboard_evidence/models'
    backup=next((target/'artifacts/copilot_asset_backups').iterdir())
    assert (backup/'dashboard_evidence/keep.csv').read_text()=='previous local data\n'


def test_incomplete_analysis_archive_is_not_installed_as_a_live_model(real,tmp_path):
    archive=tmp_path/'incomplete.zip'
    with zipfile.ZipFile(archive,'w') as package:
        for p in real.iterdir():package.write(p,'models/'+p.name)
    target=tmp_path/'local';(target/'awareml/ui_v2').mkdir(parents=True)
    with pytest.raises((ValueError,FileNotFoundError)):importer().install(archive,target)
    assert not (target/'artifacts/dashboard_evidence').exists()
