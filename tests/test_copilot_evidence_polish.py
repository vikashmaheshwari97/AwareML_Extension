import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from awareml.ui_v2.copilot_polish import interval_figure,explanation_evidence,render_explainer
from awareml.ui_v2.copilot_recommender import PRESETS
from awareml.ui_v2.recommender_bridge import historical_prior
from awareml.ui_v2 import recommender_assets as assets
from test_copilot_current_protocol import make_bundle


def test_real_historical_presets_reproduce_screenshots_and_accuracy_only_changes(tmp_path,monkeypatch):
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'artifacts/copilot_results/20261025_five_frameworks/models'
    if not (root/'manifest.json').is_file():
        pytest.skip('Requires the original local experiment export; synthetic bundle checks run separately.')
    monkeypatch.setenv('AWAREML_RECOMMENDER_BUNDLE',str(root))
    for name,w in PRESETS.items():
        ranking,_=historical_prior(w)
        assert ranking.iloc[0].framework=='OAML'
        assert abs(float(ranking.iloc[0].historical_utility)-(.7815 if name=='Accuracy first' else .8999))<.0001
    ranking,_=historical_prior({'accuracy':1})
    assert ranking.iloc[0].framework=='EvoAutoML'


def sample():
    return pd.DataFrame({'rank':[1,2],'framework':['OAML','ChaCha'],'utility':[.8,.4],
        'accuracy':[.8,.7],'runtime':[4,8],'energy':[.001,.003],'co2':[.0004,.0012],
        'energy_lower':[0,.001],'energy_upper':[.002,.005]})


def test_interval_chart_preserves_bounds_and_uses_readable_units():
    original=sample();before=original.copy(deep=True)
    fig=interval_figure(original,'energy','Energy (Wh)',1000)
    assert list(fig.data[0].x)==[0,2]
    assert list(fig.data[1].x)==[1]
    assert list(fig.data[2].x)==[1,5]
    assert fig.data[3].y[0]=='ChaCha-OVR'
    pd.testing.assert_frame_equal(original,before)


def test_explainer_evidence_has_outcomes_and_weights_without_raw_rows():
    e=explanation_evidence(sample(),{'accuracy':1},{'profile':{'profile_sample_size':1000}})
    assert e['objective_weights']=={'accuracy':1}
    assert len(e['ranked_candidates'])==2
    assert 'profile_observations' in e and 'dataset' not in e


def test_explainer_uses_supported_client_and_hides_stale_explanations(monkeypatch):
    import awareml.llm
    class Client:
        def __init__(self,**kwargs):pass
        def generate_text(self,prompt):
            assert 'not observed framework execution' in prompt
            return 'OAML leads under the active priorities.',{}
    monkeypatch.setattr(awareml.llm,'OllamaClient',Client)
    code='''from test_copilot_evidence_polish import sample
from awareml.ui_v2.copilot_polish import render_explainer
import streamlit as st
state=st.session_state.setdefault('s',{})
render_explainer(state,sample(),{'accuracy':1},{},st.session_state.get('sig','one'))'''
    app=AppTest.from_string(code).run(timeout=30)
    app.button(key='cop_current_explain').click().run()
    assert not app.exception
    assert any('OAML leads' in x.value for x in app.markdown)
    app.session_state['sig']='two';app.run()
    assert not any('OAML leads' in x.value for x in app.markdown)
