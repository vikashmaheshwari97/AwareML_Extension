from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_visual_module_exists():
    s=(ROOT/'awareml'/'ui_v2'/'phase16_research_visuals.py').read_text(encoding='utf-8')
    assert 'def render_trust_calibration_visuals' in s
    assert 'def render_secondary_visuals' in s
    assert 'Calibration overview' in s and 'Reliance behavior' in s and 'Expertise & sources' in s and 'Style & timing' in s
    assert 'Likert response distribution' in s
def test_trust_ui_hooks():
    s=(ROOT/'awareml'/'ui_v2'/'phase16_trust_calibration.py').read_text(encoding='utf-8')
    assert 'render_trust_calibration_visuals(result, responses=responses)' in s
    assert '#### Condition summary table' in s
    assert 'st.dataframe(summary_frame' in s
def test_secondary_ui_hooks():
    s=(ROOT/'awareml'/'ui_v2'/'phase16_secondary_questionnaire.py').read_text(encoding='utf-8')
    assert 'render_secondary_visuals(summary, rows)' in s
    assert '#### Detailed item table' in s
    assert 'st.dataframe(item_frame' in s and 'st.dataframe(by_expertise' in s
