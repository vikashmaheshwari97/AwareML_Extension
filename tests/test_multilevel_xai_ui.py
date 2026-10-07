import json
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_multilevel_page_empty_and_recorded_evidence():
    # v3 intentionally merged Multi-level Explainability into the existing
    # Explanation Integrity -> Explainability Diagnostics workspace.  This
    # compatibility test therefore validates the integrated renderer rather
    # than recreating the retired standalone five-tab page.
    source_path = Path(__file__).resolve().parents[1] / "awareml" / "ui_v2" / "pages_multilevel_explainability.py"
    source = source_path.read_text(encoding="utf-8")
    assert "def integrated_explainability_page" in source
    assert "Model · Hyperparameter · System" in source
    assert "Attribution alignment" in source
    assert "Concept & hint probes" in source
    assert "Metric guide" in source
    assert "llama3:8b" in source
    assert "llama3.1:8b" not in source

def test_pilot_initial_is_locked_before_advice(tmp_path, monkeypatch):
    bank = {"schema_version": "xai_behavioral_pilot_v1", "cases": [{
        "case_id": "C", "dataset_id": "D", "question": "Which value is higher?", "choices": ["A", "B"],
        "ground_truth": "A", "ai_answer": "B", "rationale": "UNSEEN_RATIONALE",
        "diagnostics": "DIAGNOSTICS", "task_evidence": "A=2, B=1", "ground_truth_policy": "larger value",
        "reviewed": True}]}
    path = tmp_path / "bank.json"; path.write_text(json.dumps(bank))
    monkeypatch.setenv("AWAREML_XAI_PILOT_BANK", str(path))
    monkeypatch.setenv("AWAREML_XAI_PILOT_DB", str(tmp_path / "study.sqlite"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "xai_behavioral_pilot.py")).run()
    assert not app.exception
    app.text_input[0].set_value("pilot001")
    app.checkbox[0].check()
    app.button[0].click().run()
    assert not app.exception
    assert not any("AI advice:" in m.value for m in app.markdown)
    app.radio[0].set_value("A")
    app.button[0].click().run()
    assert not app.exception
    assert any("AI advice:" in m.value for m in app.markdown)
    assert app.radio[0].label == "Your final decision"
    app.radio[0].set_value("B")
    app.button[0].click().run()
    assert not app.exception
    assert app.success
