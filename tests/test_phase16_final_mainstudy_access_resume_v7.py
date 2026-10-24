from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_bootstrap_is_installed():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "bootstrap_study_runtime" in source


def test_private_runtime_is_final_mode(tmp_path, monkeypatch):
    # CI tests loading with disposable values, never real recruitment secrets.
    from awareml.studies.study_runtime import bootstrap_study_runtime
    path = tmp_path / "private_mainstudy_runtime.json"
    fixture = {
        "AWAREML_PHASE16_COLLECTION_MODE": "final",
        "AWAREML_PHASE16_FINAL_ARMED": "YES",
        "AWAREML_INFORMATION_SEEKING_COLLECTION_MODE": "final",
        "AWAREML_INFORMATION_SEEKING_FINAL_ARMED": "YES",
        "AWAREML_STUDY_RESEARCHER_KEY": "test-only-researcher-key",
        "AWAREML_PHASE16_ID_SALT": "test-only-participant-salt",
    }
    path.write_text(json.dumps(fixture))
    for key in fixture:
        monkeypatch.delenv(key, raising=False)
    data = bootstrap_study_runtime(path)
    import os
    assert all(os.environ[key] == value for key, value in fixture.items())
    assert bootstrap_study_runtime(tmp_path / "missing.json") == {}
    assert data["AWAREML_PHASE16_COLLECTION_MODE"] == "final"
    assert data["AWAREML_PHASE16_FINAL_ARMED"] == "YES"
    assert data["AWAREML_INFORMATION_SEEKING_COLLECTION_MODE"] == "final"
    assert data["AWAREML_INFORMATION_SEEKING_FINAL_ARMED"] == "YES"
    assert len(data["AWAREML_STUDY_RESEARCHER_KEY"]) >= 16
    assert len(data["AWAREML_PHASE16_ID_SALT"]) >= 16


def test_resume_logic_keeps_original_expertise():
    source = (ROOT / "awareml/studies/trust.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert '"participant_resumed"' in source
    assert "stored_expertise_group" in source
    assert "requested_expertise_group" in source
    assert 'expertise_group=str(existing.get("expertise_group"))' in source


def test_ui_uses_simple_researcher_key_wording():
    source = (ROOT / "awareml/ui_v2/phase16_trust_calibration.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "Enter the Researcher Workspace key to unlock the researcher console." in source
