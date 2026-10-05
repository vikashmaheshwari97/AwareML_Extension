from pathlib import Path
import json
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_phase16_internal_ethics_is_truthful():
    p = ROOT / "data/journal/trust_calibration_phase16_v1/design/protocol.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    ethics = data["ethics"]
    assert ethics["status"] == "research_team_self_assessed"
    assert ethics["institutional_approval_claimed"] is False
    assert ethics["institutional_review_status"] == "not_claimed"
    assert ethics["risk_classification"] == "minimal_risk_behavioral_user_study"


def test_phase17_internal_ethics_is_truthful():
    p = ROOT / "data/journal/information_seeking_v1/design/protocol.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["ethics"]["status"] == "research_team_self_assessed"
    assert data["ethics"]["institutional_approval_claimed"] is False
    assert "research_team_self_assessed" in data["finalization_gate"]["ethics_status_allowed"]
    assert data["participant_target"]["final_target"] == 12


def test_phase16_gate_accepts_internal_self_assessment():
    source = (ROOT / "awareml/studies/trust.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "research_team_self_assessed" in source


def test_frozen_designs_exist_after_apply():
    assert (ROOT / "data/journal/trust_calibration_phase16_v1/frozen_design/manifest.json").exists()
    assert (ROOT / "data/journal/information_seeking_v1/frozen/design_manifest.json").exists()
