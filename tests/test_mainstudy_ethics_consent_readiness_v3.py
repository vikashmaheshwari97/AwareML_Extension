from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]


def test_phase16_protocol_scientifically_final_but_ethics_not_fabricated():
    p = ROOT / "data/journal/trust_calibration_phase16_v1/design/protocol.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["status"] == "final"
    assert data["participant_materials"]["status"] == "final"
    assert data["participant_materials"]["minimum_age"] == 18
    ethics = data["ethics"]
    assert ethics["status"] == "research_team_self_assessed"
    assert ethics["institutional_approval_claimed"] is False
    assert ethics["institutional_review_status"] == "not_claimed"
    assert ethics["risk_classification"] == "minimal_risk_behavioral_user_study"


def test_phase17_methodologically_final():
    p = ROOT / "data/journal/information_seeking_v1/design/protocol.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["status"] == "final_ready"
    assert data["participant_target"]["final_target"] == 12
    assert data["participant_materials"]["status"] == "final"


def test_visual_padding_hotfix():
    p = ROOT / "awareml/ui_v2/phase16_research_visuals.py"
    if not p.exists():
        return
    source = p.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'padding={"left": 26, "right": 26, "top": 18, "bottom": 22}' not in source


def test_real_ethics_reference_is_required():
    p = ROOT / "scripts/record_ethics_and_freeze_mainstudy.py"
    source = p.read_text(encoding="utf-8")
    ast.parse(source)
    assert "--reference" in source
    assert "real written determination/reference is required" in source


def test_ethics_documents_exist():
    for rel in [
        "docs/ethics/PHASE16_17_ETHICS_SELF_ASSESSMENT.md",
        "docs/ethics/PARTICIPANT_INFORMATION_MAIN_STUDY.md",
        "docs/ethics/CONSENT_RECORD_MAIN_STUDY.md",
        "docs/ethics/PARTICIPANT_DEBRIEF_MAIN_STUDY.md",
    ]:
        assert (ROOT / rel).exists()
