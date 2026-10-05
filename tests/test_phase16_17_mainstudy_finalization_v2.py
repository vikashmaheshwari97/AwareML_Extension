from pathlib import Path
import json
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_phase16_mainstudy_measure_and_power():
    protocol = json.loads(
        (ROOT / "data/journal/trust_calibration_phase16_v1/design/protocol.json").read_text(encoding="utf-8")
    )
    tm = protocol["trust_measure"]
    assert tm["status"] == "final"
    assert "S-TIAS" in tm["instrument_name"]
    assert len(tm["items"]) == 3
    assert all(not item.get("pilot_only") for item in tm["items"])
    power = protocol["power_calculation"]
    assert power["status"] == "final"
    assert power["required_completed_participants"] == 34
    assert power["recruitment_target"] == 40


def test_phase17_target_is_twelve():
    protocol = json.loads(
        (ROOT / "data/journal/information_seeking_v1/design/protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["participant_target"]["final_target"] == 12
    assert protocol["sampling_strategy"]["target_total"] == 12


def test_admin_gates_are_not_falsely_auto_approved():
    protocol = json.loads(
        (ROOT / "data/journal/trust_calibration_phase16_v1/design/protocol.json").read_text(encoding="utf-8")
    )
    if protocol["ethics"]["status"] == "pending":
        assert protocol["status"] != "final"


def test_visual_module_has_readable_source_names_and_enhanced_layout():
    path = ROOT / "awareml/ui_v2/phase16_research_visuals.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    for label in [
        "Framework recommendation",
        "Fairness explanation",
        "Conversational explanation",
        "Feature-attribution explanation",
        "Novice / practitioner / expert comparison",
        "Likert response distribution",
    ]:
        assert label in source
