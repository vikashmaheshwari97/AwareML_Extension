from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_visual_module_contains_upgraded_views():
    s = (ROOT / "awareml/ui_v2/phase16_research_visuals.py").read_text(encoding="utf-8")
    assert "Likert response distribution" in s
    assert "Novice / practitioner / expert comparison" in s
    assert "Trust discrimination by explanation source" in s
    assert "Framework recommendation" in s
    assert "Feature-attribution explanation" in s


def test_protocol_consent_text_is_shortened_and_single_contact():
    for rel in [
        "data/journal/trust_calibration_phase16_v1/design/protocol.json",
        "data/journal/information_seeking_v1/design/protocol.json",
    ]:
        data = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        consent = (data.get("participant_materials") or {}).get("consent_text") or ""
        assert "**Study:** AwareML Trust Calibration and Human Oversight Study" in consent
        assert "**Institution:** University of Tartu, Institute of Computer Science" in consent
        assert "**Research contact:** Vikash Maheshwari — maheshwari@ut.ee" in consent
        assert "andmekaitse@ut.ee" not in consent
        assert "eetikakomitee@ut.ee" not in consent
        assert consent.count("maheshwari@ut.ee") >= 3
