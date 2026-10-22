from pathlib import Path
import hashlib, json

def _sha(path: Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def test_no_frozen_track1_wording_in_active_study_ui():
    for rel in ["awareml/ui_v2/phase16_morten_v2.py","awareml/ui_v2/phase17_morten_v2.py"]:
        text=Path(rel).read_text(encoding="utf-8")
        assert "frozen Track-1" not in text

def test_stimulus_bank_has_no_frozen_track1_wording_and_m2117_is_distinct():
    root=Path("data/journal/trust_stimulus_bank_v2/frozen")
    researcher=json.loads((root/"stimuli_researcher.json").read_text(encoding="utf-8"))
    participant=json.loads((root/"stimuli_participant.json").read_text(encoding="utf-8"))
    assert "frozen Track-1" not in json.dumps(researcher,ensure_ascii=False)
    assert "frozen Track-1" not in json.dumps(participant,ensure_ascii=False)
    pair=[r for r in researcher if r.get("pair_id")=="M21_17_F_CHAT"]
    assert len(pair)==2
    correct=next(r for r in pair if r.get("researcher_label")=="known_correct")
    wrong=next(r for r in pair if r.get("researcher_label")=="known_incorrect")
    assert correct["explanation"] != wrong["explanation"]
    assert "ChaCha best fits" in wrong["explanation"]

def test_stimulus_manifest_hashes_are_current():
    root=Path("data/journal/trust_stimulus_bank_v2/frozen")
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for name,meta in manifest["files"].items():
        assert _sha(root/name)==meta["sha256"]

def test_frozen_design_hashes_are_current():
    troot=Path("data/journal/trust_calibration_phase16_v2/frozen_design")
    tm=json.loads((troot/"manifest.json").read_text(encoding="utf-8"))
    assert _sha(troot/"protocol.json")==tm["files"]["protocol.json"]["sha256"]
    assert _sha(Path("data/journal/trust_stimulus_bank_v2/frozen/manifest.json"))==tm["source_stimulus_bank"]["manifest_sha256"]
    iroot=Path("data/journal/information_seeking_v2/frozen")
    im=json.loads((iroot/"design_manifest.json").read_text(encoding="utf-8"))
    assert _sha(iroot/"protocol.json")==im["files"]["protocol.json"]["sha256"]
