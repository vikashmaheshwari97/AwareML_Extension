from pathlib import Path
import json
import sqlite3

from awareml.studies.phase16_secondary_questionnaire import (
    SECONDARY_ITEMS,
    has_secondary_response,
    save_secondary_response,
    secondary_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def test_secondary_questionnaire_contains_meeting_agency_items():
    texts = {item["text"] for item in SECONDARY_ITEMS}
    assert "I felt able to critically evaluate the recommendation provided by AwareML." in texts
    assert "I felt free to reject or override AwareML's recommendation when I disagreed with it." in texts
    assert "I understood that the final model-selection decision remained under my control." in texts


def test_secondary_store_is_append_only_and_summarizes(tmp_path):
    db = tmp_path / "phase16.sqlite"
    with sqlite3.connect(str(db)) as con:
        con.execute(
            "CREATE TABLE participants (collection_mode TEXT, participant_hash TEXT, expertise_group TEXT)"
        )
        con.execute("INSERT INTO participants VALUES (?,?,?)", ("pilot", "p1", "Novice / student"))

    ratings = {item["id"]: 6 for item in SECONDARY_ITEMS}
    save_secondary_response(db, "pilot", "p1", ratings, "helpful")
    assert has_secondary_response(db, "pilot", "p1")
    summary = secondary_summary(db, "pilot")
    assert summary["responses"] == 1
    assert abs(summary["construct_means"]["human_agency"] - 6.0) < 1e-12


def test_phase16_ui_contains_secondary_hooks():
    source = (ROOT / "awareml" / "ui_v2" / "phase16_trust_calibration.py").read_text(encoding="utf-8")
    assert "render_phase16_secondary_participant" in source
    assert "render_phase16_secondary_researcher" in source


def test_mainstudy_secondary_questionnaire_is_finalized():
    source = (ROOT / "awareml" / "studies" / "trust.py").read_text(encoding="utf-8")
    assert "secondary_poststudy.status must be final" in source
    protocol = json.loads(
        (ROOT / "data" / "journal" / "trust_calibration_phase16_v1" / "design" / "protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["secondary_poststudy"]["status"] == "final"
    assert protocol["recruitment_plan"]["status"] == "provisional_pending_power_calculation"
    assert protocol["randomization"]["version"] == "phase16_balanced_within_subject_v2_judgeable"
