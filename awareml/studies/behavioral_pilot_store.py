"""Small transactional store for the supplementary pilot, isolated from Phase 16."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .behavioral_calibration import pilot_assignment, validate_pilot_bank


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class BehavioralPilotStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS participants (id TEXT PRIMARY KEY, bank_hash TEXT NOT NULL, assignment TEXT NOT NULL, created TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS trials (participant_id TEXT, trial_id TEXT, initial_json TEXT NOT NULL, final_json TEXT, PRIMARY KEY(participant_id,trial_id))")

    def connect(self):
        return sqlite3.connect(str(self.path), timeout=15)

    def register(self, participant_id, bank):
        validate_pilot_bank(bank)
        if not participant_id or len(participant_id) > 80:
            raise ValueError("Use a pseudonymous participant code of 1–80 characters.")
        digest = hashlib.sha256(json.dumps(bank, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT bank_hash, assignment FROM participants WHERE id=?", (participant_id,)).fetchone()
            if row:
                if row[0] != digest:
                    raise ValueError("This participant was registered with a different bank; do not mix versions.")
                return json.loads(row[1])
            index = db.execute("SELECT count(*) FROM participants").fetchone()[0]
            assignment = pilot_assignment(bank, index)
            db.execute("INSERT INTO participants VALUES (?,?,?,?)", (participant_id, digest, json.dumps(assignment), utc_now()))
        return assignment

    def get_trial(self, participant_id, trial_id):
        with self.connect() as db:
            row = db.execute("SELECT initial_json, final_json FROM trials WHERE participant_id=? AND trial_id=?", (participant_id, trial_id)).fetchone()
        return (json.loads(row[0]), json.loads(row[1]) if row[1] else None) if row else (None, None)

    def save_initial(self, participant_id, trial_id, response):
        with self.connect() as db:
            db.execute("INSERT INTO trials VALUES (?,?,?,NULL)", (participant_id, trial_id, json.dumps(response)))

    def save_final(self, participant_id, trial_id, response):
        with self.connect() as db:
            cursor = db.execute("UPDATE trials SET final_json=? WHERE participant_id=? AND trial_id=? AND final_json IS NULL",
                                (json.dumps(response), participant_id, trial_id))
            if cursor.rowcount != 1:
                raise ValueError("An initial response is required and a final response can only be submitted once.")

    def export_participant(self, participant_id):
        with self.connect() as db:
            rows = db.execute("SELECT final_json FROM trials WHERE participant_id=? AND final_json IS NOT NULL ORDER BY trial_id", (participant_id,)).fetchall()
        return [json.loads(r[0]) for r in rows]
