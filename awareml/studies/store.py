from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from awareml.config import settings


class StudyStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or settings.study_db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init_db(self):
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study TEXT NOT NULL,
                    session_hash TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    @staticmethod
    def session_hash(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]

    def log(self, study: str, session_id: str, event_type: str, payload: dict[str, Any]):
        with self._connect() as con:
            con.execute(
                "INSERT INTO events(study, session_hash, event_type, payload_json) VALUES (?, ?, ?, ?)",
                (study, self.session_hash(session_id), event_type, json.dumps(payload, default=str)),
            )

    def export(self, study: str):
        import pandas as pd
        with self._connect() as con:
            return pd.read_sql_query(
                "SELECT id, study, session_hash, event_type, payload_json, created_at FROM events WHERE study=? ORDER BY id",
                con,
                params=(study,),
            )
