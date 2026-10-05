from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

SECONDARY_SCHEMA_VERSION = "phase16_secondary_hcai_v1"

SECONDARY_ITEMS = [
    {"id": "critical_evaluation", "construct": "human_agency",
     "text": "I felt able to critically evaluate the recommendation provided by AwareML."},
    {"id": "freedom_to_override", "construct": "human_agency",
     "text": "I felt free to reject or override AwareML's recommendation when I disagreed with it."},
    {"id": "final_control", "construct": "human_agency",
     "text": "I understood that the final model-selection decision remained under my control."},
    {"id": "rely_or_question", "construct": "calibrated_reliance_support",
     "text": "The explanations helped me judge when to rely on the recommendation and when to question it."},
    {"id": "evidence_checking", "construct": "calibrated_reliance_support",
     "text": "I compared the AI explanation with the reference evidence before deciding whether to rely on it."},
    {"id": "explanation_usefulness", "construct": "explainability",
     "text": "The explanations were useful for understanding why the system made its recommendation."},
    {"id": "interface_usability", "construct": "usability",
     "text": "The study interface made it easy to compare an explanation with its supporting evidence."},
    {"id": "real_world_support_trust", "construct": "global_reliance",
     "text": "I would trust AwareML to support me in a real-world streaming AutoML decision."},
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path), timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    return con


def ensure_secondary_table(db_path: Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS phase16_secondary_questionnaire (
                collection_mode TEXT NOT NULL,
                participant_hash TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                ratings_json TEXT NOT NULL,
                open_comment TEXT,
                submitted_at TEXT NOT NULL,
                PRIMARY KEY(collection_mode, participant_hash)
            )
            """
        )


def has_secondary_response(db_path: Path, collection_mode: str, participant_hash: str) -> bool:
    ensure_secondary_table(db_path)
    with _connect(Path(db_path)) as con:
        row = con.execute(
            "SELECT 1 FROM phase16_secondary_questionnaire WHERE collection_mode=? AND participant_hash=?",
            (str(collection_mode), str(participant_hash)),
        ).fetchone()
    return row is not None


def save_secondary_response(
    db_path: Path,
    collection_mode: str,
    participant_hash: str,
    ratings: Dict[str, int],
    open_comment: str = "",
) -> None:
    ensure_secondary_table(db_path)
    expected = {item["id"] for item in SECONDARY_ITEMS}
    supplied = set(ratings)
    if supplied != expected:
        raise ValueError(
            "Secondary questionnaire item mismatch; missing={}, extra={}".format(
                sorted(expected - supplied), sorted(supplied - expected)
            )
        )
    clean: Dict[str, int] = {}
    for item_id, value in ratings.items():
        value = int(value)
        if value < 1 or value > 7:
            raise ValueError("{} must be between 1 and 7".format(item_id))
        clean[str(item_id)] = value

    with _connect(Path(db_path)) as con:
        try:
            con.execute(
                """
                INSERT INTO phase16_secondary_questionnaire(
                    collection_mode, participant_hash, schema_version,
                    ratings_json, open_comment, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(collection_mode),
                    str(participant_hash),
                    SECONDARY_SCHEMA_VERSION,
                    json.dumps(clean, sort_keys=True, separators=(",", ":")),
                    str(open_comment or "").strip()[:4000],
                    _utc_now(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "This post-study questionnaire has already been submitted; responses are append-only."
            ) from exc


def secondary_rows(db_path: Path, collection_mode: str) -> List[Dict[str, Any]]:
    ensure_secondary_table(db_path)
    with _connect(Path(db_path)) as con:
        rows = con.execute(
            """
            SELECT collection_mode, participant_hash, schema_version,
                   ratings_json, open_comment, submitted_at
            FROM phase16_secondary_questionnaire
            WHERE collection_mode=?
            ORDER BY submitted_at, participant_hash
            """,
            (str(collection_mode),),
        ).fetchall()

    output: List[Dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        try:
            ratings = json.loads(rec.pop("ratings_json"))
        except Exception:
            ratings = {}
            rec.pop("ratings_json", None)
        rec.update(ratings)
        output.append(rec)
    return output


def _mean(values: List[Any]) -> Optional[float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return None if arr.empty else float(arr.mean())


def secondary_summary(db_path: Path, collection_mode: str) -> Dict[str, Any]:
    rows = secondary_rows(db_path, collection_mode)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"responses": 0, "item_means": {}, "construct_means": {}, "by_expertise": []}

    item_means = {
        item["id"]: _mean(frame[item["id"]].tolist()) if item["id"] in frame.columns else None
        for item in SECONDARY_ITEMS
    }
    constructs: Dict[str, List[str]] = {}
    for item in SECONDARY_ITEMS:
        constructs.setdefault(item["construct"], []).append(item["id"])

    construct_means: Dict[str, Optional[float]] = {}
    for construct, item_ids in constructs.items():
        cols = [c for c in item_ids if c in frame.columns]
        construct_means[construct] = (
            None if not cols
            else float(frame[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).mean())
        )

    with _connect(Path(db_path)) as con:
        p_rows = con.execute(
            "SELECT participant_hash, expertise_group FROM participants WHERE collection_mode=?",
            (str(collection_mode),),
        ).fetchall()
    p_map = {str(r["participant_hash"]): str(r["expertise_group"]) for r in p_rows}
    frame["expertise_group"] = frame["participant_hash"].astype(str).map(p_map)

    by_expertise: List[Dict[str, Any]] = []
    for expertise, group in frame.groupby("expertise_group", dropna=False):
        rec: Dict[str, Any] = {
            "expertise_group": "Unknown" if pd.isna(expertise) else str(expertise),
            "participants": int(group["participant_hash"].nunique()),
        }
        for construct, item_ids in constructs.items():
            cols = [c for c in item_ids if c in group.columns]
            rec[construct] = (
                None if not cols
                else float(group[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).mean())
            )
        by_expertise.append(rec)

    return {
        "responses": int(len(frame)),
        "item_means": item_means,
        "construct_means": construct_means,
        "by_expertise": by_expertise,
    }
