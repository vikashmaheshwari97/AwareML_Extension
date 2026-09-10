from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .information_seeking import (
    CATEGORY_LABELS,
    derive_representative_pattern,
)


def _payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _events(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame(
            columns=["id", "session_hash", "event_type", "payload_json", "created_at"]
        )
    frame = events.copy()
    if "id" in frame.columns:
        frame = frame.sort_values("id")
    return frame


def session_summaries(events: pd.DataFrame) -> pd.DataFrame:
    frame = _events(events)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "session_hash",
                "completed",
                "experience_group",
                "follow_up_occurred",
                "follow_up_count",
                "first_follow_up_category",
                "time_before_first_follow_up_sec",
                "evidence_viewed",
                "evidence_view_count",
                "information_seeking_occurred",
                "recommendation_decision",
                "final_confidence",
                "representative_pattern",
            ]
        )

    sessions: Dict[str, Dict[str, Any]] = {}
    for _, row in frame.iterrows():
        session_hash = str(row.get("session_hash") or "")
        if not session_hash:
            continue
        event_type = str(row.get("event_type") or "")
        payload = _payload(row.get("payload_json"))
        rec = sessions.setdefault(
            session_hash,
            {
                "session_hash": session_hash,
                "completed": False,
                "experience_group": None,
                "follow_up_occurred": False,
                "follow_up_count": 0,
                "first_follow_up_category": None,
                "time_before_first_follow_up_sec": None,
                "evidence_viewed": [],
                "recommendation_decision": None,
                "final_confidence": None,
                "participant_rationale": None,
            },
        )
        if event_type == "session_start":
            rec["experience_group"] = payload.get("experience_group")
        elif event_type == "chat_turn":
            rec["follow_up_count"] += 1
            rec["follow_up_occurred"] = True
            if rec["first_follow_up_category"] is None:
                rec["first_follow_up_category"] = payload.get("category")
                rec["time_before_first_follow_up_sec"] = payload.get(
                    "time_since_session_start_sec"
                )
        elif event_type == "evidence_view":
            area = payload.get("evidence_area")
            if area and area not in rec["evidence_viewed"]:
                rec["evidence_viewed"].append(area)
        elif event_type == "session_end":
            rec["completed"] = True
            rec["follow_up_occurred"] = bool(
                payload.get("follow_up_occurred", rec["follow_up_occurred"])
            )
            rec["follow_up_count"] = int(
                payload.get("follow_up_count", rec["follow_up_count"]) or 0
            )
            rec["first_follow_up_category"] = payload.get(
                "first_follow_up_category", rec["first_follow_up_category"]
            )
            rec["time_before_first_follow_up_sec"] = payload.get(
                "time_before_first_follow_up_sec",
                rec["time_before_first_follow_up_sec"],
            )
            for area in payload.get("evidence_viewed") or []:
                if area not in rec["evidence_viewed"]:
                    rec["evidence_viewed"].append(area)
            rec["recommendation_decision"] = payload.get("recommendation_decision")
            rec["final_confidence"] = payload.get("final_confidence")
            rec["participant_rationale"] = payload.get("participant_rationale")

    rows: List[Dict[str, Any]] = []
    for rec in sessions.values():
        rec = dict(rec)
        rec["evidence_viewed"] = list(rec.get("evidence_viewed") or [])
        rec["evidence_view_count"] = len(rec["evidence_viewed"])
        rec["information_seeking_occurred"] = bool(
            rec.get("follow_up_occurred") or rec["evidence_view_count"]
        )
        rec["representative_pattern"] = derive_representative_pattern(
            rec.get("follow_up_count") or 0,
            rec.get("first_follow_up_category"),
            rec.get("evidence_viewed"),
            rec.get("recommendation_decision"),
        )
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("session_hash").reset_index(drop=True)


def manual_coding(events: pd.DataFrame) -> pd.DataFrame:
    frame = _events(events)
    if frame.empty:
        return pd.DataFrame(columns=["coded_session_hash", "themes", "coder_note"])
    rows = []
    for _, row in frame.iterrows():
        if str(row.get("event_type")) != "qualitative_code":
            continue
        payload = _payload(row.get("payload_json"))
        coded = str(payload.get("coded_session_hash") or "").strip()
        if not coded:
            continue
        rows.append(
            {
                "event_id": row.get("id"),
                "coded_session_hash": coded,
                "themes": list(payload.get("themes") or []),
                "coder_note": payload.get("coder_note"),
                "created_at": row.get("created_at"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["coded_session_hash", "themes", "coder_note"])
    codes = pd.DataFrame(rows).sort_values("event_id")
    # Keep the latest saved coding decision for each participant session.
    return codes.groupby("coded_session_hash", as_index=False).tail(1).reset_index(drop=True)


def analyze_information_seeking(events: pd.DataFrame) -> Dict[str, Any]:
    frame = _events(events)
    sessions = session_summaries(frame)
    completed = sessions[sessions["completed"] == True].copy() if not sessions.empty else sessions
    codes = manual_coding(frame)

    chat_payloads = []
    evidence_payloads = []
    for _, row in frame.iterrows():
        payload = _payload(row.get("payload_json"))
        if str(row.get("event_type")) == "chat_turn":
            chat_payloads.append(payload)
        elif str(row.get("event_type")) == "evidence_view":
            evidence_payloads.append(payload)

    categories = Counter(
        str(p.get("category") or "other_follow_up") for p in chat_payloads
    )
    requested_topics: Counter = Counter()
    for payload in chat_payloads:
        for topic in payload.get("requested_topics") or []:
            requested_topics[str(topic)] += 1
    first_categories = Counter(
        str(v)
        for v in completed.get("first_follow_up_category", pd.Series(dtype=object)).dropna().tolist()
    )
    decisions = Counter(
        str(v)
        for v in completed.get("recommendation_decision", pd.Series(dtype=object)).dropna().tolist()
    )
    evidence = Counter(
        str(p.get("evidence_area") or "unknown") for p in evidence_payloads
    )
    patterns = Counter(
        str(v)
        for v in completed.get("representative_pattern", pd.Series(dtype=object)).dropna().tolist()
    )

    theme_counts: Counter = Counter()
    if not codes.empty:
        for themes in codes["themes"]:
            for theme in themes or []:
                theme_counts[str(theme)] += 1

    follow_up_rate = (
        float(completed["follow_up_occurred"].mean())
        if len(completed)
        else None
    )
    information_seeking_rate = (
        float(completed["information_seeking_occurred"].mean())
        if len(completed) and "information_seeking_occurred" in completed
        else None
    )
    mean_followups = (
        float(pd.to_numeric(completed["follow_up_count"], errors="coerce").mean())
        if len(completed)
        else None
    )
    first_time = pd.to_numeric(
        completed.get("time_before_first_follow_up_sec", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    median_first_time = float(first_time.median()) if len(first_time) else None
    accept_rate = None
    if len(completed):
        values = completed["recommendation_decision"].astype(str).str.lower()
        accept_rate = float((values == "accept").mean())

    return {
        "status": "ok" if len(completed) else "insufficient_data",
        "registered_sessions": int(len(sessions)),
        "completed_sessions": int(len(completed)),
        "follow_up_rate": follow_up_rate,
        "information_seeking_rate": information_seeking_rate,
        "mean_follow_up_count": mean_followups,
        "median_time_before_first_follow_up_sec": median_first_time,
        "recommendation_acceptance_rate": accept_rate,
        "category_counts": dict(categories),
        "requested_topic_counts": dict(requested_topics),
        "first_follow_up_category_counts": dict(first_categories),
        "evidence_view_counts": dict(evidence),
        "decision_counts": dict(decisions),
        "representative_pattern_counts": dict(patterns),
        "manual_theme_counts": dict(theme_counts),
        "manually_coded_sessions": int(len(codes)),
    }
