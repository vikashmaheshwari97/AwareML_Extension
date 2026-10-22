from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_PATH = ROOT / "data" / "journal" / "objective_selection_benchmark_v2" / "frozen" / "primary_benchmark.csv"
CONTEXT_DIR = ROOT / "artifacts"
DESIGN_DIR = ROOT / "data" / "journal" / "information_seeking_v2" / "frozen"
CONTEXT_SCHEMA = "information_seeking_context_v2_1"


def protocol() -> dict:
    return json.loads((DESIGN_DIR / "protocol.json").read_text(encoding="utf-8"))


def coding_scheme() -> dict:
    return json.loads((DESIGN_DIR / "coding_scheme.json").read_text(encoding="utf-8"))


def scenario_catalog() -> pd.DataFrame:
    frame = pd.read_csv(SCENARIO_PATH)
    keep = [c for c in ["benchmark_id", "scenario_id", "scenario", "ground_truth", "domain", "source"] if c in frame.columns]
    return frame[keep].head(60).copy()


def context_path(mode: str) -> Path:
    return CONTEXT_DIR / ("information_seeking_context_v2_{}.json".format("final" if mode == "final" else "pilot"))


def _compact_result(row: dict) -> dict:
    return {
        "framework": row.get("framework"),
        "accuracy": row.get("accuracy"),
        "f1_macro": row.get("f1_macro"),
        "runtime_sec": row.get("runtime_sec"),
        "energy_kwh": row.get("energy_kwh"),
        "co2_kg": row.get("co2_kg"),
        "fairness": row.get("fairness") or {},
        "drift_summary": row.get("drift_summary") or {},
        "drift_events": row.get("drift_events") or [],
        "explainability": row.get("explainability") or {},
    }


def save_context(mode: str, results: Sequence[dict], ranking: Sequence[dict], scenario: dict, alignment_attested: bool = False) -> dict:
    path = context_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CONTEXT_SCHEMA,
        "collection_mode": mode,
        "scenario": dict(scenario),
        "scenario_alignment_attested": bool(alignment_attested),
        "evidence_contract": (
            "The Track-1 scenario supplies the decision priorities. The saved AwareML run supplies measured benchmark evidence. "
            "This does not claim that the benchmark dataset is literally the textual scenario."
        ),
        "results": [_compact_result(x) for x in results],
        "ranking": [dict(x) for x in ranking],
    }
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode()
    payload["context_id"] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return payload


def load_context(mode: str) -> Optional[dict]:
    path = context_path(mode)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") in {"information_seeking_context_v2", CONTEXT_SCHEMA}:
        return data
    return None


def mean_sd(values: Sequence[Any]) -> Tuple[Optional[float], Optional[float]]:
    s = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if s.empty:
        return None, None
    return float(s.mean()), (None if len(s) < 2 else float(s.std(ddof=1)))


def _payload(value: Any) -> dict:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def session_summaries(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    frame = events.sort_values("id").copy()
    sessions: Dict[str, dict] = {}
    for _, row in frame.iterrows():
        sh = str(row.get("session_hash") or "")
        if not sh:
            continue
        rec = sessions.setdefault(sh, {
            "session_hash": sh,
            "completed": False,
            "experience_group": None,
            "expertise_rating": None,
            "scenario_id": None,
            "follow_up_count": 0,
            "first_follow_up_category": None,
            "time_before_first_follow_up_sec": None,
            "evidence_view_count": 0,
            "unique_evidence_areas": [],
            "repeat_evidence_access_count": 0,
            "recommendation_decision": None,
            "final_confidence": None,
            "participant_rationale": None,
            "total_time_to_final_decision_sec": None,
        })
        event_type = str(row.get("event_type") or "")
        payload = _payload(row.get("payload_json"))
        if event_type == "session_start":
            rec["experience_group"] = payload.get("experience_group")
            rec["expertise_rating"] = payload.get("expertise_rating")
            rec["scenario_id"] = payload.get("scenario_id")
        elif event_type == "evidence_view":
            rec["evidence_view_count"] += 1
            area = str(payload.get("evidence_area") or "")
            if area:
                if area in rec["unique_evidence_areas"]:
                    rec["repeat_evidence_access_count"] += 1
                else:
                    rec["unique_evidence_areas"].append(area)
        elif event_type == "chat_turn":
            rec["follow_up_count"] += 1
            if rec["first_follow_up_category"] is None:
                rec["first_follow_up_category"] = payload.get("category")
                rec["time_before_first_follow_up_sec"] = payload.get("time_since_session_start_sec")
        elif event_type == "session_end":
            rec["completed"] = True
            rec["recommendation_decision"] = payload.get("recommendation_decision")
            rec["final_confidence"] = payload.get("final_confidence")
            rec["participant_rationale"] = payload.get("participant_rationale")
            rec["total_time_to_final_decision_sec"] = payload.get("total_time_to_final_decision_sec")
            rec["override_correction"] = payload.get("override_correction")
            rec["override_targets"] = payload.get("override_targets") or []
    return pd.DataFrame(list(sessions.values()))


def manual_coding(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame(columns=["coded_session_hash", "themes", "coder_note"])
    rows = []
    for _, row in events.sort_values("id").iterrows():
        if str(row.get("event_type")) != "qualitative_code":
            continue
        payload = _payload(row.get("payload_json"))
        coded = str(payload.get("coded_session_hash") or "").strip()
        if not coded:
            continue
        rows.append({
            "event_id": row.get("id"),
            "coded_session_hash": coded,
            "themes": list(payload.get("themes") or []),
            "coder_note": payload.get("coder_note"),
            "created_at": row.get("created_at"),
        })
    if not rows:
        return pd.DataFrame(columns=["coded_session_hash", "themes", "coder_note"])
    frame = pd.DataFrame(rows).sort_values("event_id")
    return frame.groupby("coded_session_hash", as_index=False).tail(1).reset_index(drop=True)


def interaction_sequence(events: pd.DataFrame, session_hash: str) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    out = events[events["session_hash"].astype(str) == str(session_hash)].copy().sort_values("id")
    out["sequence_index"] = range(1, len(out) + 1)
    return out[["sequence_index", "created_at", "event_type", "payload_json"]]


def _kmeans_numpy(X: np.ndarray, k: int, iterations: int = 50) -> np.ndarray:
    if len(X) < k:
        return np.zeros(len(X), dtype=int)
    order = np.argsort(X[:, 0] + 0.37 * X[:, 1] + 0.19 * X[:, 2] + 0.11 * X[:, 3])
    seeds = np.linspace(0, len(order) - 1, k).round().astype(int)
    centers = X[order[seeds]].copy()
    labels = np.zeros(len(X), dtype=int)
    for iteration in range(iterations):
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = dist.argmin(axis=1)
        if iteration > 0 and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            pts = X[labels == j]
            if len(pts):
                centers[j] = pts.mean(axis=0)
    return labels


def analyze(events: pd.DataFrame) -> Dict[str, Any]:
    sessions = session_summaries(events)
    if sessions.empty:
        return {"status": "insufficient_data", "registered_sessions": 0, "completed_sessions": 0}
    completed = sessions[sessions["completed"] == True].copy()
    out: Dict[str, Any] = {
        "status": "ok" if len(completed) else "insufficient_data",
        "registered_sessions": int(len(sessions)),
        "completed_sessions": int(len(completed)),
    }

    category_counts: Counter = Counter()
    topic_counts: Counter = Counter()
    evidence_counts: Counter = Counter()
    revisit_counts: Counter = Counter()
    if events is not None and not events.empty:
        for _, row in events.iterrows():
            payload = _payload(row.get("payload_json"))
            if str(row.get("event_type")) == "chat_turn":
                category_counts[str(payload.get("category") or "other_follow_up")] += 1
                for topic in payload.get("requested_topics") or []:
                    topic_counts[str(topic)] += 1
            elif str(row.get("event_type")) == "evidence_view":
                area = str(payload.get("evidence_area") or "unknown")
                evidence_counts[area] += 1
                if int(payload.get("area_visit_number") or 1) > 1:
                    revisit_counts[area] += 1
    out["category_counts"] = dict(category_counts)
    out["requested_topic_counts"] = dict(topic_counts)
    out["evidence_view_counts"] = dict(evidence_counts)
    out["evidence_revisit_counts"] = dict(revisit_counts)

    codes = manual_coding(events)
    theme_counts: Counter = Counter()
    for values in codes.get("themes", pd.Series(dtype=object)).tolist() if not codes.empty else []:
        for value in values or []:
            theme_counts[str(value)] += 1
    out["manual_theme_counts"] = dict(theme_counts)
    out["manually_coded_sessions"] = int(len(codes))

    if completed.empty:
        return out

    numeric = [
        "follow_up_count", "evidence_view_count", "repeat_evidence_access_count",
        "total_time_to_final_decision_sec", "final_confidence", "time_before_first_follow_up_sec"
    ]
    for col in numeric:
        mean, sd = mean_sd(completed[col])
        out[col] = {"mean": mean, "sd": sd}

    out["follow_up_rate"] = float((pd.to_numeric(completed["follow_up_count"], errors="coerce").fillna(0) > 0).mean())
    out["information_seeking_rate"] = float(((pd.to_numeric(completed["follow_up_count"], errors="coerce").fillna(0) > 0) | (pd.to_numeric(completed["evidence_view_count"], errors="coerce").fillna(0) > 0)).mean())
    out["decision_counts"] = completed["recommendation_decision"].value_counts().to_dict()
    out["experience_counts"] = completed["experience_group"].value_counts().to_dict()

    features = ["follow_up_count", "evidence_view_count", "repeat_evidence_access_count", "total_time_to_final_decision_sec"]
    X = completed[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    if len(completed) >= 4:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        k = 3 if len(completed) >= 8 else 2
        labels = _kmeans_numpy(Z, k)
        completed = completed.copy()
        completed["cluster"] = labels.astype(int) + 1
        profiles = []
        for cluster, group in completed.groupby("cluster"):
            rec = {"cluster": int(cluster), "participants": int(len(group))}
            for col in features:
                rec[col + "_mean"] = float(pd.to_numeric(group[col], errors="coerce").mean())
            profiles.append(rec)
        out["exploratory_clusters"] = profiles
        out["cluster_by_expertise"] = pd.crosstab(completed["cluster"], completed["experience_group"]).reset_index().to_dict(orient="records")
        out["session_clusters"] = completed[["session_hash", "cluster", "experience_group"]].to_dict(orient="records")
    else:
        out["exploratory_clusters"] = []
        out["cluster_by_expertise"] = []
        out["session_clusters"] = []
    return out
