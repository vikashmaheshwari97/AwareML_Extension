from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STATE = {
    "dataset": None, "dataset_name": None, "target": None, "sensitive": None,
    "positive_label": 1, "run_results": None, "ranking": None,
    "pre_run_recommendation": None, "pre_run_recommendation_meta": None,
    "pre_run_objective": None, "ollama_model": None, "chat_session_id": None,
    "ui_v2_version": "research-ui-v2", "selected_framework": None,
    "v2_profile": None, "v2_candidates": None, "v2_ranking_meta": None,
    "v2_profile_signature": None,
    "preference_weights": {"accuracy": 0.55, "runtime": 0.15, "energy": 0.15, "co2": 0.15},
    "ranking_mode": "point", "theme_mode": "System", "copilot_goal": "", "copilot_proposal": None,
    "copilot_ranked": None, "copilot_evidence": None, "copilot_meta": None,
    "copilot_review": None, "active_experiment_id": None, "ui_notice": None,
}


def ensure_research_state() -> Dict[str, Any]:
    if "awareml_state" not in st.session_state:
        st.session_state.awareml_state = {}
    state = st.session_state.awareml_state
    for key, value in DEFAULT_STATE.items():
        if key not in state:
            state[key] = dict(value) if isinstance(value, dict) else list(value) if isinstance(value, list) else value
    return state


def _manifest_status(active_marker: Path) -> Dict[str, Any]:
    if not active_marker.exists():
        return {"ready": False, "marker": str(active_marker), "manifest": None, "status": "missing"}
    rel = active_marker.read_text(encoding="utf-8").strip()
    if not rel:
        return {"ready": False, "marker": str(active_marker), "manifest": None, "status": "empty-marker"}
    manifest = active_marker.parent / rel
    if not manifest.exists():
        return {"ready": False, "marker": str(active_marker), "manifest": str(manifest), "status": "manifest-missing"}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ready": False, "marker": str(active_marker), "manifest": str(manifest), "status": "invalid-manifest", "error": type(exc).__name__}
    frozen = payload.get("status") in {"frozen", "pass"} or payload.get("phase") in {"6.3", "6.5"}
    return {"ready": bool(frozen), "marker": str(active_marker), "manifest": str(manifest), "status": payload.get("status") or "available", "payload": payload}


def phase_status() -> Dict[str, Any]:
    return {
        "phase6": _manifest_status(ROOT / "data" / "meta" / "active_recommender_v2.txt"),
        "phase7": _manifest_status(ROOT / "data" / "llm" / "active_copilot.txt"),
        "phase8": _manifest_status(ROOT / "data" / "llm" / "active_faithfulness.txt"),
    }


def dataset_signature(state: Dict[str, Any]) -> Optional[str]:
    df = state.get("dataset")
    target = state.get("target")
    if df is None or not target:
        return None
    payload = {
        "name": state.get("dataset_name"), "rows": int(len(df)),
        "columns": [str(c) for c in df.columns], "dtypes": [str(d) for d in df.dtypes],
        "target": str(target),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def result_dicts(state: Optional[Dict[str, Any]] = None):
    state = state or ensure_research_state()
    rows = []
    for result in state.get("run_results") or []:
        if hasattr(result, "to_dict"):
            rows.append(result.to_dict())
        elif isinstance(result, dict):
            rows.append(dict(result))
    return rows
