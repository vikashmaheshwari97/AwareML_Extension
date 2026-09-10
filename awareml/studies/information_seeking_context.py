from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except Exception:
            pass
    return str(value)


def _compact_result(row: Dict[str, Any]) -> Dict[str, Any]:
    keep = {
        "framework": row.get("framework"),
        "backend": row.get("backend"),
        "accuracy": row.get("accuracy"),
        "f1_macro": row.get("f1_macro"),
        "runtime_sec": row.get("runtime_sec"),
        "energy_kwh": row.get("energy_kwh"),
        "co2_kg": row.get("co2_kg"),
        "fairness": row.get("fairness"),
        "explainability": row.get("explainability"),
        "drift_events": row.get("drift_events"),
        "drift_summary": row.get("drift_summary"),
        "sustainability": row.get("sustainability"),
        "samples": row.get("samples"),
    }
    return _safe(keep)


def _compact_ranking(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "framework",
        "rank",
        "utility",
        "near_pareto",
        "pareto_epsilon",
        "accuracy",
        "runtime_sec",
        "energy_kwh",
        "co2_kg",
        "fairness_score",
        "interpretability_score",
    ]
    return _safe({key: row.get(key) for key in keys if key in row})


def context_path(mode: str) -> Path:
    mode = "final" if str(mode).lower() == "final" else "pilot"
    return ARTIFACTS / "information_seeking_context_{}.json".format(mode)


def save_study_context(
    mode: str,
    results: List[Dict[str, Any]],
    ranking: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not results or not ranking:
        raise ValueError("Both measured run results and an observed ranking are required.")
    payload = {
        "artifact": "information_seeking_study_context_v1",
        "collection_mode": "final" if str(mode).lower() == "final" else "pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "results": [_compact_result(row) for row in results],
        "ranking": [_compact_ranking(row) for row in ranking],
    }
    raw = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
    payload["sha256_without_self"] = hashlib.sha256(raw).hexdigest()
    path = context_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_study_context(mode: str) -> Optional[Dict[str, Any]]:
    path = context_path(mode)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if not payload.get("results") or not payload.get("ranking"):
        return None
    return payload


def resolve_study_context(
    mode: str,
    current_results: List[Dict[str, Any]],
    current_ranking: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, Optional[Dict[str, Any]]]:
    if current_results and current_ranking:
        return current_results, current_ranking, "current_session", None
    saved = load_study_context(mode)
    if saved:
        return (
            list(saved.get("results") or []),
            list(saved.get("ranking") or []),
            "saved_study_context",
            saved,
        )
    return current_results or [], current_ranking or [], "missing", None
