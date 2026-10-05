from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

LOCAL_RUNTIME_PATH = Path("artifacts/phase16/private_mainstudy_runtime.json")

_ALLOWED_KEYS = (
    "AWAREML_STUDY_RESEARCHER_KEY",
    "AWAREML_PHASE16_ID_SALT",
    "AWAREML_PHASE16_COLLECTION_MODE",
    "AWAREML_PHASE16_FINAL_ARMED",
    "AWAREML_INFORMATION_SEEKING_COLLECTION_MODE",
    "AWAREML_INFORMATION_SEEKING_FINAL_ARMED",
)


def load_local_runtime(path: Path = LOCAL_RUNTIME_PATH) -> Dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        key: str(raw.get(key) or "").strip()
        for key in _ALLOWED_KEYS
        if str(raw.get(key) or "").strip()
    }


def bootstrap_study_runtime(path: Path = LOCAL_RUNTIME_PATH) -> Dict[str, str]:
    # Local private runtime is authoritative for this workstation.
    values = load_local_runtime(path)
    for key, value in values.items():
        os.environ[key] = value
    return values
