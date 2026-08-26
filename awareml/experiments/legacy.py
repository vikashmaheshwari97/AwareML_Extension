from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .registry import file_sha256


def audit_legacy_meta_logs(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Legacy meta log must be a JSON list.")
    df = pd.DataFrame(payload)
    co2_col = "CO2 Emission (µg)" if "CO2 Emission (µg)" in df.columns else "co2_ug" if "co2_ug" in df.columns else None
    energy_col = "energy_consumption_kwh" if "energy_consumption_kwh" in df.columns else "energy_kwh" if "energy_kwh" in df.columns else None
    zero_co2 = 0
    positive_co2 = 0
    if co2_col:
        co2 = pd.to_numeric(df[co2_col], errors="coerce")
        zero_co2 = int((co2 == 0).sum())
        positive_co2 = int((co2 > 0).sum())
    energy_support = 0
    if energy_col:
        energy = pd.to_numeric(df[energy_col], errors="coerce")
        energy_support = int((energy >= 0).sum())
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "rows": int(len(df)),
        "datasets": int(df["dataset_name"].nunique()) if "dataset_name" in df.columns else 0,
        "frameworks": dict(Counter(df["framework"].astype(str))) if "framework" in df.columns else {},
        "zero_co2_placeholders": zero_co2,
        "positive_co2_measurements": positive_co2,
        "energy_rows": energy_support,
    }


def legacy_row_for_v2(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a legacy row without pretending old zeros were measurements."""
    out = dict(row)
    if "CO2 Emission (µg)" in out and "co2_ug" not in out:
        out["co2_ug"] = out.get("CO2 Emission (µg)")
    try:
        if out.get("co2_ug") is not None and float(out["co2_ug"]) <= 0:
            out["co2_ug"] = None
            out["co2_measurement_status"] = "not_measured_legacy_placeholder"
    except Exception:
        out["co2_ug"] = None
        out["co2_measurement_status"] = "invalid_legacy_value"
    if "energy_consumption_kwh" not in out and "energy_kwh" in out:
        out["energy_consumption_kwh"] = out.get("energy_kwh")
    return out
