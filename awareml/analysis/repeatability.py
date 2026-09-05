from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .repeatability_registry import PAPER_READY_MIN_REPETITIONS


REPEATABILITY_METRICS = {
    "runtime_sec": "Runtime (s)",
    "energy_kwh": "Energy (kWh)",
    "co2_kg": "CO2 (kg)",
}


def _as_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "to_dict"):
        return dict(record.to_dict())
    if hasattr(record, "__dict__") and not isinstance(record, Mapping):
        return dict(record.__dict__)
    return dict(record)


def summarize_repeatability(
    records: Iterable[Any],
    *,
    group_keys: Sequence[str] = ("framework",),
    min_repetitions: int = PAPER_READY_MIN_REPETITIONS,
) -> pd.DataFrame:
    """Return mean ± sample-SD with the formal paper-ready repetition gate."""
    rows = [_as_dict(record) for record in records]
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    keys = [key for key in group_keys if key in frame.columns]
    if not keys:
        frame["_all"] = "all"
        keys = ["_all"]

    # Avoid pandas' single-list-grouper FutureWarning.
    grouper = keys[0] if len(keys) == 1 else keys

    output = []
    for group, part in frame.groupby(grouper, dropna=False):
        group_values = group if isinstance(group, tuple) else (group,)
        row = {
            key: value
            for key, value in zip(keys, group_values)
        }
        row["Repetitions recorded"] = int(len(part))
        row["Paper-ready minimum"] = int(min_repetitions)
        row["Repeatability gate"] = (
            "PASS"
            if len(part) >= int(min_repetitions)
            else "NEEDS REPETITIONS"
        )

        for source, label in REPEATABILITY_METRICS.items():
            values = (
                pd.to_numeric(part[source], errors="coerce").dropna()
                if source in part.columns
                else pd.Series(dtype=float)
            )
            n = int(len(values))
            mean = float(values.mean()) if n else None
            sd = float(values.std(ddof=1)) if n >= 2 else None
            median = float(values.median()) if n else None

            row[label + " n"] = n
            row[label + " mean"] = mean
            row[label + " SD"] = sd
            row[label + " median"] = median
            row[label + " mean ± SD"] = (
                "{:.6g} ± {:.6g}".format(mean, sd)
                if mean is not None and sd is not None
                else (
                    "{:.6g} ± N/A".format(mean)
                    if mean is not None
                    else "N/A"
                )
            )
        output.append(row)

    out = pd.DataFrame(output)
    if "_all" in out.columns:
        out = out.drop(columns=["_all"])
    return out


def hardware_table(records: Iterable[Any]) -> pd.DataFrame:
    rows = []
    for raw in records:
        record = _as_dict(raw)
        sustain = record.get("sustainability") or {}
        rows.append({
            "Framework": record.get("framework"),
            "CPU": sustain.get("cpu"),
            "Physical CPUs": sustain.get("physical_cpus"),
            "Logical CPUs": sustain.get("logical_cpus"),
            "GPU": sustain.get("gpu"),
            "RAM (GB)": sustain.get("ram_gb"),
            "Country": sustain.get("country_iso"),
            "Region": sustain.get("region"),
            "CodeCarbon": sustain.get("codecarbon_version"),
            "Backend": sustain.get("measurement_backend"),
            "Carbon intensity (gCO2/kWh)": sustain.get(
                "carbon_intensity_g_per_kwh"
            ),
            "Carbon intensity source": sustain.get(
                "carbon_intensity_source"
            ),
            "Measurement duration (s)": sustain.get("duration_sec"),
            "Warm-up (s)": sustain.get("warmup_sec"),
            "Warm-up samples": sustain.get("warmup_samples"),
            "Repetition": sustain.get("repetition_id"),
            "Repetitions planned": sustain.get("repetitions_planned"),
            "Measurement status": sustain.get("status"),
            "Failure reason": sustain.get("measurement_failure_reason"),
        })
    return pd.DataFrame(rows)


def phase14_gate(
    records: Iterable[Any],
    min_repetitions: int = PAPER_READY_MIN_REPETITIONS,
) -> dict[str, Any]:
    rows = [_as_dict(record) for record in records]
    if not rows:
        return {
            "fairness_implementation": "implemented",
            "calibration_fairness_evidence": "no run evidence",
            "sustainability_protocol": "implemented",
            "repeatability_evidence": "no run evidence",
            "hardware_table": "no run evidence",
            "paper_ready_evidence": False,
            "paper_ready_min_repetitions": int(min_repetitions),
        }

    calibration_statuses = [
        (row.get("fairness") or {}).get("calibration_status")
        for row in rows
    ]
    calibration_ok = any(
        status == "ok"
        for status in calibration_statuses
    )

    repeats = summarize_repeatability(
        rows,
        min_repetitions=min_repetitions,
    )
    repeatability_ok = (
        not repeats.empty
        and bool((repeats["Repeatability gate"] == "PASS").all())
    )

    hardware = hardware_table(rows)
    hardware_ok = (
        not hardware.empty
        and hardware["CPU"].notna().any()
        and hardware["RAM (GB)"].notna().any()
    )

    return {
        "fairness_implementation": "implemented",
        "calibration_fairness_evidence": (
            "available" if calibration_ok else "unavailable/pending"
        ),
        "sustainability_protocol": "implemented",
        "repeatability_evidence": (
            "pass"
            if repeatability_ok
            else "needs >= {} repetitions".format(min_repetitions)
        ),
        "hardware_table": "available" if hardware_ok else "partial/pending",
        "paper_ready_evidence": bool(
            calibration_ok and repeatability_ok and hardware_ok
        ),
        "paper_ready_min_repetitions": int(min_repetitions),
    }
