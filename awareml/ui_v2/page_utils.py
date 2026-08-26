from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .state import ensure_research_state, phase_status, result_dicts


def plot(fig, key, height=None):
    if height is not None:
        fig.update_layout(height=height)
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
    )


def fmt(value, digits=3, suffix=""):
    try:
        value = float(value)
        if not np.isfinite(value):
            return "N/A"
        return "{:.{digits}f}{suffix}".format(
            value, digits=digits, suffix=suffix
        )
    except Exception:
        return "N/A"


def phase_pills():
    status = phase_status()
    return [
        ("ML Recommender V2", "good" if status["phase6"]["ready"] else "warn"),
        ("LLM Copilot", "good" if status["phase7"]["ready"] else "warn"),
        ("Faithfulness", "good" if status["phase8"]["ready"] else "warn"),
    ]


def dataset_ready():
    state = ensure_research_state()
    df, target = state.get("dataset"), state.get("target")
    return df is not None and target and target in df.columns


def results_frame():
    rows = []
    for result in result_dicts():
        fairness = result.get("fairness") or {}
        explainability = result.get("explainability") or {}
        drift = result.get("drift_summary") or {}
        rows.append({
            "Framework": result.get("framework"),
            "Backend": result.get("backend"),
            "Accuracy": result.get("accuracy"),
            "Macro-F1": result.get("f1_macro"),
            "Runtime (s)": result.get("runtime_sec"),
            "Energy (kWh)": result.get("energy_kwh"),
            "CO2 (kg)": result.get("co2_kg"),
            "Drift events": len(result.get("drift_events") or []),
            "Recovery rate": drift.get("recovery_rate"),
            "DP gap": fairness.get("dp_diff"),
            "Equalized odds gap": fairness.get("equalized_odds_gap"),
            "XAI fidelity": explainability.get("fidelity"),
            "XAI stability": explainability.get("stability"),
        })
    return pd.DataFrame(rows)
