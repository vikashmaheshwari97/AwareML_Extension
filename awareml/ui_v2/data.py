from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from awareml.recommender.v2_profile import profile_from_dataframe_v2
from awareml.recommender.v2_service import V2Recommender

from .state import ROOT, dataset_signature, ensure_research_state


@st.cache_resource(show_spinner=False)
def load_v2_recommender():
    return V2Recommender(root=ROOT)


@st.cache_data(show_spinner=False)
def load_phase8_report():
    path = ROOT / "artifacts" / "phase8" / "phase8_faithfulness_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_phase8_cases():
    path = ROOT / "artifacts" / "phase8" / "deterministic_faithfulness_cases.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_phase8_ollama_cases():
    path = ROOT / "artifacts" / "phase8" / "ollama_faithfulness_cases.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def normalized_preferences(weights):
    values = {
        key: max(0.0, float((weights or {}).get(key, 0.0)))
        for key in ["accuracy", "runtime", "energy", "co2"]
    }
    total = sum(values.values())
    if total <= 0:
        return {"accuracy": 0.55, "runtime": 0.15, "energy": 0.15, "co2": 0.15}
    return {key: value / total for key, value in values.items()}


def compute_active_recommendation(force=False):
    state = ensure_research_state()
    df, target = state.get("dataset"), state.get("target")
    if df is None or not target or target not in df.columns:
        return None, None, None

    signature = dataset_signature(state)
    weights = normalized_preferences(state.get("preference_weights") or {})
    mode = state.get("ranking_mode", "point")
    cache_signature = "{}|{}|{}".format(
        signature, json.dumps(weights, sort_keys=True), mode
    )

    if (
        not force
        and state.get("v2_profile_signature") == cache_signature
        and state.get("v2_candidates") is not None
    ):
        return (
            state.get("v2_candidates"),
            state.get("v2_ranking_meta"),
            state.get("v2_profile"),
        )

    profile = profile_from_dataframe_v2(
        df,
        target=target,
        window_size=1000,
        time_budget_sec=60.0,
        dataset_family="unknown",
        source_type="ui_active_dataset",
        drift_type="unknown",
    )
    ranked, meta = load_v2_recommender().recommend_profile(
        profile,
        weights=weights,
        ranking_mode=mode,
        coverage=0.90,
    )

    state["v2_profile"] = profile
    state["v2_candidates"] = ranked
    state["v2_ranking_meta"] = meta
    state["v2_profile_signature"] = cache_signature

    frameworks = ranked["framework"].tolist()
    if state.get("selected_framework") not in frameworks:
        state["selected_framework"] = str(ranked.iloc[0]["framework"])

    return ranked, meta, profile
