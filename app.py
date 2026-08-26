from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from awareml.ui.theme import inject_theme
from awareml.ui.pages import PAGE_REGISTRY

st.set_page_config(
    page_title="AwareML Extension",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

if "awareml_state" not in st.session_state:
    st.session_state.awareml_state = {
        "dataset": None,
        "dataset_name": None,
        "target": None,
        "sensitive": None,
        "positive_label": 1,
        "run_results": None,
        "ranking": None,
        "pre_run_recommendation": None,
        "pre_run_recommendation_meta": None,
        "pre_run_objective": None,
        "ollama_model": None,
        "chat_session_id": None,
    }

state = st.session_state.awareml_state

with st.sidebar:
    st.markdown('<div class="brand-mark">◈</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-title">AwareML</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Research-grade streaming AutoML</div>', unsafe_allow_html=True)

    df = state.get("dataset")
    if df is not None:
        name = state.get("dataset_name") or "active dataset"
        target = state.get("target")
        n_features = max(0, df.shape[1] - (1 if target in df.columns else 0))
        st.markdown(
            f'<div class="sidebar-dataset"><div class="sidebar-kicker">Active dataset</div>'
            f'<div class="sidebar-dataset-name">{name}</div>'
            f'<div class="sidebar-dataset-meta">{len(df):,} rows · {n_features} features'
            + (f' · target {target}' if target else '')
            + '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sidebar-dataset empty"><div class="sidebar-kicker">Dataset context</div>'
            '<div class="sidebar-dataset-meta">Load once in Run Studio. Every workspace will reuse it.</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("Workflow")
    page = st.radio(
        "Workspace",
        list(PAGE_REGISTRY.keys()),
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Five frameworks · one auditable temporal protocol")
    st.caption("AutoStreamML · AutoClass · EvoAutoML · OAML · ChaCha")

PAGE_REGISTRY[page]()
