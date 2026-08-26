from __future__ import annotations

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from awareml.ui.theme import inject_theme
from awareml.ui_v2.runtime_hygiene import configure_runtime_hygiene
from awareml.ui_v2.state import ensure_research_state, phase_status
from awareml.ui_v2.theme import inject_research_theme
from awareml.ui_v2.pages import PAGE_REGISTRY_V2

configure_runtime_hygiene()

st.set_page_config(
    page_title="AwareML Research OS",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

state = ensure_research_state()
inject_theme()
inject_research_theme(state.get("theme_mode", "System"))
status = phase_status()

with st.sidebar:
    st.markdown(
        """
        <div class="r9-brand">
          <div class="r9-brand-mark">◈</div>
          <div>
            <div class="r9-brand-title">AwareML</div>
            <div class="r9-brand-sub">Research OS · UI V2</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    theme_mode = st.selectbox(
        "Appearance",
        ["System", "Dark", "Light"],
        index=["System", "Dark", "Light"].index(state.get("theme_mode", "System")),
        key="r9_theme_mode_selector",
        help="System follows browser preference for the page chrome. Dark and Light force a theme.",
    )
    if theme_mode != state.get("theme_mode", "System"):
        state["theme_mode"] = theme_mode
        st.rerun()

    df = state.get("dataset")
    if df is not None:
        target = state.get("target")
        n_features = max(0, int(df.shape[1]) - (1 if target in df.columns else 0))
        st.markdown(
            """
            <div class="r9-side-context">
              <div class="r9-eyebrow">ACTIVE DATASET</div>
              <div class="r9-side-dataset">{}</div>
              <div class="r9-side-meta">{:,} rows · {} features{}</div>
            </div>
            """.format(
                state.get("dataset_name") or "active dataset",
                len(df),
                n_features,
                (" · target {}".format(target) if target else ""),
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="r9-side-context r9-muted-panel">
              <div class="r9-eyebrow">DATASET CONTEXT</div>
              <div class="r9-side-meta">
                Load a stream in Run Studio. The same context is reused
                across every Research OS workspace.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="r9-side-status">
          <span>Recommender V2</span><b class="{c6}">{s6}</b>
          <span>LLM Copilot</span><b class="{c7}">{s7}</b>
          <span>Faithfulness</span><b class="{c8}">{s8}</b>
        </div>
        """.format(
            c6="ok" if status["phase6"]["ready"] else "warn",
            s6="READY" if status["phase6"]["ready"] else "OFF",
            c7="ok" if status["phase7"]["ready"] else "warn",
            s7="READY" if status["phase7"]["ready"] else "OFF",
            c8="ok" if status["phase8"]["ready"] else "warn",
            s8="READY" if status["phase8"]["ready"] else "OFF",
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="r9-nav-label">WORKSPACES</div>', unsafe_allow_html=True)
    page = st.radio(
        "Workspace",
        list(PAGE_REGISTRY_V2.keys()),
        label_visibility="collapsed",
        key="r9_page",
    )

    with st.expander("Quick start", expanded=False):
        st.markdown(
            """
            1. Open **Run Studio** and load **Synthetic drift** for a fast demo.  
            2. Set **Target = `target`**, **Sensitive attribute = `group`**, **Positive label = `1`**.  
            3. Run the experiment, then open:  
               - **3D Decision Space** for the pre-run recommendation  
               - **Streaming Observatory** for temporal metrics and drift markers  
               - **Responsible AI** for fairness, XAI, sustainability, and AEF  
               - **Copilot Workspace** for proposal + human review  
               - **Export Center** for the reproducibility bundle
            """
        )

    st.divider()
    st.caption(
        "Five frameworks · shared temporal protocol · "
        "human review · evidence-grounded explanations"
    )

PAGE_REGISTRY_V2[page]()
