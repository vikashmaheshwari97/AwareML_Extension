from __future__ import annotations

import streamlit as st

from awareml.ui import pages as legacy_pages

from .components import hero
from .page_utils import phase_pills
from .pages_specialist import (
    decision_lab_v2_page,
    drift_temporal_v2_page,
    fairness_v2_page,
    explainability_v2_page,
    sustainability_v2_page,
    trust_calibration_v2_page,
    information_seeking_v2_page,
)


def advanced_labs_page():
    hero(
        "SPECIALIST RESEARCH WORKSPACES",
        "Advanced Research Labs",
        (
            "Specialist research workspaces integrate the current journal-extension analyses while keeping the original paper-baseline recommender and protocol available for historical comparison."
        ),
        pills=phase_pills(),
    )

    labs = {
        "Decision Lab · observed post-run ranking": decision_lab_v2_page,
        "Drift & Temporal Lab": drift_temporal_v2_page,
        "Fairness Lab": fairness_v2_page,
        "Explainability Lab": explainability_v2_page,
        "Sustainability Lab": sustainability_v2_page,
        "Trust Calibration": trust_calibration_v2_page,
        "Information-Seeking Lab": information_seeking_v2_page,
        "Research Protocol": legacy_pages.protocol_page,
        "Legacy Recommender Lab · paper baseline": legacy_pages.recommender_lab_page,
    }

    selected = st.selectbox(
        "Specialist workspace",
        list(labs.keys()),
        key="r95_advanced_lab",
    )

    if selected == "Legacy Recommender Lab · paper baseline":
        st.error(
            "Legacy baseline only — not the authoritative Phase-9 recommender. "
            "It uses the older historical meta-log recommender and nearest-dataset evidence. "
            "Its ChaCha/AutoStreamML ranking can differ from the frozen Phase-6 Recommender V2."
        )
        st.info(
            "Use **3D Decision Space** for the current pre-run ML Recommender V2 and "
            "**Copilot Workspace** for natural-language goal interpretation + human review."
        )

    labs[selected]()
