from __future__ import annotations

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from awareml.ui.theme import inject_theme
from awareml.ui_v2.phase16_trust_calibration import render_phase16_participant_study
from awareml.ui_v2.theme import inject_research_theme

st.set_page_config(
    page_title="AwareML · Trust Calibration",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_theme()
inject_research_theme("System")

# Dedicated participant entry point: no AwareML workspace navigation is created here.
render_phase16_participant_study(isolated=True)
