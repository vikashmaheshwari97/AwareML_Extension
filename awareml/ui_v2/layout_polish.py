from __future__ import annotations

import streamlit as st


def inject_layout_polish() -> None:
    """Conservative spacing polish shared by AwareML workspaces.

    Presentation only: no state, experiment, study, ranking, or analysis logic.
    """
    st.markdown(
        """
        <style>
        /* Global AwareML spacing polish */
        .block-container {
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-bottom: 4.5rem !important;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 1rem !important;
            align-items: stretch;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            min-width: 0;
        }

        .r9-hero {
            margin-bottom: 1.35rem !important;
        }

        .r9-card,
        .r9-panel,
        div[data-testid="stMetric"] {
            margin-bottom: .55rem;
        }

        .r9-section {
            margin-top: 2rem !important;
            margin-bottom: .8rem !important;
        }

        .r9-section-sub {
            margin-bottom: 1rem !important;
        }

        div[data-testid="stExpander"] {
            margin-top: .35rem;
            margin-bottom: .8rem;
        }

        div[data-testid="stForm"] {
            margin-top: .35rem;
            margin-bottom: .85rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            margin-top: .15rem;
            margin-bottom: .35rem;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            div[data-testid="stHorizontalBlock"] {
                gap: .75rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
