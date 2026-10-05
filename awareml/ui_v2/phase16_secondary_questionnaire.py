from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .phase16_research_visuals import render_secondary_visuals

from awareml.studies.phase16_secondary_questionnaire import (
    SECONDARY_ITEMS,
    has_secondary_response,
    save_secondary_response,
    secondary_rows,
    secondary_summary,
)


def _fmt(value: Any) -> str:
    try:
        return "{:.2f}".format(float(value))
    except Exception:
        return "N/A"


def render_phase16_secondary_participant(
    db_path: Path,
    participant_hash: str,
    collection_mode: str,
) -> bool:
    if has_secondary_response(db_path, collection_mode, participant_hash):
        return True

    st.markdown(
        """
        <div class="tc-design">
          <strong>Final questionnaire · Human agency and study experience</strong><br>
          These questions are separate from the trial-by-trial trust measure.
          Please answer based on your overall experience in this study.
          There are no right or wrong answers.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Scale: 1 = strongly disagree · 4 = neither agree nor disagree · 7 = strongly agree.")

    with st.form("phase16_secondary_poststudy_form"):
        ratings = {}
        for item in SECONDARY_ITEMS:
            ratings[item["id"]] = st.slider(
                item["text"], 1, 7, 4, key="p16_secondary_{}".format(item["id"])
            )
        comment = st.text_area(
            "Optional: What most influenced whether you trusted, questioned, overrode, or rejected an explanation?",
            max_chars=4000,
        )
        submitted = st.form_submit_button(
            "Submit final questionnaire", type="primary", use_container_width=True
        )

    if submitted:
        try:
            save_secondary_response(
                db_path, collection_mode, participant_hash, ratings, open_comment=comment
            )
        except ValueError as exc:
            st.error(str(exc))
            return False
        st.success("Final questionnaire saved.")
        st.rerun()

    return False


def render_phase16_secondary_researcher(db_path: Path, collection_mode: str) -> None:
    st.markdown("### Secondary human-agency and study-experience measures")
    st.caption(
        "These measures complement the primary correct-vs-incorrect Trust Calibration analysis. "
        "They are exploratory/custom unless the research team validates or replaces them before Main Study."
    )

    summary = secondary_summary(db_path, collection_mode)
    rows = secondary_rows(db_path, collection_mode)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questionnaires", summary.get("responses", 0))
    constructs = summary.get("construct_means") or {}
    c2.metric("Human agency", _fmt(constructs.get("human_agency")))
    c3.metric("Calibration support", _fmt(constructs.get("calibrated_reliance_support")))
    c4.metric("Usability", _fmt(constructs.get("usability")))

    render_secondary_visuals(summary, rows)

    st.markdown("#### Detailed item table")
    item_frame = pd.DataFrame(
        [
            {
                "Construct": item["construct"],
                "Item": item["text"],
                "Mean (1-7)": (summary.get("item_means") or {}).get(item["id"]),
            }
            for item in SECONDARY_ITEMS
        ]
    )
    st.dataframe(item_frame, use_container_width=True, hide_index=True)

    by_expertise = pd.DataFrame(summary.get("by_expertise") or [])
    if not by_expertise.empty:
        st.markdown("#### Descriptive novice / practitioner / expert comparison table")
        st.dataframe(by_expertise, use_container_width=True, hide_index=True)

    if rows:
        export = pd.DataFrame(rows)
        st.download_button(
            "Download secondary questionnaire responses",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="phase16_{}_secondary_questionnaire.csv".format(collection_mode),
            mime="text/csv",
            key="p16_secondary_download_{}".format(collection_mode),
            use_container_width=True,
        )
