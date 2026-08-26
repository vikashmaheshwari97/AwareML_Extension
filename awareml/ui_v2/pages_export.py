from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from .components import hero, section, status_panel
from .data import load_phase8_report
from .export import build_research_zip, html_report_bytes, research_payload, results_csv_bytes
from .page_utils import phase_pills
from .state import ensure_research_state, result_dicts


def export_center_page():
    state = ensure_research_state()
    faith = load_phase8_report()

    hero(
        "REPRODUCIBILITY",
        "Export Center",
        (
            "Package current research context, pre-run recommendation evidence, "
            "human-review metadata, observed metrics, faithfulness evidence and "
            "release provenance without exporting raw dataset rows."
        ),
        pills=phase_pills(),
    )

    payload = research_payload(state, faithfulness_report=faith)
    json_bytes = json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    csv_bytes = results_csv_bytes(state)
    html_bytes = html_report_bytes(payload)
    zip_bytes = build_research_zip(state, faithfulness_report=faith)

    cols = st.columns(4)
    with cols[0]:
        st.download_button(
            "Download evidence JSON",
            data=json_bytes,
            file_name="awareml_evidence.json",
            mime="application/json",
            use_container_width=True,
        )
    with cols[1]:
        st.download_button(
            "Download metrics CSV",
            data=csv_bytes,
            file_name="run_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with cols[2]:
        st.download_button(
            "Download HTML report",
            data=html_bytes,
            file_name="awareml_research_report.html",
            mime="text/html",
            use_container_width=True,
        )
    with cols[3]:
        st.download_button(
            "Download reproducibility ZIP",
            data=zip_bytes,
            file_name="awareml_research_bundle.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )

    section(
        "Export contents",
        "The ZIP includes evidence JSON, flat run metrics, HTML report, release manifests when present, and SHA256 checksums.",
    )
    status_panel({
        "Raw dataset rows": "excluded",
        "23-dataset held-out split": "not used by UI",
        "Current run results": "{} framework(s)".format(len(result_dicts())),
        "Pre-run ranking": (
            "available"
            if isinstance(state.get("v2_candidates"), pd.DataFrame)
            else "not available"
        ),
        "Copilot review": "available" if state.get("copilot_review") else "not available",
        "Faithfulness report": "available" if faith else "not available",
    })

    section(
        "Evidence preview",
        "This preview shows metadata only. Raw dataset rows are deliberately absent.",
    )
    st.json(payload, expanded=False)
