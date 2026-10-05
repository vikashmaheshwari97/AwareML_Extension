from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st

from .components import empty_state, hero, metric_card, section, status_panel
from .data import compute_active_recommendation, load_phase8_report
from .page_utils import dataset_ready
from .state import ensure_research_state, phase_status, result_dicts


def _workflow_card(step: str, title: str, body: str, status: str = "") -> str:
    status_html = (
        '<div style="margin-top:10px;font-size:.74rem;font-weight:700;color:#0F766E;">{}</div>'.format(status)
        if status else ""
    )
    return (
        '<div style="height:100%;padding:16px 17px;border:1px solid #E2E8F0;border-radius:16px;'
        'background:linear-gradient(145deg,#FFFFFF,#F8FAFC);box-shadow:0 8px 20px rgba(15,23,42,.045);">'
        '<div style="font-size:.70rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:#2563EB;">{}</div>'
        '<div style="font-size:1.02rem;font-weight:760;color:#0F172A;margin-top:5px;">{}</div>'
        '<div style="font-size:.81rem;line-height:1.45;color:#64748B;margin-top:6px;">{}</div>{}'
        '</div>'
    ).format(step, title, body, status_html)


def _context_status(state: Dict[str, Any]) -> Dict[str, str]:
    dataset = state.get("dataset")
    ranked = state.get("v2_candidates")
    observed = result_dicts()

    if dataset is None:
        dataset_text = "Not loaded"
        recommendation_text = "Waiting for dataset"
    else:
        dataset_text = str(state.get("dataset_name") or "Active dataset")
        recommendation_text = "Ready" if isinstance(ranked, pd.DataFrame) and not ranked.empty else "Available after prediction"

    benchmark_text = "Observed results available" if observed else "Not executed yet"
    review_text = "Human review available in Copilot Workspace"

    return {
        "Dataset": dataset_text,
        "Pre-run recommendation": recommendation_text,
        "Streaming benchmark": benchmark_text,
        "Human oversight": review_text,
    }


def render_command_center_upgrade() -> None:
    state = ensure_research_state()
    status = phase_status()
    faith = load_phase8_report()

    hero(
        "COMMAND CENTRE",
        "AwareML Dashboard",
        (
            "One place to understand the active dataset, obtain a pre-execution framework recommendation, "
            "review and modify the plan, execute the streaming benchmark, inspect drift and responsible-AI evidence, "
            "and export reproducible results."
        ),
        pills=[
            ("ML recommender", "good" if status["phase6"]["ready"] else "warn"),
            ("Copilot", "good" if status["phase7"]["ready"] else "warn"),
            ("Faithfulness", "good" if status["phase8"]["ready"] else "warn"),
            ("Human oversight", "good"),
        ],
    )

    cols = st.columns(4)
    with cols[0]:
        metric_card(
            "FRAMEWORKS",
            "5",
            "AutoStreamML · AutoClass · EvoAutoML · OAML · ChaCha",
        )
    with cols[1]:
        metric_card(
            "ML RECOMMENDER",
            "READY" if status["phase6"]["ready"] else "OFF",
            "Frozen objective-specific models",
            "good" if status["phase6"]["ready"] else "warn",
        )
    with cols[2]:
        metric_card(
            "COPILOT",
            "READY" if status["phase7"]["ready"] else "OFF",
            "Goal interpretation · human review · execution plan",
            "good" if status["phase7"]["ready"] else "warn",
        )
    with cols[3]:
        aef = (
            ((faith.get("deterministic") or {}).get("mean_evidence_fidelity_score"))
            if faith else None
        )
        metric_card(
            "FAITHFULNESS",
            "{:.3f}".format(float(aef)) if aef is not None else "N/A",
            "Evidence-grounding diagnostic",
            "good" if status["phase8"]["ready"] else "warn",
        )

    section(
        "How to use AwareML",
        "The dashboard separates prediction, human decision-making, execution and post-run evidence so users can see exactly what is known at each stage.",
    )

    row1 = st.columns(4)
    cards1 = [
        ("01 · CONTEXT", "Run Studio", "Load the dataset, choose the target and define the streaming experiment.", "Starts the shared workspace context"),
        ("02 · PREDICT", "3D Decision Space", "Inspect the five predicted framework profiles and preference-aware pre-execution ranking.", "No framework run required"),
        ("03 · REVIEW", "Copilot Workspace", "Translate a deployment goal into priorities, compare evidence paths and approve, edit or reject the plan.", "Human decision remains explicit"),
        ("04 · EXECUTE", "Run Studio", "Run the selected framework configuration under the shared streaming protocol.", "Creates observed evidence"),
    ]
    for col, card in zip(row1, cards1):
        with col:
            st.markdown(_workflow_card(*card), unsafe_allow_html=True)

    row2 = st.columns(3)
    cards2 = [
        ("05 · OBSERVE", "Streaming Observatory", "Inspect temporal performance, latency, drift markers and recovery after execution.", "Post-run evidence"),
        ("06 · AUDIT", "Responsible AI + Explanation Integrity", "Review fairness, explainability, sustainability, correctness and faithfulness.", "Responsible-AI evidence"),
        ("07 · DOCUMENT", "Export Center", "Save the configuration, measured outcomes, provenance and evidence needed for reproducibility.", "Research artifact"),
    ]
    for col, card in zip(row2, cards2):
        with col:
            st.markdown(_workflow_card(*card), unsafe_allow_html=True)

    section(
        "Current workspace state",
        "This area shows what evidence is currently available. It never invents benchmark values when no dataset or run exists.",
    )
    status_panel(_context_status(state))

    if not dataset_ready():
        st.info(
            "Start in **Run Studio** when you want a dataset-specific recommendation or benchmark. "
            "You can open **Copilot Workspace** without a dataset to explore goal interpretation, but the dataset-aware ranking activates only after dataset and target context are available."
        )
        return

    ranked = state.get("v2_candidates")
    if ranked is None:
        try:
            ranked, _, _ = compute_active_recommendation()
        except Exception as exc:
            st.warning("The pre-execution recommendation is not available yet: {}".format(exc))
            ranked = None

    if isinstance(ranked, pd.DataFrame) and not ranked.empty:
        section(
            "Current pre-execution recommendation",
            "This is a prediction from the frozen recommender for the active dataset. It is not an observed benchmark winner.",
        )
        ranked_display = ranked.copy()
        keep = [
            c for c in [
                "rank", "framework", "utility", "accuracy", "runtime", "energy", "co2", "near_pareto"
            ]
            if c in ranked_display.columns
        ]
        st.dataframe(
            ranked_display[keep].head(5),
            use_container_width=True,
            hide_index=True,
        )

    if result_dicts():
        st.success(
            "Observed benchmark results are available. Continue to **Streaming Observatory** for temporal behavior, "
            "**Responsible AI** for fairness/explainability/sustainability, and **Explanation Integrity Lab** for correctness and faithfulness."
        )
    else:
        st.info(
            "The recommendation is still pre-execution. Run the agreed configuration in **Run Studio** when you are ready to create measured evidence."
        )
