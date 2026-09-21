from __future__ import annotations

import streamlit as st

from .components import hero
from .page_utils import phase_pills
from .pages_faithfulness import faithfulness_lab_page
from .pages_specialist import explainability_v2_page
from .phase15_explanation_integrity import phase15_explanation_integrity_page


WORKSPACES = {
    "Correctness & Faithfulness": (
        "Validate whether explanation claims agree with structured evidence and "
        "whether explanations react to decision-relevant evidence changes."
    ),
    "Faithfulness Benchmark": (
        "Inspect the controlled counterfactual-evidence benchmark, grounding, "
        "attribution alignment and rationale sensitivity."
    ),
    "Explainability Diagnostics": (
        "Inspect explanation availability, feature-attribution evidence, "
        "method audit trails and cross-framework XAI quality."
    ),
}


def explanation_integrity_lab_page():
    labels = list(WORKSPACES)
    selected = st.session_state.get(
        "explanation_integrity_workspace_select",
        "Correctness & Faithfulness",
    )
    if selected not in WORKSPACES:
        selected = "Correctness & Faithfulness"

    hero(
        "EXPLANATION INTEGRITY",
        selected,
        WORKSPACES[selected],
        pills=phase_pills(),
    )

    selected = st.selectbox(
        "Explanation workspace",
        labels,
        index=labels.index(selected),
        key="explanation_integrity_workspace_select",
        help=(
            "Choose the explanation-integrity view. The evaluators remain "
            "methodologically separate; this control only changes the view."
        ),
    )

    if selected == "Faithfulness Benchmark":
        faithfulness_lab_page(show_header=False)
    elif selected == "Explainability Diagnostics":
        explainability_v2_page(show_header=False)
    else:
        phase15_explanation_integrity_page(show_header=False)
