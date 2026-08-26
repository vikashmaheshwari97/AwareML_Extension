from __future__ import annotations

import json
import streamlit as st

from .components import empty_state, evidence_chips, hero, section
from .data import load_phase8_cases, load_phase8_ollama_cases, load_phase8_report
from .plots import faithfulness_components
from .page_utils import fmt, phase_pills, plot



def faithfulness_lab_page():
    hero(
        "EXPLANATION FAITHFULNESS",
        "Faithfulness Lab",
        (
            "Inspect the Phase-8 counterfactual evidence experiments, rationale "
            "sensitivity and AwareML Evidence Fidelity (AEF) without confusing "
            "external evidence interventions with internal LLM attention attribution."
        ),
        pills=phase_pills(),
    )

    report = load_phase8_report()
    cases = load_phase8_cases()
    live_cases = load_phase8_ollama_cases()

    if not report:
        empty_state(
            "Phase-8 artifacts not found",
            "Complete and freeze Phase 8 before using the Faithfulness Lab.",
        )
        return

    deterministic = report.get("deterministic") or {}
    ollama = report.get("ollama") or {}

    st.warning(
        "This page shows the frozen **Phase-8 development/meta faithfulness benchmark**. "
        "It is intentionally kept separate from the currently uploaded dataset so that "
        "the validation benchmark remains stable and comparable across demonstrations."
    )
    st.info(
        "Recommended interpretation: keep this page as the validated benchmark view. "
        "A future extension could add an optional **current-dataset faithfulness probe**, "
        "but it should be presented separately from the frozen benchmark rather than replacing it."
    )

    with st.expander("How to read the Faithfulness scores", expanded=True):
        st.markdown(
            """
            **AEF (AwareML Evidence Fidelity)** is a project-defined composite:

            - **Grounding** — are the cited evidence IDs valid?
            - **Decision alignment** — does the rationale name the framework that was actually ranked first?
            - **Attribution alignment** — does the rationale cite the objective evidence that most influenced the ranking?
            - **Counterfactual sensitivity** — if important evidence changes, does the explanation change appropriately?
            - **Irrelevant invariance** — if irrelevant metadata changes, does the explanation stay stable?

            AEF combines these five components. A high value means the explanation is
            strongly tied to the recommendation evidence; it does **not** mean the model
            itself is more accurate or fair.
            """
        )

    cols = st.columns(5)
    with cols[0]:
        st.metric("AEF", fmt(deterministic.get("mean_evidence_fidelity_score"), 3))
    with cols[1]:
        st.metric("Grounding", fmt(deterministic.get("mean_grounding_validity"), 3))
    with cols[2]:
        st.metric("Attribution align.", fmt(deterministic.get("mean_attribution_alignment"), 3))
    with cols[3]:
        st.metric("Counterfactual", fmt(deterministic.get("mean_counterfactual_sensitivity"), 3))
    with cols[4]:
        st.metric("Live Ollama AEF", fmt(ollama.get("mean_evidence_fidelity_score"), 3))

    plot(faithfulness_components(deterministic), "faith_component_bar")

    st.markdown(
        """
        <div class="r9-callout">
          <b>Method boundary:</b> Phase 8 uses external evidence interventions,
          reranking and evidence-citation analysis. It does not claim access to
          Ollama attention tensors, PE-LRP, or hidden-state causal traces.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if cases.empty:
        return

    section(
        "Counterfactual evidence explorer",
        (
            "Pick one of the frozen Phase-8 development datasets. "
            "A scenario such as accuracy_evidence_flip deliberately changes one "
            "objective's recommendation evidence and then reruns only the ranking/explanation, "
            "not the AutoML benchmark."
        ),
    )
    dataset_id = st.selectbox(
        "Development/meta dataset",
        sorted(cases["dataset_id"].astype(str).unique().tolist()),
        key="r9_faith_dataset",
    )

    block = cases[cases["dataset_id"].astype(str).eq(dataset_id)].copy()
    st.dataframe(
        block[[
            "scenario",
            "original_top_framework",
            "counterfactual_top_framework",
            "decision_flipped",
            "changed_objective_cited",
            "new_winner_acknowledged",
            "citation_change",
            "counterfactual_sensitivity",
        ]],
        use_container_width=True,
        hide_index=True,
    )

    scenario = st.selectbox(
        "Inspect scenario",
        block["scenario"].tolist(),
        key="r9_faith_scenario",
    )
    row = block[block["scenario"].eq(scenario)].iloc[0]

    left, right = st.columns(2)
    with left:
        st.markdown("**Original rationale**")
        st.write(row["original_rationale"])
        try:
            evidence_chips(json.loads(row["original_evidence_keys"]), show_raw=True)
        except Exception:
            pass

    with right:
        st.markdown("**Counterfactual rationale**")
        st.write(row["counterfactual_rationale"])
        try:
            evidence_chips(json.loads(row["counterfactual_evidence_keys"]), show_raw=True)
        except Exception:
            pass

    if not live_cases.empty:
        section(
            "Live Ollama cases",
            "The live subset is shown separately from the deterministic development baseline.",
        )
        st.dataframe(
            live_cases[[
                "dataset_id",
                "scenario",
                "original_top_framework",
                "counterfactual_top_framework",
                "decision_flipped",
                "counterfactual_sensitivity",
            ]],
            use_container_width=True,
            hide_index=True,
        )
