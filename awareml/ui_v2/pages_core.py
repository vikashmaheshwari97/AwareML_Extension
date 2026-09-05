from __future__ import annotations

import pandas as pd
import streamlit as st

from awareml.ui import pages as legacy_pages

from .components import empty_state, hero, metric_card, section, status_panel
from .data import compute_active_recommendation, load_phase8_report, normalized_preferences
from .dataset_advisor import render_dataset_advisor
from .plots import decision_space_3d, decision_space_3d_normalized, ranking_bar
from .state import ensure_research_state, phase_status, result_dicts
from .page_utils import dataset_ready, fmt, phase_pills, plot, results_frame
from .pre14_usability import (
    preference_context,
    quick_framework_selector,
    render_dataset_task_guard,
    render_decision_space_context,
)



def command_center_page():
    state = ensure_research_state()
    status = phase_status()
    faith = load_phase8_report()

    hero(
        "PHASE 9 · RESEARCH UI V2",
        "AwareML Research OS",
        (
            "One integrated command center for streaming AutoML, "
            "multi-objective recommendation, human review, responsible-AI "
            "evidence, grounded Copilot interaction, faithfulness analysis, "
            "and reproducible export."
        ),
        pills=phase_pills(),
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
            "Frozen objective-specific V2 models",
            "good" if status["phase6"]["ready"] else "warn",
        )
    with cols[2]:
        metric_card(
            "COPILOT",
            "READY" if status["phase7"]["ready"] else "OFF",
            "Goal → reviewable configuration → evidence",
            "good" if status["phase7"]["ready"] else "warn",
        )
    with cols[3]:
        aef = (
            ((faith.get("deterministic") or {}).get("mean_evidence_fidelity_score"))
            if faith
            else None
        )
        metric_card(
            "FAITHFULNESS",
            "{:.3f}".format(float(aef)) if aef is not None else "N/A",
            "Development/meta AEF",
            "good" if status["phase8"]["ready"] else "warn",
        )

    section(
        "Research pipeline",
        "The interface follows the same evidence flow used by the frozen backend.",
    )
    st.markdown(
        """
        <div class="r9-panel">
          <div class="r9-pipeline">
            <div class="r9-pipeline-node">Dataset<br>context</div>
            <div class="r9-pipeline-arrow">→</div>
            <div class="r9-pipeline-node">ML recommender<br>predictions</div>
            <div class="r9-pipeline-arrow">→</div>
            <div class="r9-pipeline-node">Human review<br>& configuration</div>
            <div class="r9-pipeline-arrow">→</div>
            <div class="r9-pipeline-node">Streaming<br>experiment</div>
          </div>
          <div class="r9-pipeline" style="margin-top:8px">
            <div class="r9-pipeline-node">Performance<br>& drift</div>
            <div class="r9-pipeline-arrow">→</div>
            <div class="r9-pipeline-node">Fairness · XAI<br>· sustainability</div>
            <div class="r9-pipeline-arrow">→</div>
            <div class="r9-pipeline-node">Grounded<br>Copilot</div>
            <div class="r9-pipeline-arrow">→</div>
            <div class="r9-pipeline-node">Faithfulness<br>& export</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section(
        "Active research context",
        "No decorative benchmark values are synthesized. Decision visuals appear only from active measured/predicted evidence.",
    )
    left, right = st.columns([1.05, 1.45])

    with left:
        if not dataset_ready():
            empty_state(
                "No active dataset yet",
                (
                    "Open Run Studio and load a stream. The dataset, target "
                    "and experiment state then propagate across every workspace."
                ),
            )
        else:
            df, target = state["dataset"], state["target"]
            n_features = max(0, df.shape[1] - 1)
            classes = int(df[target].dropna().nunique())
            status_panel({
                "Dataset": state.get("dataset_name") or "active dataset",
                "Rows": "{:,}".format(len(df)),
                "Features": str(n_features),
                "Classes": str(classes),
                "Target": str(target),
                "Sensitive attribute": str(state.get("sensitive") or "not selected"),
            })

            if st.button("Refresh pre-run recommendation", type="primary", key="cc_refresh"):
                compute_active_recommendation(force=True)
                st.rerun()

    with right:
        ranked = state.get("v2_candidates")
        if ranked is None and dataset_ready():
            try:
                ranked, _, _ = compute_active_recommendation()
            except Exception as exc:
                st.warning("Pre-run recommender unavailable: {}".format(exc))

        if isinstance(ranked, pd.DataFrame) and not ranked.empty:
            plot(
                decision_space_3d(
                    ranked,
                    selected_framework=state.get("selected_framework"),
                ),
                "cc_3d",
                height=500,
            )
            st.caption(
                "Interactive 3D decision space · X=Accuracy ↑ · Y=Runtime ↓ · "
                "Z=Energy ↓ · marker size=CO₂ · white outline=selected framework · "
                "green outline=ε-Pareto candidate (ε=0.05)."
            )
        else:
            empty_state(
                "Decision space waiting for context",
                (
                    "Load an active dataset to obtain Phase-6 predicted "
                    "framework outcomes and canonical ε-Pareto ranking (ε=0.05)."
                ),
            )

    if result_dicts():
        section("Current experiment snapshot", "Observed values from the active benchmark run.")
        st.dataframe(results_frame(), use_container_width=True, hide_index=True)



def decision_space_page():
    state = ensure_research_state()

    hero(
        "MULTI-OBJECTIVE RECOMMENDATION",
        "3D Decision Space",
        (
            "Explore how the five frameworks occupy the predicted Accuracy × "
            "Runtime × Energy landscape. CO₂ is encoded as marker size and the "
            "same frozen predictions are reranked instantly when your preferences change."
        ),
        pills=phase_pills(),
    )

    st.markdown(
        """
        <div class="r9-callout">
          <b>PRE-RUN PREFERENCE RECOMMENDATION</b><br>
          <b>Evidence source:</b> predicted framework outcomes from frozen ML Recommender V2.<br>
          <b>Preference source:</b> the manual Accuracy / Runtime / Energy / CO₂ sliders on this page.<br>
          <b>Ranking engine:</b> ML Recommender V2 preference-aware reranking.<br>
          <b>Framework execution:</b> not required; these are predictions, not observed results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not dataset_ready():
        empty_state(
            "Load a dataset first",
            "Use Run Studio to create a dataset/target context for the frozen Phase-6 recommender.",
        )
        return

    section(
        "Preference controls",
        "Weights normalize automatically. Changing them does not rerun AutoML; it reranks the same predicted framework outcomes from the frozen recommender.",
    )
    current = normalized_preferences(state.get("preference_weights") or {})
    current, manual_pref = preference_context(state, current)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        accuracy = st.slider("Accuracy ↑", 0, 100, int(round(current["accuracy"] * 100)), key="r9_w_accuracy", disabled=not manual_pref)
    with c2:
        runtime = st.slider("Runtime ↓", 0, 100, int(round(current["runtime"] * 100)), key="r9_w_runtime", disabled=not manual_pref)
    with c3:
        energy = st.slider("Energy ↓", 0, 100, int(round(current["energy"] * 100)), key="r9_w_energy", disabled=not manual_pref)
    with c4:
        co2 = st.slider("CO₂ ↓", 0, 100, int(round(current["co2"] * 100)), key="r9_w_co2", disabled=not manual_pref)

    state["preference_weights"] = normalized_preferences({
        "accuracy": accuracy, "runtime": runtime, "energy": energy, "co2": co2
    })

    mode = st.segmented_control(
        "Ranking evidence",
        ["Point prediction", "Conservative 90% bounds"],
        default=(
            "Conservative 90% bounds"
            if state.get("ranking_mode") == "conservative"
            else "Point prediction"
        ),
        key="r9_rank_mode",
    )
    state["ranking_mode"] = "conservative" if mode == "Conservative 90% bounds" else "point"

    st.markdown(
        """
        <div class="r9-callout">
          <b>Point prediction</b> ranks frameworks using the central predicted Accuracy, Runtime, Energy and CO₂.
          <b>Conservative 90% bounds</b> uses the <b>lower accuracy bound</b> and the <b>upper Runtime/Energy/CO₂ bounds</b>,
          giving a risk-aware ranking that asks: “which framework still looks good under an unfavorable but calibrated scenario?”
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        ranked, meta, profile = compute_active_recommendation(force=True)
    except Exception as exc:
        st.error("Unable to compute the V2 recommendation: {}".format(exc))
        return

    frameworks = ranked["framework"].tolist()
    selected = quick_framework_selector(frameworks, state)

    top = ranked.iloc[0]
    selected_row = ranked[ranked["framework"].eq(selected)].iloc[0]
    render_decision_space_context(ranked, state, selected)
    cols = st.columns(6)
    with cols[0]:
        st.metric("Pre-run recommended framework", str(top["framework"]), "Predicted rank #1")
    with cols[1]:
        st.metric("Pre-run preference utility", fmt(top["utility"], 4))
    with cols[2]:
        st.metric("{} predicted accuracy".format(selected), fmt(selected_row["accuracy"], 4))
    with cols[3]:
        st.metric("{} predicted runtime".format(selected), fmt(selected_row["runtime"], 3, " s"))
    with cols[4]:
        st.metric("{} predicted energy".format(selected), fmt(selected_row["energy"], 6, " kWh"))
    with cols[5]:
        st.metric("{} predicted CO₂".format(selected), fmt(selected_row.get("co2"), 6, " kg"))

    st.caption(
        "Recommended framework and inspected framework are different concepts: "
        "the first two cards describe the current predicted rank #1, while the "
        "four objective cards describe the framework chosen in 'Inspect framework'."
    )

    view_mode = st.segmented_control(
        "3D view",
        ["Raw objective space", "Normalized desirability space"],
        default="Normalized desirability space",
        key="r95_decision_3d_view",
        help=(
            "Raw objective space preserves physical units. Normalized desirability rescales all three axes to 0–1 and inverts cost axes so higher is always better, which makes framework separation easier to inspect."
        ),
    )

    left, right = st.columns([1.65, 0.85])
    with left:
        if view_mode == "Normalized desirability space":
            fig3d = decision_space_3d_normalized(ranked, selected)
        else:
            fig3d = decision_space_3d(ranked, selected)
        plot(fig3d, "r9_decision_3d", height=660)
        st.caption(
            "Rotate and zoom directly in the Plotly scene. White outline = inspected framework; "
            "green outline = ε-Pareto candidate (ε=0.05). Normalized desirability is only a visualization transform; "
            "ranking still uses the selected point/conservative evidence mode and the exact preference weights."
        )
    with right:
        plot(ranking_bar(ranked), "r9_rank_bar")
        st.markdown(
            """
            <div class="r9-callout" style="margin-bottom:12px">
              <b>Why the ranking changes immediately:</b> the page does not rerun any framework.
              It keeps the same five predicted outcome profiles and only recomputes the weighted utility.
            </div>
            """,
            unsafe_allow_html=True,
        )
        for warning in meta.get("warnings") or []:
            st.warning(warning)
        st.markdown(
            """
            <div class="r9-callout"><b>Interpretation:</b> these are pre-run
            predictions from the frozen ML recommender, not observed outcomes.</div>
            """,
            unsafe_allow_html=True,
        )

    section(
        "Predicted objective evidence",
        "Intervals come from the Phase-6 empirical LODO residual calibration.",
    )
    display_cols = [
        "rank", "framework", "utility", "near_pareto",
        "accuracy", "accuracy_lower", "accuracy_upper",
        "runtime", "runtime_lower", "runtime_upper",
        "energy", "energy_lower", "energy_upper",
        "co2", "co2_lower", "co2_upper",
    ]
    st.dataframe(
        ranked[[c for c in display_cols if c in ranked.columns]],
        use_container_width=True,
        hide_index=True,
    )

    if profile is not None:
        with st.expander("Profile context used by the frozen recommender", expanded=False):
            st.json(profile, expanded=False)



def run_studio_v2_page():
    state = ensure_research_state()
    hero(
        "EXPERIMENT CONFIGURATION",
        "Run Studio",
        (
            "Configure the shared temporal experiment, then reuse the same "
            "dataset and results across Decision Space, Observatory, Copilot, "
            "Responsible AI and Export workspaces."
        ),
        pills=phase_pills(),
    )

    if dataset_ready():
        try:
            ranked, _, _ = compute_active_recommendation()
        except Exception:
            ranked = None
        if isinstance(ranked, pd.DataFrame) and not ranked.empty:
            top = ranked.iloc[0]
            cols = st.columns(4)
            with cols[0]:
                st.metric("Pre-run recommendation", str(top["framework"]))
            with cols[1]:
                st.metric("Predicted accuracy", fmt(top["accuracy"], 4))
            with cols[2]:
                st.metric("Predicted runtime", fmt(top["runtime"], 3, " s"))
            with cols[3]:
                st.metric("Pre-run preference utility", fmt(top["utility"], 4))

    st.markdown(
        """
        <div class="r9-callout">
          The experiment runner below is the validated AwareML temporal
          execution surface. Phase 9 wraps it in shared Research OS state
          rather than reimplementing the five framework adapters.
        </div>
        """,
        unsafe_allow_html=True,
    )

    advisor_slot = st.container()

    legacy_pages.run_studio_page()

    with advisor_slot:
        active = ensure_research_state()
        render_dataset_task_guard(active)
        render_dataset_advisor(
            active.get("dataset"),
            active.get("dataset_name"),
            active.get("target"),
        )
