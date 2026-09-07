from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
import streamlit as st

from awareml.explanation_integrity.evidence import flatten_evidence
from awareml.explanation_integrity.faithfulness import (
    FaithfulnessV2Evaluator,
    Intervention,
    OllamaEvidenceExplanationGenerator,
)
from awareml.explanation_integrity.live import build_live_cases
from awareml.explanation_integrity.live_guided import (
    LiveGuidedOllamaGenerator,
    LiveReferenceExplanationGenerator,
)
from awareml.explanation_integrity.verifier import GeneralEvidenceVerifier

from .components import hero, section
from .phase15_live_batch_ui import render_phase15_live_batch
from .page_utils import phase_pills
from .state import ensure_research_state


ROOT = Path(__file__).resolve().parents[2]

# The CLI benchmark already uses a longer timeout. The exploratory UI now uses
# the same runtime assumption instead of falling back to the old 60-second
# OllamaClient default.
LIVE_OLLAMA_TIMEOUT_SEC = 300
LIVE_OLLAMA_RETRIES = 1


class _RetryingLiveGenerator:
    """Guided exploratory generator; separate from the empirical benchmark."""

    source = "ollama-phase15-live-guided"

    def __init__(
        self,
        timeout_sec: float = LIVE_OLLAMA_TIMEOUT_SEC,
        retries: int = LIVE_OLLAMA_RETRIES,
    ):
        self.base = LiveGuidedOllamaGenerator(
            timeout_sec=float(timeout_sec),
            network_retries=max(0, int(retries)),
        )
        self.model = getattr(self.base, "model", None)

    def generate(self, case):
        text, meta = self.base.generate(case)
        self.model = (
            meta.get("model")
            if isinstance(meta, Mapping)
            else self.model
        )
        return text, meta


def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _metric(value, digits=3):
    if value is None:
        return "N/A"
    try:
        return ("{:.%df}" % digits).format(float(value))
    except Exception:
        return str(value)


def _frozen_paths():
    base = ROOT / "data" / "journal"
    return {
        "correctness": base / "explanation_correctness_v1" / "frozen",
        "faithfulness": base / "faithfulness_v2" / "frozen",
        "stimuli": base / "trust_stimulus_bank_v1" / "frozen",
    }


def _controlled_benchmark_tab():
    paths = _frozen_paths()
    correctness_manifest = _load_json(
        paths["correctness"] / "manifest.json", {}
    )
    faithfulness_manifest = _load_json(
        paths["faithfulness"] / "manifest.json", {}
    )
    correctness_summary = _load_json(
        paths["correctness"] / "summary.json", {}
    )
    faithfulness_summary = _load_json(
        paths["faithfulness"] / "summary.json", {}
    )

    section(
        "Controlled benchmark · frozen journal evidence",
        (
            "Correctness and faithfulness are reported separately. "
            "These frozen artifacts validate the verifier/intervention methodology "
            "on known controlled examples; they are not presented as an empirical "
            "performance estimate for a particular LLM."
        ),
    )

    with st.expander("How to use this tab", expanded=True):
        st.markdown(
            """
**Question answered by this tab:** *Does the Phase-15 evaluator itself behave correctly on cases where the ground truth is known?*

- **Known-correct** explanations should score very highly because every factual claim was constructed from the evidence.
- **Known-incorrect** explanations are deliberately *mostly correct with one seeded error*. Their claim precision therefore does **not** need to be zero.
- **Known-faithful** explanations should react to the intervention.
- **Sticky control** intentionally ignores the changed evidence and should remain near zero.
- **Irrelevant control** should remain stable when an irrelevant value changes.

These numbers are **not Dutch Census results** and are **not Llama-3 performance results**. They are frozen methodology checks.
"""
        )

    c1, c2, c3, c4 = st.columns(4)
    label_summary = correctness_summary.get("by_label") or {}
    known_correct = label_summary.get("known_correct") or {}
    known_incorrect = label_summary.get("known_incorrect") or {}
    faithful = faithfulness_summary.get("known_faithful") or {}
    sticky = faithfulness_summary.get("known_unfaithful_sticky") or {}

    c1.metric(
        "Known-correct claim precision",
        _metric(known_correct.get("claim_precision")),
    )
    c2.metric(
        "Known-correct numeric correctness",
        _metric(known_correct.get("numeric_correctness")),
    )
    c3.metric(
        "Known-faithful intervention score",
        _metric(faithful.get("mean_faithfulness_score")),
    )
    c4.metric(
        "Sticky-control intervention score",
        _metric(sticky.get("mean_faithfulness_score")),
    )

    st.markdown("#### Correctness · factual agreement with structured evidence")
    st.caption(
        "Metrics: claim precision, supported-claim rate, numeric correctness, "
        "citation validity, decision consistency, unsupported-claim rate and "
        "contradiction rate."
    )
    rows = []
    for label, values in sorted(label_summary.items()):
        rows.append({"Condition": label, **values})
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Faithfulness · response to evidence interventions")
    st.caption(
        "The evaluator changes evidence supplied to the explanation generator "
        "and checks whether the explanation updates. This is not the same "
        "question as factual correctness."
    )
    rows = []
    for condition, values in sorted(faithfulness_summary.items()):
        rows.append({"Condition": condition, **values})
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Frozen artifact manifests", expanded=False):
        st.json(
            {
                "explanation_correctness_v1": correctness_manifest,
                "faithfulness_v2": faithfulness_manifest,
            },
            expanded=False,
        )


def _render_correctness_report(report):
    metrics = report.metrics
    cols = st.columns(4)
    cols[0].metric("Claim precision", _metric(metrics.get("claim_precision")))
    cols[1].metric(
        "Numeric correctness",
        _metric(metrics.get("numeric_correctness")),
    )
    cols[2].metric(
        "Citation validity",
        _metric(metrics.get("citation_validity")),
    )
    cols[3].metric(
        "Contradiction rate",
        _metric(metrics.get("contradiction_rate")),
    )

    st.caption(
        "Claim precision asks how many extracted factual claims are supported. "
        "Numeric correctness checks extracted numbers only. Citation validity "
        "is N/A when the explanation contains no evidence citations."
    )

    rows = []
    for item in report.claims:
        rows.append({
            "Claim": item.claim.text,
            "Type": item.claim.claim_type,
            "Metric": item.claim.metric,
            "Entity / feature": item.claim.entity or item.claim.feature,
            "Status": item.status,
            "Expected": item.expected_value,
            "Observed": item.observed_value,
            "Evidence key": item.matched_evidence_key,
            "Citation valid": item.citation_valid,
            "Reason": item.reason,
        })
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No Phase-15 factual claims were extracted from this explanation."
        )


def _source_guidance(case):
    if case.source_stage == "B":
        return (
            "Stage B · recommendation",
            (
                "This is **pre-run recommendation evidence**. Check which framework "
                "is ranked first and whether the explanation correctly reports the "
                "predicted accuracy, runtime, energy, CO₂ and utility/ranking evidence."
            ),
        )
    if case.source_stage == "E":
        return (
            "Stage E · fairness",
            (
                "This is **post-run fairness evidence**. Check DP/SPD, Equal Opportunity, "
                "Equalized Odds, and—when probability evidence exists—Group Brier and "
                "Group ECE gaps. Unavailable metrics must remain N/A, never zero."
            ),
        )
    if case.source_stage == "F_XAI":
        return (
            "Stage F · XAI",
            (
                "This is **post-run feature-attribution evidence**. Verify the method "
                "actually present in the structured evidence. Exact SHAP-value claims "
                "are valid only when literal `shap_values` are present."
            ),
        )
    return (
        "Stage F · conversational answer",
        (
            "This is a **concrete active-results question**: which framework is ranked "
            "first, and which reported accuracy/runtime/energy/CO₂ values support it? "
            "The live prompt now asks that question explicitly."
        ),
    )


def _fairness_summary(case):
    frameworks = case.evidence.get("frameworks") or {}
    rows = []
    for framework, payload in frameworks.items():
        fair = (payload or {}).get("fairness") or {}
        rows.append({
            "Framework": framework,
            "Status": fair.get("status"),
            "DP/SPD": fair.get("dp_diff"),
            "EO": fair.get("equal_opportunity_diff"),
            "EOdds": fair.get("equalized_odds_gap"),
            "Brier gap": fair.get("group_brier_score_gap"),
            "ECE gap": fair.get("group_ece_gap"),
            "Calibration": fair.get("calibration_status"),
        })
    if rows:
        st.markdown("##### Fairness evidence summary")
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )


def _xai_disclosure(case):
    exp = case.evidence.get("explainability") or {}
    method = exp.get("method")
    shap_values = exp.get("shap_values")
    feature_importance = exp.get("feature_importance")

    if shap_values is None and feature_importance is not None:
        st.info(
            "XAI disclosure: this run reports method **{}**, but the structured "
            "evidence contains no literal `shap_values`. AwareML will therefore "
            "treat the available `feature_importance` as generic feature-attribution "
            "evidence and will not pretend that those values are exact SHAP values."
            .format(method or "unknown")
        )


def _is_numeric(value):
    if isinstance(value, bool):
        return False
    try:
        float(value)
        return True
    except Exception:
        return False


def _append_existing(target, ordered, seen):
    if target in seen:
        return
    ordered.append(target)
    seen.add(target)


def _preferred_intervention_keys(case, flat):
    """Return only source-relevant numeric keys, ordered by usefulness."""
    numeric = [
        key
        for key, value in flat.items()
        if _is_numeric(value)
    ]
    numeric_set = set(numeric)
    ordered = []
    seen = set()

    if case.source_stage == "B":
        top = (
            (case.evidence.get("recommendation") or {})
            .get("top_framework")
        )
        if top:
            for metric in ("accuracy", "utility", "runtime", "energy", "co2"):
                key = "evidence.candidates.{}.{}".format(top, metric)
                if key in numeric_set:
                    _append_existing(key, ordered, seen)

        allowed_suffixes = (
            ".accuracy",
            ".utility",
            ".runtime",
            ".energy",
            ".co2",
        )
        for key in numeric:
            if (
                key.startswith("evidence.candidates.")
                and key.endswith(allowed_suffixes)
            ):
                _append_existing(key, ordered, seen)

    elif case.source_stage == "E":
        focus = case.metadata.get("focus_framework")
        metrics = (
            "dp_diff",
            "equal_opportunity_diff",
            "equalized_odds_gap",
            "group_brier_score_gap",
            "group_ece_gap",
        )
        if focus:
            for metric in metrics:
                key = (
                    "evidence.frameworks.{}.fairness.{}"
                    .format(focus, metric)
                )
                if key in numeric_set:
                    _append_existing(key, ordered, seen)

        for key in numeric:
            if (
                ".fairness." in key
                and any(key.endswith("." + metric) for metric in metrics)
            ):
                _append_existing(key, ordered, seen)

    elif case.source_stage == "F_XAI":
        for key in numeric:
            if ".shap_values." in key:
                _append_existing(key, ordered, seen)
        for key in numeric:
            if (
                ".feature_importance." in key
                and key.endswith(".importance")
            ):
                _append_existing(key, ordered, seen)
        for metric in ("fidelity", "stability", "consistency"):
            key = "evidence.explainability.{}".format(metric)
            if key in numeric_set:
                _append_existing(key, ordered, seen)

    else:  # F_CHAT
        ranking = case.evidence.get("ranking") or []
        top = None
        if ranking:
            ranked = sorted(
                ranking,
                key=lambda row: (
                    float(row.get("rank"))
                    if row.get("rank") is not None
                    else 1e9
                ),
            )
            top = ranked[0].get("framework")

        metrics = (
            "accuracy",
            "f1_macro",
            "runtime_sec",
            "energy_kwh",
            "co2_kg",
        )
        if top:
            for metric in metrics:
                key = "evidence.frameworks.{}.{}".format(top, metric)
                if key in numeric_set:
                    _append_existing(key, ordered, seen)

        for key in numeric:
            if (
                key.startswith("evidence.frameworks.")
                and any(key.endswith("." + metric) for metric in metrics)
            ):
                _append_existing(key, ordered, seen)

    # If a framework did not expose the preferred metric family, keep the probe
    # usable with the remaining source-specific numeric evidence.
    if not ordered:
        ordered = numeric

    return ordered


def _live_probe_tab():
    st.warning(
        "Exploratory Live Dataset Probe · NOT frozen journal evidence. "
        "Use this page to inspect the active AwareML run. The controlled and "
        "empirical Phase-15 benchmarks remain separate."
    )

    with st.expander("How to use this page", expanded=True):
        st.markdown(
            """
**Normal workflow — only three steps**

1. Click **Run complete Phase-15 live check**.
2. Read the **Result summary** for Stage B, Stage E, Stage F XAI and Stage F conversational.
3. Open a source only when it is marked **REVIEW** or **FAILED**.

If **Also test faithfulness for all four sources** is enabled, the batch already
performs the faithfulness interventions. You do **not** need to run the four
stages manually afterward.

**Correctness** = are the explanation's factual claims supported by the active
structured evidence?

**Faithfulness** = when one relevant evidence value changes, does the
explanation update instead of repeating the old value?
"""
        )

    state = ensure_research_state()
    cases = build_live_cases(state)

    if not cases:
        st.info(
            "Run AwareML first so Stage B recommendation and post-run Stage E/F "
            "evidence are available."
        )
        return

    dataset_name = str(state.get("dataset_name") or "active dataset")
    st.caption("Active dataset · {}".format(dataset_name))

    batch_result = render_phase15_live_batch(
        state,
        ROOT,
        timeout_sec=LIVE_OLLAMA_TIMEOUT_SEC,
        network_retries=LIVE_OLLAMA_RETRIES,
    )

    st.divider()

    # Keep the manual path available for researchers, but never place it
    # inside an expander. Streamlit forbids nested expanders and this manual
    # section itself contains expandable evidence/details.
    show_advanced = st.checkbox(
        "Show advanced manual verification",
        value=False,
        key="p15_live_v9_show_advanced",
        help=(
            "Optional researcher tool. The normal four-source batch above "
            "already checks correctness and faithfulness."
        ),
    )
    if show_advanced:
        st.markdown("### Advanced manual verification")
        st.caption(
            "Use this only to verify a custom explanation or try a different "
            "counterfactual evidence value. The normal four-source batch above "
            "already covers correctness and, when enabled, faithfulness."
        )

        display_labels = {
            "B": "B · Stage B · setup / recommendation",
            "E": "E · Stage E · fairness",
            "F_XAI": "F_XAI · Stage F · XAI / feature attribution",
            "F_CHAT": "F_CHAT · Stage F · conversational answer",
        }

        labels = {
            display_labels.get(
                case.source_stage,
                "{} · {}".format(
                    case.source_stage,
                    case.source_name,
                ),
            ): case
            for case in cases
        }

        selected = st.selectbox(
            "Source to inspect",
            list(labels.keys()),
            key="p15_live_v9_manual_source",
        )
        case = labels[selected]

        title, guidance = _source_guidance(case)
        st.info("**{}** — {}".format(title, guidance))

        with st.expander(
            "Structured evidence for this source",
            expanded=False,
        ):
            st.json(case.evidence, expanded=False)

        if case.source_stage == "E":
            _fairness_summary(case)
        elif case.source_stage == "F_XAI":
            _xai_disclosure(case)

        batch_stage = None
        if isinstance(batch_result, Mapping):
            for row in batch_result.get("stages") or []:
                if row.get("case_id") == case.case_id:
                    batch_stage = row
                    break

        batch_explanation = ""
        if (
            isinstance(batch_stage, Mapping)
            and batch_stage.get("llm_generation_status") == "ok"
            and batch_stage.get("llm_explanation")
        ):
            batch_explanation = str(
                batch_stage.get("llm_explanation")
            )

        st.markdown("#### Explanation")
        mode_options = []
        if batch_explanation:
            mode_options.append(
                "Use explanation already generated by batch"
            )
        mode_options.extend([
            "Generate a new Ollama explanation",
            "Paste a custom explanation",
        ])

        mode = st.radio(
            "Explanation source",
            mode_options,
            horizontal=False,
            key="p15_live_v9_manual_mode_" + case.case_id,
        )

        explanation = ""

        if mode == "Use explanation already generated by batch":
            explanation = batch_explanation
            st.success(
                "Reusing the batch explanation. No second LLM generation is needed."
            )
            st.write(explanation)

        elif mode == "Generate a new Ollama explanation":
            generated_key = (
                "p15_live_v9_manual_generated_" + case.case_id
            )

            if st.button(
                "Generate new explanation",
                key="p15_live_v9_manual_generate_" + case.case_id,
            ):
                try:
                    with st.spinner("Generating live explanation..."):
                        generator = _RetryingLiveGenerator()
                        text, meta = generator.generate(case)
                    st.session_state[generated_key] = text
                except Exception as exc:
                    st.error(
                        "No usable grounded explanation was generated: {}: {}"
                        .format(type(exc).__name__, exc)
                    )

            explanation = str(
                st.session_state.get(generated_key) or ""
            ).strip()

            if explanation:
                st.write(explanation)
            else:
                st.caption(
                    "Generate an explanation above before verifying it."
                )

        else:
            explanation = st.text_area(
                "Custom explanation to verify",
                key=(
                    "p15_live_v9_manual_custom_"
                    + case.case_id
                ),
                height=150,
                placeholder=(
                    "Paste an explanation from another source. "
                    "AwareML will compare its factual claims with the "
                    "structured evidence above."
                ),
            ).strip()

        if explanation:
            st.markdown("#### Correctness")
            verifier = GeneralEvidenceVerifier()
            report = verifier.verify(case, explanation)
            _render_correctness_report(report)

            supported_claims = [
                item
                for item in report.claims
                if item.supported
            ]
            supported_numeric = [
                item
                for item in supported_claims
                if item.claim.claim_type == "numeric"
            ]
            intervention_ready = (
                bool(supported_numeric)
                if case.source_stage == "E"
                else bool(supported_claims)
            )

            if st.checkbox(
                "Try a different faithfulness intervention",
                value=False,
                key=(
                    "p15_live_v9_manual_intervention_toggle_"
                    + case.case_id
                ),
                help=(
                    "Optional. The batch already performed one intervention "
                    "when the faithfulness checkbox was enabled."
                ),
            ):
                flat = flatten_evidence(case.evidence)
                numeric_keys = _preferred_intervention_keys(
                    case,
                    flat,
                )

                if not intervention_ready:
                    st.info(
                        "The current explanation does not contain a supported "
                        "claim that can be used as a faithfulness baseline."
                    )
                elif not numeric_keys:
                    st.info(
                        "No source-relevant numeric evidence is available "
                        "for an intervention."
                    )
                else:
                    intervention_key = st.selectbox(
                        "Evidence value to change",
                        numeric_keys,
                        key=(
                            "p15_live_v9_manual_intervention_key_"
                            + case.case_id
                        ),
                    )

                    original_value = float(flat[intervention_key])
                    default_cf = (
                        original_value * 0.8
                        if abs(original_value) > 1e-12
                        else 0.1
                    )

                    c1, c2 = st.columns(2)
                    c1.metric(
                        "Current value",
                        "{:.8g}".format(original_value),
                    )
                    with c2:
                        counterfactual = st.number_input(
                            "Counterfactual value",
                            value=float(default_cf),
                            format="%.8f",
                            key=(
                                "p15_live_v9_manual_intervention_value_"
                                + case.case_id
                            ),
                        )

                    faith_key = (
                        "p15_live_v9_manual_faith_"
                        + case.case_id
                    )

                    if st.button(
                        "Run manual counterfactual",
                        key=(
                            "p15_live_v9_manual_intervention_run_"
                            + case.case_id
                        ),
                    ):
                        try:
                            intervention = Intervention(
                                intervention_id="LIVE_{}".format(
                                    datetime.now(timezone.utc).strftime(
                                        "%H%M%S"
                                    )
                                ),
                                kind="live_numeric_counterfactual",
                                evidence_key=intervention_key,
                                original_value=original_value,
                                counterfactual_value=float(
                                    counterfactual
                                ),
                                relevant=True,
                                metadata={
                                    "exploratory_live_probe": True,
                                    "claim_level_sensitivity": (
                                        case.source_stage
                                        in {"B", "F_CHAT"}
                                    ),
                                },
                            )

                            evaluator = FaithfulnessV2Evaluator()
                            generator = _RetryingLiveGenerator()

                            with st.spinner(
                                "Generating counterfactual response..."
                            ):
                                record = evaluator.evaluate(
                                    case,
                                    intervention,
                                    generator,
                                    original_explanation=explanation,
                                )

                            st.session_state[
                                faith_key
                            ] = record.to_dict()

                        except Exception as exc:
                            st.error(
                                "Counterfactual generation failed: {}: {}"
                                .format(type(exc).__name__, exc)
                            )

                    faith = st.session_state.get(faith_key)
                    if isinstance(faith, Mapping):
                        st.markdown("##### Faithfulness result")
                        cols = st.columns(4)
                        cols[0].metric(
                            "Faithfulness",
                            _metric(
                                faith.get("faithfulness_score")
                            ),
                        )
                        cols[1].metric(
                            "Changed evidence",
                            _metric(
                                faith.get(
                                    "changed_evidence_acknowledged"
                                )
                            ),
                        )
                        cols[2].metric(
                            "Numeric update",
                            _metric(
                                faith.get(
                                    "numeric_update_accuracy"
                                )
                            ),
                        )
                        cols[3].metric(
                            "Stale-claim rate",
                            _metric(
                                faith.get("stale_claim_rate")
                            ),
                        )

                        st.markdown("**Counterfactual response**")
                        st.write(
                            faith.get(
                                "counterfactual_explanation"
                            )
                        )
        else:
            st.info(
                "No explanation selected yet. The normal workflow does not "
                "require this advanced section."
            )


def _stimulus_tab():
    paths = _frozen_paths()
    summary = _load_json(paths["stimuli"] / "summary.json", {})
    researcher = _load_json(
        paths["stimuli"] / "stimuli_researcher.json", []
    )
    participant = _load_json(
        paths["stimuli"] / "stimuli_participant.json", []
    )
    manifest = _load_json(paths["stimuli"] / "manifest.json", {})

    section(
        "Track 2 · trust-calibration stimulus bank",
        (
            "Known-correct and known-incorrect explanation pairs are balanced "
            "across Stage B, Stage E, Stage F XAI and Stage F conversational "
            "sources. Participant-facing exports intentionally omit correctness "
            "labels."
        ),
    )

    with st.expander("How to use this tab", expanded=True):
        st.markdown(
            """
**Question answered by this tab:** *What explanation material will be shown in the human trust-calibration study?*

- The bank contains **paired known-correct and known-incorrect explanations**.
- The **participant-facing** export is blinded: participants do not see the correctness label, error type or verifier score.
- The **researcher-only** section contains the ground-truth condition and error type for later analysis.
- This bank is **not generated from the current Dutch run**. It is a controlled Track-2 study asset.
- Do not expose the researcher-labeled file to participants.
"""
        )

    labels = summary.get("labels") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Known-correct", labels.get("known_correct", 0))
    c2.metric("Known-incorrect", labels.get("known_incorrect", 0))
    c3.metric("Dataset contexts", summary.get("dataset_contexts", 0))

    sources = summary.get("sources") or {}
    if sources:
        st.dataframe(
            pd.DataFrame([
                {"Explanation source": key, "Stimuli": value}
                for key, value in sorted(sources.items())
            ]),
            use_container_width=True,
            hide_index=True,
        )

    if participant:
        st.download_button(
            "Download participant-facing stimulus bank",
            data=json.dumps(
                participant,
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8"),
            file_name="trust_stimulus_bank_v1_participant.json",
            mime="application/json",
            key="p15_stimulus_participant_download",
        )

    with st.expander(
        "Researcher-only labels and verifier evidence",
        expanded=False,
    ):
        st.warning(
            "Do not expose this researcher-labeled file to study participants."
        )
        if researcher:
            preview = pd.DataFrame(researcher)
            columns = [
                col for col in [
                    "stimulus_id",
                    "pair_id",
                    "dataset_id",
                    "source_name",
                    "researcher_label",
                    "error_type",
                ]
                if col in preview.columns
            ]
            st.dataframe(
                preview[columns],
                use_container_width=True,
                hide_index=True,
            )
        st.json(manifest, expanded=False)


def phase15_explanation_integrity_page():
    hero(
        "PHASE 15 · EXPLANATION INTEGRITY",
        "Explanation Correctness & Faithfulness V2",
        (
            "Correctness asks whether explanation claims agree with structured "
            "evidence. Faithfulness asks whether the explanation changes when "
            "decision-relevant evidence changes. AwareML keeps these questions "
            "separate."
        ),
        pills=phase_pills(),
    )

    st.caption(
        "Three views, three different purposes: Controlled benchmark = validate "
        "the evaluator; Live Dataset Probe = inspect the current AwareML run; "
        "Track 2 stimulus bank = prepare the human trust-calibration study."
    )

    tabs = st.tabs([
        "Controlled benchmark",
        "Live Dataset Probe · exploratory",
        "Track 2 stimulus bank",
    ])

    with tabs[0]:
        _controlled_benchmark_tab()
    with tabs[1]:
        _live_probe_tab()
    with tabs[2]:
        _stimulus_tab()
