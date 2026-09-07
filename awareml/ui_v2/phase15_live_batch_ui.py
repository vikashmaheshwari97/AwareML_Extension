from __future__ import annotations

import json
from typing import Mapping

import pandas as pd
import streamlit as st

from awareml.explanation_integrity.live_batch import (
    run_live_batch,
    save_live_batch,
)


def _correctness_status(stage):
    if stage.get("llm_generation_status") != "ok":
        return "FAILED"

    metrics = ((stage.get("correctness") or {}).get("metrics") or {})
    if not metrics:
        return "FAILED"

    contradiction = metrics.get("contradiction_rate")
    numeric = metrics.get("numeric_correctness")
    unsupported = metrics.get("unsupported_claim_rate")
    citation = metrics.get("citation_validity")

    try:
        if contradiction is not None and float(contradiction) > 1e-12:
            return "REVIEW"
    except Exception:
        return "REVIEW"

    try:
        if numeric is not None and float(numeric) < 0.999999:
            return "REVIEW"
    except Exception:
        return "REVIEW"

    try:
        if unsupported is not None and float(unsupported) > 1e-12:
            return "REVIEW"
    except Exception:
        return "REVIEW"

    try:
        if citation is not None and float(citation) < 0.999999:
            return "REVIEW"
    except Exception:
        return "REVIEW"

    return "OK"


def _faithfulness_status(stage, included):
    if not included:
        return "NOT RUN"

    if stage.get("faithfulness_status") != "ok":
        return "FAILED"

    faith = stage.get("faithfulness") or {}
    changed = faith.get("changed_evidence_acknowledged")
    stale = faith.get("stale_claim_rate")
    numeric = faith.get("numeric_update_accuracy")

    try:
        if changed is not None and float(changed) < 0.999999:
            return "REVIEW"
    except Exception:
        return "REVIEW"

    try:
        if stale is not None and float(stale) > 1e-12:
            return "REVIEW"
    except Exception:
        return "REVIEW"

    try:
        if numeric is not None and float(numeric) < 0.999999:
            return "REVIEW"
    except Exception:
        return "REVIEW"

    return "OK"


def _overall(correctness, faithfulness):
    values = {correctness, faithfulness}
    if "FAILED" in values:
        return "FAILED"
    if "REVIEW" in values:
        return "REVIEW"
    return "OK"


def _stage_help(stage_label):
    text = str(stage_label or "")
    if "Stage B" in text:
        return (
            "Checks the pre-run recommendation and predicted framework evidence."
        )
    if "fairness" in text.lower():
        return (
            "Checks DP/SPD, Equal Opportunity, Equalized Odds and calibration gaps."
        )
    if "XAI" in text:
        return (
            "Checks feature-attribution evidence; generic importance is not "
            "silently relabelled as exact SHAP."
        )
    return (
        "Checks a post-run conversational answer against the active framework results."
    )


def render_phase15_live_batch(
    state,
    root,
    timeout_sec=300,
    network_retries=1,
):
    st.markdown("### Phase-15 live check")
    st.caption(
        "Recommended workflow: run all four explanation sources together, "
        "read the summary, and inspect details only when a source is marked "
        "REVIEW or FAILED."
    )

    with st.container(border=True):
        st.markdown("**What will be checked**")
        source_rows = [
            {
                "Source": "Stage B · recommendation",
                "Meaning": "Pre-run recommendation and predicted metrics",
            },
            {
                "Source": "Stage E · fairness",
                "Meaning": "Fairness + calibration-gap explanation",
            },
            {
                "Source": "Stage F · XAI",
                "Meaning": "Feature attribution / XAI explanation",
            },
            {
                "Source": "Stage F · conversational",
                "Meaning": "Post-run conversational answer",
            },
        ]
        st.dataframe(
            pd.DataFrame(source_rows),
            use_container_width=True,
            hide_index=True,
        )

        include_interventions = st.checkbox(
            "Also test faithfulness for all four sources",
            value=True,
            key="p15_live_v9_batch_include_interventions",
            help=(
                "AwareML changes one relevant numeric evidence value for each "
                "source and checks whether the LLM explanation updates."
            ),
        )

        st.info(
            "A source is marked FAILED only when no usable grounded LLM output "
            "is produced. Imperfect claims are now kept and scored as REVIEW "
            "instead of being incorrectly discarded as grounding failures."
        )

        if st.button(
            "Run complete Phase-15 live check",
            key="p15_live_v9_batch_run",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Running four live sources"
                    + (
                        " plus four faithfulness interventions..."
                        if include_interventions
                        else "..."
                    )
                ):
                    result = run_live_batch(
                        state,
                        include_interventions=include_interventions,
                        timeout_sec=timeout_sec,
                        network_retries=network_retries,
                    )
                    path = save_live_batch(result, root)

                st.session_state["p15_live_v9_batch_result"] = result
                st.session_state["p15_live_v9_batch_path"] = str(path)

            except Exception as exc:
                st.error(
                    "The Phase-15 live check could not start: {}: {}"
                    .format(type(exc).__name__, exc)
                )

    result = st.session_state.get("p15_live_v9_batch_result")
    if not isinstance(result, Mapping):
        st.markdown("#### How to read the result")
        st.markdown(
            """
- **OK** — the source completed and the displayed checks are clean.
- **REVIEW** — the LLM produced a usable explanation, but one or more factual
  or faithfulness metrics need inspection.
- **FAILED** — no usable explanation/intervention was produced; inspect the
  failure reason.
"""
        )
        return None

    included = bool(result.get("include_interventions"))
    simple_rows = []
    detailed_rows = []
    review_count = 0

    for stage in result.get("stages") or []:
        correctness_metrics = (
            (stage.get("correctness") or {}).get("metrics") or {}
        )
        faith = stage.get("faithfulness") or {}

        correctness_status = _correctness_status(stage)
        faithfulness_status = _faithfulness_status(stage, included)
        overall = _overall(correctness_status, faithfulness_status)

        if overall != "OK":
            review_count += 1

        simple_rows.append({
            "Source": stage.get("stage_label"),
            "Generation": (
                "OK"
                if stage.get("llm_generation_status") == "ok"
                else stage.get("llm_generation_status")
            ),
            "Correctness": correctness_status,
            "Faithfulness": faithfulness_status,
            "Overall": overall,
        })

        detailed_rows.append({
            "Source": stage.get("stage_label"),
            "Claim precision": correctness_metrics.get("claim_precision"),
            "Numeric correctness": correctness_metrics.get(
                "numeric_correctness"
            ),
            "Unsupported rate": correctness_metrics.get(
                "unsupported_claim_rate"
            ),
            "Contradiction rate": correctness_metrics.get(
                "contradiction_rate"
            ),
            "Faithfulness score": faith.get("faithfulness_score"),
            "Changed evidence": faith.get(
                "changed_evidence_acknowledged"
            ),
            "Numeric update": faith.get("numeric_update_accuracy"),
            "Stale-claim rate": faith.get("stale_claim_rate"),
        })

    st.markdown("### Result summary")
    st.dataframe(
        pd.DataFrame(simple_rows),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Explanations generated",
        "{}/{}".format(
            result.get("generation_ok_count", 0),
            result.get("stage_count", 0),
        ),
    )
    c2.metric(
        "Faithfulness tests completed",
        (
            "{}/{}".format(
                result.get("faithfulness_ok_count", 0),
                result.get("stage_count", 0),
            )
            if included
            else "Not requested"
        ),
    )
    c3.metric(
        "Sources needing review",
        "{}/{}".format(
            review_count,
            result.get("stage_count", 0),
        ),
    )

    st.info(
        "**What REVIEW means:** the LLM produced a usable explanation, but the "
        "deterministic verifier found at least one claim that needs inspection. "
        "You do not manually approve a REVIEW result. If the mismatch is a real "
        "LLM error, keep it as REVIEW and report it. If it is a verifier/parser "
        "false positive, fix the verifier before the final empirical benchmark."
    )

    st.caption(
        "Correctness and faithfulness are separate. REVIEW is an explanation-"
        "quality outcome, not a pipeline failure."
    )

    with st.expander(
        "Detailed metrics · open only if you need them",
        expanded=False,
    ):
        st.dataframe(
            pd.DataFrame(detailed_rows),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Correctness: numeric correctness should ideally be 1, unsupported "
            "and contradiction rates 0. Faithfulness: changed evidence should "
            "be 1, numeric update 1 when applicable, stale-claim rate 0."
        )

    st.markdown("### Results by source")
    st.caption(
        "You do not need to run the stages again. Open a source here only to "
        "read the generated explanation, intervention and counterfactual response."
    )

    for stage in result.get("stages") or []:
        c_status = _correctness_status(stage)
        f_status = _faithfulness_status(stage, included)
        overall = _overall(c_status, f_status)

        label = "{} · {}".format(
            stage.get("stage_label"),
            overall,
        )

        with st.expander(label, expanded=(overall != "OK")):
            st.caption(_stage_help(stage.get("stage_label")))

            if stage.get("llm_explanation"):
                st.markdown("**LLM explanation**")
                st.write(stage.get("llm_explanation"))
            else:
                st.error(
                    stage.get("error")
                    or "No usable grounded LLM explanation was produced."
                )
                reference = stage.get("reference_explanation")
                if reference:
                    st.markdown("**Evidence template · not LLM performance**")
                    st.write(reference)
                continue

            correctness = stage.get("correctness")
            if isinstance(correctness, Mapping):
                metrics = correctness.get("metrics") or {}
                a, b, c, d = st.columns(4)
                a.metric(
                    "Claim precision",
                    "N/A"
                    if metrics.get("claim_precision") is None
                    else "{:.3f}".format(float(metrics.get("claim_precision"))),
                )
                b.metric(
                    "Numeric correctness",
                    "N/A"
                    if metrics.get("numeric_correctness") is None
                    else "{:.3f}".format(float(metrics.get("numeric_correctness"))),
                )
                c.metric(
                    "Unsupported rate",
                    "N/A"
                    if metrics.get("unsupported_claim_rate") is None
                    else "{:.3f}".format(float(metrics.get("unsupported_claim_rate"))),
                )
                d.metric(
                    "Contradiction rate",
                    "N/A"
                    if metrics.get("contradiction_rate") is None
                    else "{:.3f}".format(float(metrics.get("contradiction_rate"))),
                )

                if c_status == "REVIEW":
                    issue_rows = []
                    for item in correctness.get("claims") or []:
                        if not isinstance(item, Mapping):
                            continue
                        if item.get("status") == "supported":
                            continue
                        claim = item.get("claim") or {}
                        issue_rows.append({
                            "Claim": claim.get("text"),
                            "Status": item.get("status"),
                            "Metric": claim.get("metric"),
                            "Expected": item.get("expected_value"),
                            "Observed": item.get("observed_value"),
                            "Reason": item.get("reason"),
                        })

                    if issue_rows:
                        st.markdown("**Why this source is marked REVIEW**")
                        st.dataframe(
                            pd.DataFrame(issue_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

            if included:
                intervention = stage.get("intervention")
                faith = stage.get("faithfulness")

                st.markdown("**Faithfulness intervention**")
                if isinstance(intervention, Mapping):
                    st.write(
                        "{}: {} → {}".format(
                            intervention.get("evidence_key"),
                            intervention.get("original_value"),
                            intervention.get("counterfactual_value"),
                        )
                    )

                if isinstance(faith, Mapping):
                    st.markdown("**Counterfactual LLM response**")
                    st.write(faith.get("counterfactual_explanation"))

                    a, b, c, d = st.columns(4)
                    a.metric(
                        "Faithfulness score",
                        "N/A"
                        if faith.get("faithfulness_score") is None
                        else "{:.3f}".format(
                            float(faith.get("faithfulness_score"))
                        ),
                    )
                    b.metric(
                        "Changed evidence",
                        "N/A"
                        if faith.get("changed_evidence_acknowledged") is None
                        else "{:.3f}".format(
                            float(faith.get("changed_evidence_acknowledged"))
                        ),
                    )
                    c.metric(
                        "Numeric update",
                        "N/A"
                        if faith.get("numeric_update_accuracy") is None
                        else "{:.3f}".format(
                            float(faith.get("numeric_update_accuracy"))
                        ),
                    )
                    d.metric(
                        "Stale-claim rate",
                        "N/A"
                        if faith.get("stale_claim_rate") is None
                        else "{:.3f}".format(
                            float(faith.get("stale_claim_rate"))
                        ),
                    )
                elif stage.get("faithfulness_status") != "not_run":
                    st.warning(
                        stage.get("error")
                        or "The faithfulness intervention did not complete."
                    )

    saved = st.session_state.get("p15_live_v9_batch_path")
    if saved:
        st.caption("Saved exploratory run: {}".format(saved))

    st.download_button(
        "Download complete Phase-15 live JSON",
        data=json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8"),
        file_name="phase15_live_batch.json",
        mime="application/json",
        key="p15_live_v9_batch_download",
    )

    return result
