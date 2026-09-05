from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from awareml.analysis.repeatability import (
    hardware_table,
    summarize_repeatability,
)
from awareml.analysis.repeatability_registry import (
    PAPER_READY_MIN_REPETITIONS,
    canonical_dataframe_sha256,
    find_latest_matching_run,
    list_dataset_runs,
)


ARTIFACT_ROOT = Path("artifacts/phase14/repeatability")


def _awareml_state() -> dict[str, Any]:
    try:
        state = st.session_state.get("awareml_state") or {}
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def _active_dataframe() -> pd.DataFrame | None:
    state = _awareml_state()

    direct = state.get("dataset")
    if isinstance(direct, pd.DataFrame):
        return direct

    # Defensive fallback for future state refactors.
    for value in state.values():
        if isinstance(value, pd.DataFrame):
            return value

    for value in st.session_state.values():
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, pd.DataFrame):
                    return nested
    return None


def _active_context() -> dict[str, Any]:
    state = _awareml_state()
    df = _active_dataframe()

    content_hash = state.get("dataset_content_sha256")
    if not content_hash and isinstance(df, pd.DataFrame):
        try:
            content_hash = canonical_dataframe_sha256(df)
            state["dataset_content_sha256"] = content_hash
        except Exception:
            content_hash = None

    return {
        "dataset_name": state.get("dataset_name") or "dataset",
        "dataset_content_sha256": content_hash,
        "target": state.get("target"),
        "sensitive_attribute": state.get("sensitive"),
        "positive_label": state.get("positive_label"),
    }


def _fairness_rows(results):
    rows = []
    for result in results:
        fair = result.get("fairness") or {}
        rows.append({
            "Framework": result.get("framework"),
            "Prediction status": fair.get(
                "prediction_behavior_status", "unavailable"
            ),
            "Majority prediction share": fair.get(
                "prediction_majority_fraction"
            ),
            "Predicted positive share": fair.get(
                "predicted_positive_fraction"
            ),
            "Predicted classes": fair.get("predicted_class_count"),
            "Probability behavior": fair.get(
                "probability_behavior_status", "unavailable"
            ),
            "Positive-probability SD": fair.get(
                "positive_probability_std"
            ),
            "Interpretation warning": fair.get(
                "fairness_interpretation_warning"
            ),
            "Calibration status": fair.get("calibration_status")
            or "unavailable",
            "Probability coverage": fair.get("probability_coverage"),
            "Group Brier-score gap ↓": fair.get(
                "group_brier_score_gap"
            ),
            "Group ECE gap ↓": fair.get("group_ece_gap"),
            "Calibration bins": fair.get("calibration_bins"),
            "Reason when unavailable": fair.get("calibration_reason"),
        })
    return pd.DataFrame(rows)


def render_phase14_fairness_details(results):
    diagnostic = _fairness_rows(results)

    st.markdown("### Prediction-behaviour validity check")
    st.caption(
        "Zero disparity is not automatically evidence of a useful fair model. "
        "If a classifier predicts almost the same class or probability for "
        "everyone, some parity gaps can become mechanically zero."
    )

    behavior_cols = [
        "Framework",
        "Prediction status",
        "Majority prediction share",
        "Predicted positive share",
        "Predicted classes",
        "Probability behavior",
        "Positive-probability SD",
    ]
    st.dataframe(
        diagnostic[behavior_cols],
        use_container_width=True,
        hide_index=True,
    )

    flagged = diagnostic[
        diagnostic["Prediction status"].isin(
            ["constant", "near_constant"]
        )
        | diagnostic["Probability behavior"].isin(
            ["constant", "near_constant"]
        )
    ]
    if not flagged.empty:
        names = ", ".join(
            flagged["Framework"].astype(str).tolist()
        )
        st.warning(
            "Degenerate/near-degenerate prediction behaviour detected for: {}. "
            "Their raw fairness gaps remain visible for auditability, but they "
            "must not be used to claim a fairness winner without qualification."
            .format(names)
        )

    st.markdown("### Calibration fairness · Phase 14")
    st.caption(
        "Probability-based fairness is reported only when valid class "
        "probabilities exist. Missing probability evidence remains N/A and is "
        "never substituted with zero."
    )

    calibration_cols = [
        "Framework",
        "Calibration status",
        "Probability coverage",
        "Group Brier-score gap ↓",
        "Group ECE gap ↓",
        "Calibration bins",
        "Reason when unavailable",
    ]
    st.dataframe(
        diagnostic[calibration_cols],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(
        "Per-group calibration and hard-label details",
        expanded=False,
    ):
        for result in results:
            fair = result.get("fairness") or {}
            st.markdown("**{}**".format(result.get("framework")))

            calibration = fair.get("group_calibration") or {}
            if calibration:
                frame = pd.DataFrame([
                    {"Group": group, **values}
                    for group, values in calibration.items()
                ])
                st.caption("Calibration")
                st.dataframe(
                    frame,
                    use_container_width=True,
                    hide_index=True,
                )

            performance = fair.get("group_performance") or {}
            if performance:
                frame = pd.DataFrame([
                    {"Group": group, **values}
                    for group, values in performance.items()
                ])
                st.caption("Hard-label group behaviour")
                st.dataframe(
                    frame,
                    use_container_width=True,
                    hide_index=True,
                )

            if not calibration and not performance:
                st.caption(
                    fair.get("calibration_reason")
                    or "No group-level evidence is available."
                )

    st.download_button(
        "Download Phase-14 fairness diagnostics",
        data=diagnostic.to_csv(index=False).encode("utf-8"),
        file_name="phase14_fairness_diagnostics.csv",
        mime="text/csv",
        key="phase14_fairness_download",
    )


def _load_matching_repeatability_artifact():
    context = _active_context()

    if not context.get("dataset_content_sha256"):
        return None, None, (
            "AwareML cannot fingerprint the active dataframe yet. "
            "Reload the dataset in Run Studio, then reopen Sustainability Lab."
        )

    if not context.get("target"):
        return None, None, (
            "Select the target in Run Studio before looking up repeatability evidence."
        )

    match = find_latest_matching_run(
        ARTIFACT_ROOT,
        dataset_content_sha256=context["dataset_content_sha256"],
        target=context["target"],
        sensitive_attribute=context.get("sensitive_attribute"),
        positive_label=context.get("positive_label"),
    )

    if not match:
        return None, context, None

    try:
        rows = json.loads(
            match["results_path"].read_text(encoding="utf-8")
        )
        manifest = json.loads(
            match["manifest_path"].read_text(encoding="utf-8")
        )
        if not isinstance(rows, list):
            raise ValueError("results JSON is not a list")
        match["rows"] = rows
        match["manifest_payload"] = manifest
        return match, context, None
    except Exception as exc:
        return None, context, (
            "A matching repeatability run was found, but its files could not "
            "be read: {}".format(exc)
        )


def _beginner_command(context):
    name = str(context.get("dataset_name") or "dataset.csv")
    target = str(context.get("target") or "<target>")
    sensitive = context.get("sensitive_attribute")
    positive = context.get("positive_label")

    parts = [
        'python -m scripts.run_phase14_repeatability `',
        '  --csv "<path-to-{}>" `'.format(name),
        '  --target "{}" `'.format(target),
    ]
    if sensitive:
        parts.append(
            '  --sensitive "{}" `'.format(sensitive)
        )
    if positive is not None:
        parts.append(
            '  --positive-label "{}" `'.format(positive)
        )
    parts.append(
        "  --repetitions {}".format(PAPER_READY_MIN_REPETITIONS)
    )
    return "\n".join(parts)


def render_phase14_sustainability_details(results):
    st.markdown("### Measurement protocol · Phase 14")
    st.caption(
        "Hardware and measurement context are part of the evidence. Carbon "
        "intensity is reported in gCO₂/kWh and missing measurements remain N/A."
    )

    hardware = hardware_table(results)
    with st.expander(
        "Hardware and measurement context",
        expanded=False,
    ):
        st.dataframe(
            hardware,
            use_container_width=True,
            hide_index=True,
        )

    match, context, error = _load_matching_repeatability_artifact()

    st.markdown("### Dataset-specific repeatability · Phase 14")

    if error:
        st.warning(error)

    if context:
        st.caption(
            "Matching key = dataset content SHA256 + target + sensitive "
            "attribute + positive label."
        )
        context_table = pd.DataFrame([{
            "Dataset": context.get("dataset_name"),
            "Content SHA256": context.get("dataset_content_sha256"),
            "Target": context.get("target"),
            "Sensitive attribute": context.get("sensitive_attribute"),
            "Positive label": context.get("positive_label"),
        }])
        st.dataframe(
            context_table,
            use_container_width=True,
            hide_index=True,
        )

    if match:
        manifest = match["manifest_payload"]
        repeat_source = match["rows"]

        st.success(
            "Found the latest exact repeatability match for this dataset and "
            "configuration. Repetitions: {} · seeds: {} · paper-ready: {}."
            .format(
                manifest.get("repetitions"),
                manifest.get("seeds"),
                "YES" if manifest.get("paper_ready") else "NO",
            )
        )
        st.caption(
            "Saved run: {}".format(match["run_dir_path"])
        )

        repeatability = summarize_repeatability(
            repeat_source,
            min_repetitions=PAPER_READY_MIN_REPETITIONS,
        )
        st.dataframe(
            repeatability,
            use_container_width=True,
            hide_index=True,
        )

        ready = (
            not repeatability.empty
            and bool(
                (
                    repeatability["Repeatability gate"]
                    == "PASS"
                ).all()
            )
        )

        if ready:
            st.success(
                "Paper-ready repeatability gate PASS: at least {} independent "
                "repetitions are available per framework."
                .format(PAPER_READY_MIN_REPETITIONS)
            )
        else:
            current = int(manifest.get("repetitions") or 0)
            st.warning(
                "Repeatability evidence exists, but it is development evidence "
                "only: {} repetition(s) recorded; {} are required for the "
                "paper-ready gate. The saved run is retained and not overwritten."
                .format(
                    current,
                    PAPER_READY_MIN_REPETITIONS,
                )
            )

        repeated_hardware = hardware_table(repeat_source)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download matching repeatability table",
                data=repeatability.to_csv(index=False).encode("utf-8"),
                file_name="phase14_repeatability_table.csv",
                mime="text/csv",
                key="phase14_repeatability_download",
            )
        with c2:
            st.download_button(
                "Download matching hardware table",
                data=repeated_hardware.to_csv(index=False).encode("utf-8"),
                file_name="phase14_hardware_table.csv",
                mime="text/csv",
                key="phase14_hardware_download",
            )

    else:
        st.info(
            "No saved repeatability evidence matches this exact dataset and "
            "configuration yet. Nothing is wrong: run the isolated repeatability "
            "experiment once, then return here. Results from other datasets are "
            "kept separate and will never be mixed into this view."
        )
        if context:
            st.code(
                _beginner_command(context),
                language="powershell",
            )

    with st.expander(
        "Repeatability registry · all saved datasets",
        expanded=False,
    ):
        runs = list_dataset_runs(ARTIFACT_ROOT)
        if runs:
            registry_frame = pd.DataFrame(runs)
            visible = [
                "dataset_name",
                "target",
                "sensitive_attribute",
                "positive_label",
                "repetitions",
                "paper_ready",
                "created_utc",
                "run_dir",
            ]
            st.dataframe(
                registry_frame[
                    [c for c in visible if c in registry_frame.columns]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No registered Phase-14 repeatability runs yet.")
