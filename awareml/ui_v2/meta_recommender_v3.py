from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from awareml.protocol_v3.common import ROOT, OBJECTIVES


def render_meta_recommender_v3():
    st.title("Meta-Recommender V3")
    st.write("Recommendations from the revised stream benchmark use the first 1,000 labeled observations. The model predicts framework outcomes under its frozen sample-cap protocol.")
    bundle = st.text_input("Trained V3 bundle directory", str(ROOT / "artifacts/recommender_v3/primary_prefix"), key="v3_bundle_path")
    path = Path(bundle)
    if not (path / "manifest.json").exists():
        st.info("V3 training is pending. Collect the new benchmark runs, then train the prefix-only model. Installing the upgrade does not create new experimental results.")
        st.code("python scripts/awareml_v3.py prepare --cohort dev --campaign artifacts/stream_v3_dev\npython scripts/awareml_v3.py status --campaign artifacts/stream_v3_dev", language="bash")
        return
    from awareml.protocol_v3.service import V3Recommender
    try:
        service = V3Recommender(path)
    except Exception as exc:
        st.error(str(exc))
        return
    manifest = service.manifest
    st.caption("Bundle: {} · Portfolio: {} · Training datasets: {} · Features: prefix only".format(
        manifest["created_utc"], manifest["portfolio"], len(manifest["training_dataset_ids"])))
    from .state import ensure_research_state
    state = ensure_research_state()
    uploaded = st.file_uploader("Labeled classification CSV", type=["csv"], key="v3_csv")
    if uploaded:
        frame = pd.read_csv(uploaded, nrows=int(manifest["profile_size"]))
        default_target = frame.columns[-1]
    else:
        frame, default_target = state.get("dataset"), state.get("target")
    if frame is None:
        st.info("Upload a labeled CSV prefix or load a dataset in Run Studio.")
        return
    columns = frame.columns.tolist()
    target = st.selectbox("Target", columns, index=columns.index(default_target) if default_target in columns else len(columns) - 1, key="v3_target")
    weights = {}
    controls = st.columns(4)
    labels = {"accuracy": "Accuracy", "runtime": "Runtime", "energy": "Energy", "co2": "CO₂"}
    for column, objective in zip(controls, OBJECTIVES):
        weights[objective] = column.slider(labels[objective], 0, 100, 25, key="v3_w_" + objective)
    if sum(weights.values()) <= 0:
        st.warning("Choose at least one positive objective weight.")
        return
    try:
        ranked, meta = service.recommend_dataframe(frame, target, weights)
    except Exception as exc:
        st.error(str(exc))
        return
    st.metric("Recommended framework", ranked.iloc[0].framework_label)
    st.caption("Utility is a relative preference score. It is not a probability of being correct.")
    st.dataframe(ranked[["rank", "framework_label", "utility", "accuracy", "runtime", "energy", "co2"]], use_container_width=True, hide_index=True)
    st.bar_chart(ranked.set_index("framework_label").utility)
    with st.expander("Calibration prefix and measurement protocol"):
        st.json(meta)
        st.write("Prediction intervals use nested residual calibration and require external coverage assessment. The calibration prefix has already been observed; no zero-observation deployment claim is made.")
    with st.expander("Nested validation and baseline evidence"):
        validation = path.parent / (path.name + "_development_evaluation")
        if (validation / "baseline_summary.csv").exists():
            st.dataframe(pd.read_csv(validation / "baseline_summary.csv"), use_container_width=True, hide_index=True)
        else:
            st.caption("Run the evaluation command to export baseline comparisons and dataset-level statistics.")
