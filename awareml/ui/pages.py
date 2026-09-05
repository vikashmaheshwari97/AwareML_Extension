from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from awareml.data import make_drift_stream, profile_dataset, load_csv
from awareml.benchmarks import BUILTIN_STREAMS, load_builtin_stream
from awareml.engine.runner import run_benchmark
from awareml.engine.pareto import fairness_score_from_result
from awareml.llm import parse_objective_text, GroundedChat, ollama_status
from awareml.recommender import (
    RecommendationService, HistoricalMLRecommender, load_meta_logs,
    meta_log_coverage, profile_from_dataframe, locate_meta_logs,
)
from awareml.recommender.evaluation import grouped_benchmark
from awareml.studies import StudyStore, TrustCalibrationStudy, classify_follow_up
from awareml.studies.information_seeking import THINK_ALOUD_PROMPTS
from awareml.types import ObjectiveWeights, RunConfig
from awareml.analysis.repeatability_registry import canonical_dataframe_sha256
from .theme import hero


FAIRNESS_OPTIONS = {
    "Composite (all available gaps)": "composite",
    "Demographic parity": "demographic_parity",
    "Equal opportunity": "equal_opportunity",
    "Equalized odds": "equalized_odds",
    "Predictive parity": "predictive_parity",
    "Error-rate parity": "error_rate",
}


def _state():
    return st.session_state.awareml_state


def _results():
    return _state().get("run_results") or []


@st.cache_resource(show_spinner=False)
def _cached_historical_recommender(meta_path: str, modified_ns: int):
    # modified_ns is intentionally part of the cache key so replacing the log file retrains the models.
    df = load_meta_logs(meta_path)
    model = HistoricalMLRecommender(meta_df=df).train()
    return model, meta_log_coverage(df)


def _result_dicts():
    return [r.to_dict() if hasattr(r, "to_dict") else r for r in _results()]


def _plot(fig, key: str, height: int | None = None):
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config={"displaylogo": False, "responsive": True},
    )


def _results_df():
    rows = []
    for r in _result_dicts():
        fair = r.get("fairness", {}) or {}
        exp = r.get("explainability", {}) or {}
        pred_diag = r.get("prediction_diagnostics", {}) or {}
        rows.append({
            "Framework": r["framework"],
            "Backend": r.get("backend"),
            "Accuracy": r.get("accuracy"),
            "F1 macro": r.get("f1_macro"),
            "Runtime (s)": r.get("runtime_sec"),
            "Mean latency (ms)": r.get("mean_prediction_latency_ms"),
            "p95 latency (ms)": r.get("p95_prediction_latency_ms"),
            "Throughput (samples/s)": r.get("throughput_samples_sec"),
            "Drift recovery rate": (r.get("drift_summary") or {}).get("recovery_rate"),
            "Max drift accuracy drop": (r.get("drift_summary") or {}).get("max_accuracy_drop"),
            "Energy (kWh)": r.get("energy_kwh"),
            "CO2 (kg)": r.get("co2_kg"),
            "Drifts": len(r.get("drift_events") or []),
            "DP gap": fair.get("dp_diff"),
            "EOdds gap": fair.get("equalized_odds_gap"),
            "Explanation consistency": exp.get("consistency"),
            "Prediction coverage": pred_diag.get("prediction_coverage"),
            "Majority prediction fraction": pred_diag.get("majority_prediction_fraction"),
            "Near-constant prediction": pred_diag.get("near_constant_prediction"),
        })
    return pd.DataFrame(rows)

def _dataset_context(compact: bool = True):
    df = _state().get("dataset")
    name = _state().get("dataset_name")
    target = _state().get("target")
    if df is None:
        st.info("No active dataset. Load a stream once in Run Studio; every research workspace will reuse it.")
        return None
    n_features = max(0, df.shape[1] - (1 if target in df.columns else 0))
    n_classes = int(df[target].dropna().nunique()) if target in df.columns else None
    if compact:
        st.markdown(
            f'<div class="context-strip"><b>Active dataset</b> · {name or "dataset"} &nbsp; '
            f'<span>{len(df):,} rows</span> · <span>{n_features} features</span>'
            + (f' · <span>{n_classes} classes</span>' if n_classes is not None else '')
            + (f' · <span>target: {target}</span>' if target else '')
            + '</div>',
            unsafe_allow_html=True,
        )
    return df


def _metric_text(value, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
        if not np.isfinite(value):
            return "N/A"
        return f"{value:.{digits}f}{suffix}"
    except Exception:
        return "N/A"


def _top_feature_dict(result: dict, k: int = 8) -> dict:
    exp = result.get("explainability", {}) or {}
    vals = exp.get("feature_importance", []) or []
    pairs = []
    for row in vals:
        if isinstance(row, dict) and row.get("feature") is not None:
            try:
                pairs.append((str(row["feature"]), float(row.get("importance", 0.0))))
            except Exception:
                pass
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return dict(pairs[:k])


def _jaccard_top_features(a: dict, b: dict) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else 0.0


def _ollama_controls(prefix: str, compact: bool = False):
    status = ollama_status()
    enabled = st.toggle("Use local Ollama", value=False, key=f"{prefix}_enabled")
    models = status.get("models") or []
    default_model = _state().get("ollama_model") or (models[0] if models else "llama3.1:8b")
    if models and default_model not in models:
        default_model = models[0]
    if enabled:
        if status.get("reachable") and models:
            model = st.selectbox("Ollama model", models, index=models.index(default_model), key=f"{prefix}_model")
            _state()["ollama_model"] = model
            st.caption(f"Local Ollama connected · {len(models)} model(s) available")
        else:
            model = st.text_input("Ollama model", value=default_model, key=f"{prefix}_model_text")
            st.warning("Ollama is not reachable at the configured local endpoint. A deterministic fallback will be used if a request fails.")
    else:
        model = default_model
        if not compact:
            st.caption("LLM is optional. Deterministic, evidence-grounded behavior remains available without Ollama.")
    return enabled, model, status


def overview_page():
    hero(
        "AwareML research extension",
        "Streaming AutoML, made inspectable",
        "Compare five streaming AutoML systems under one temporal protocol, then inspect performance, fairness, explanations, sustainability, recommendation evidence, and human trust behavior.",
    )
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Frameworks", "5", "one comparison surface"),
        (c2, "Evaluation", "Prequential", "test → observe → train"),
        (c3, "Responsible-AI views", "3", "fairness · XAI · sustainability"),
        (c4, "LLM data exposure", "Derived facts", "raw rows excluded by default"),
    ]
    for c, label, val, note in cards:
        with c:
            st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{val}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Research workspaces</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    blocks = [
        ("01 · Streaming evidence", "Run real/synthetic streams with shared budgets, temporal order, drift detection, uncertainty and explicit backend provenance."),
        ("02 · Responsible optimization", "Treat fairness, interpretability, runtime, energy and carbon as selectable objectives rather than decorative post-run metrics."),
        ("03 · Human-centered evidence", "Measure trust calibration and information-seeking instead of assuming a fluent explanation automatically improves decisions."),
    ]
    for c, (title, body) in zip(cols, blocks):
        with c:
            st.markdown(f'<div class="card"><div class="label">{title}</div><div class="note" style="font-size:14px;line-height:1.55;margin-top:10px">{body}</div></div>', unsafe_allow_html=True)

    if _results():
        st.markdown('<div class="section-title">Current backend provenance</div>', unsafe_allow_html=True)
        st.dataframe(_results_df()[["Framework", "Backend", "Accuracy", "Runtime (s)"]], use_container_width=True, hide_index=True)


def run_studio_page():
    hero("Stages A–C", "Run Studio", "Configure one temporal experiment. Every selected framework sees the same stream order, sample cap and time budget.")
    left, right = st.columns([1.0, 1.55])
    with left:
        st.subheader("1 · Data")
        source = st.segmented_control("Source", ["Synthetic drift", "Built-in real stream", "Upload CSV"], default="Synthetic drift", key="run_source")
        if source == "Upload CSV":
            up = st.file_uploader("CSV stream", type=["csv"], key="run_csv")
            if up is not None:
                _state()["dataset"] = load_csv(up)
                _state()["dataset_name"] = up.name
        elif source == "Built-in real stream":
            built_name = st.selectbox("Built-in stream", list(BUILTIN_STREAMS.keys()), key="run_builtin")
            built_n = st.slider("Load first N samples", 500, 10000, 5000, 500, key="run_builtin_n")
            if st.button("Load built-in stream", use_container_width=True, key="load_builtin"):
                try:
                    _state()["dataset"] = load_builtin_stream(built_name, built_n)
                    _state()["dataset_name"] = built_name
                    st.success(f"Loaded {built_name} in stream order.")
                except Exception as exc:
                    st.error(str(exc))
        else:
            n_demo = st.slider("Synthetic samples", 1000, 12000, 6000, 500, key="run_synth_n")
            if _state().get("dataset") is None or _state().get("dataset_name") != "Synthetic drift" or len(_state().get("dataset", [])) != n_demo:
                _state()["dataset"] = make_drift_stream(n_demo)
                _state()["dataset_name"] = "Synthetic drift"

        df = _state().get("dataset")
        if isinstance(df, pd.DataFrame):
            _state()["dataset_content_sha256"] = canonical_dataframe_sha256(df)
        if df is not None:
            target = st.selectbox("Target", list(df.columns), index=max(0, len(df.columns)-1), key="run_target")
            sens_options = ["None"] + [c for c in df.columns if c != target]
            sens_default = sens_options.index("group") if "group" in sens_options else 0
            sens = st.selectbox("Sensitive attribute — explicit confirmation", sens_options, index=sens_default, key="run_sensitive")
            _state()["target"] = target
            _state()["sensitive"] = None if sens == "None" else sens
            if _state()["sensitive"] is not None:
                sensitive_feature_policy = st.radio(
                    "Protected attribute usage",
                    ["Audit only (exclude from model)", "Include in model + audit"],
                    index=0,
                    horizontal=True,
                    key="run_sensitive_policy",
                    help=(
                        "Audit only keeps the sensitive attribute for fairness measurement but removes it from model inputs. "
                        "Include reproduces experiments where the model can directly use that attribute. The choice is stored in the protocol."
                    ),
                )
                _state()["sensitive_feature_policy"] = (
                    "audit_only" if sensitive_feature_policy.startswith("Audit only") else "include"
                )
            else:
                _state()["sensitive_feature_policy"] = "audit_only"
            classes = df[target].dropna().unique().tolist()

            # Research-safe positive-label default. The Dutch Census demo
            # profile declares occupation_binary=1 as the positive outcome.
            dataset_name = str(_state().get("dataset_name") or "")
            preferred_positive = None

            if (
                Path(dataset_name).name.lower()
                == "dutch_census_stream_awareml.csv"
                and target == "occupation_binary"
            ):
                preferred_positive = next(
                    (
                        value
                        for value in classes
                        if str(value) == "1"
                    ),
                    None,
                )

            if preferred_positive is None:
                preferred_positive = next(
                    (
                        value
                        for value in classes
                        if value == 1 or str(value) == "1"
                    ),
                    classes[-1] if classes else 1,
                )

            profile_key = "{}|{}".format(
                Path(dataset_name).name.lower(),
                target,
            )

            if (
                st.session_state.get("_awareml_positive_profile_key")
                != profile_key
            ):
                st.session_state["_awareml_positive_profile_key"] = profile_key
                st.session_state["run_positive"] = preferred_positive
            elif (
                classes
                and st.session_state.get("run_positive") not in classes
            ):
                st.session_state["run_positive"] = preferred_positive

            default_positive_index = (
                classes.index(preferred_positive)
                if classes and preferred_positive in classes
                else 0
            )

            positive_label = (
                st.selectbox(
                    "Positive label",
                    classes,
                    index=default_positive_index,
                    key="run_positive",
                )
                if classes
                else 1
            )

            _state()["positive_label"] = positive_label

            if (
                Path(dataset_name).name.lower()
                == "dutch_census_stream_awareml.csv"
                and target == "occupation_binary"
            ):
                st.caption(
                    "Dutch Census demo profile: positive label = 1."
                )
            p = profile_dataset(df, target, _state().get("dataset_name") or "dataset")
            st.caption(f"{p.n_samples:,} rows · {p.n_features} features · {p.n_classes} classes · missing {p.missing_fraction:.2%}")

        st.subheader("2 · Shared protocol")
        window = st.number_input("Window size", 100, 5000, 500, 100, key="run_window")
        max_samples = st.number_input("Maximum samples", 500, 100000, 5000, 500, key="run_max_samples")
        budget = st.number_input("Per-framework time budget (seconds)", 5, 600, 60, 5, key="run_budget")
        seed = st.number_input("Seed", 0, 999999, 42, key="run_seed")
        track = st.toggle(
            "Measure energy & CO₂ with CodeCarbon",
            value=True,
            key="run_codecarbon",
            help="When enabled, AwareML attempts CodeCarbon directly. Missing measurements remain N/A rather than zero.",
        )
        with st.expander("Research instrumentation", expanded=False):
            xai_method = st.selectbox(
                "Model-level XAI strategy",
                ["auto", "shap", "lime", "permutation"],
                index=0,
                key="run_xai_method",
                help="auto tries SHAP, then LIME, then repeated permutation. SHAP/LIME are accepted only when genuine class probabilities are available.",
            )
            xai_max_rows = st.number_input("Recent rows for post-hoc XAI", 30, 1000, 250, 10, key="run_xai_rows")
            drift_assessment = st.number_input(
                "Minimum post-drift assessment samples",
                5,
                1000,
                max(10, min(100, int(window) // 5)),
                5,
                key="run_drift_assessment",
                help="Recovery cannot be declared before this many post-drift samples have been observed.",
            )
        oaml_mode = st.selectbox(
            "OAML execution mode",
            ["online", "gama"],
            index=0,
            key="run_oaml_mode",
            help="online = River 0.8 streaming path. gama = GAMA search on a warm-up window with periodic refits and River fallback.",
        )
        selected = st.multiselect(
            "Frameworks",
            ["AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha"],
            default=["AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha"],
            key="run_frameworks",
        )

        # Human-readable experiment configuration inspired by LLM+AutoML systems,
        # but generated deterministically so it is reproducible and auditable.
        protocol_yaml = yaml.safe_dump({
            "dataset": _state().get("dataset_name"),
            "target": _state().get("target"),
            "sensitive_attribute": _state().get("sensitive"),
            "sensitive_feature_policy": _state().get("sensitive_feature_policy", "audit_only"),
            "positive_label": _state().get("positive_label", 1),
            "stream_protocol": {
                "evaluation": "prequential-test-then-train",
                "window_size": int(window),
                "max_samples": int(max_samples),
                "time_budget_sec_per_framework": float(budget),
                "seed": int(seed),
            },
            "frameworks": selected,
            "oaml_mode": oaml_mode,
            "measure_energy_co2": bool(track),
            "research_instrumentation": {
                "xai_method": xai_method,
                "xai_max_rows": int(xai_max_rows),
                "drift_min_assessment_samples": int(drift_assessment),
            },
        }, sort_keys=False, allow_unicode=True)
        st.download_button(
            "Export experiment YAML",
            protocol_yaml,
            file_name="awareml_experiment.yaml",
            mime="text/yaml",
            use_container_width=True,
            key="run_export_yaml",
        )
        run = st.button("Run benchmark", type="primary", use_container_width=True, key="run_benchmark")

    with right:
        st.subheader("Stream evidence preview")
        df = _state().get("dataset")
        if df is not None:
            preview_tab, profile_tab = st.tabs(["Stream preview", "Dataset diagnostics"])
            with preview_tab:
                st.dataframe(df.head(12), use_container_width=True, hide_index=True)
                if _state().get("target"):
                    fig = px.line(df.head(min(len(df), 1500)).reset_index(), x="index", y=_state()["target"], title="Target over stream order")
                    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
                    _plot(fig, "run_target_preview", 260)
            with profile_tab:
                target_name = _state().get("target")
                if target_name in df.columns:
                    counts = df[target_name].astype(str).value_counts().head(20).rename_axis("Class").reset_index(name="Count")
                    fig = px.bar(counts, x="Class", y="Count", title="Target distribution")
                    _plot(fig, "run_target_distribution", 280)
                numeric = df.select_dtypes(include=[np.number]).copy()
                if target_name in numeric.columns and numeric.shape[1] > 1:
                    corr = numeric.iloc[: min(len(numeric), 5000)].corr(method="spearman")
                    if corr.shape[0] > 18:
                        # Keep the most target-correlated numeric features so the heatmap remains readable.
                        rel = corr[target_name].abs().sort_values(ascending=False).head(18).index
                        corr = corr.loc[rel, rel]
                    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto", title="Spearman correlation heatmap · diagnostic only")
                    _plot(fig, "run_correlation_heatmap", 420)
                missing = df.isna().mean().sort_values(ascending=False).head(15)
                if float(missing.max()) > 0:
                    miss_df = missing.rename("Missing fraction").reset_index().rename(columns={"index": "Feature"})
                    fig = px.bar(miss_df, x="Feature", y="Missing fraction", title="Missingness profile")
                    fig.update_xaxes(tickangle=-35)
                    _plot(fig, "run_missingness", 290)
        else:
            st.info("Choose a data source to continue.")

    if run:
        if df is None or not selected:
            st.error("Load data and select at least one framework.")
            return
        os.environ["AWAREML_OAML_MODE"] = oaml_mode
        cfg = RunConfig(
            target=_state()["target"],
            sensitive_attribute=_state()["sensitive"],
            window_size=int(window),
            max_samples=int(max_samples),
            seed=int(seed),
            time_budget_sec=float(budget),
            track_sustainability=bool(track),
            positive_label=_state().get("positive_label", 1),
            sensitive_feature_policy=_state().get("sensitive_feature_policy", "audit_only"),
            xai_method=xai_method,
            xai_max_rows=int(xai_max_rows),
            drift_min_assessment_samples=int(drift_assessment),
        )
        progress = st.progress(0.0, text="Preparing benchmark…")
        status = st.empty()
        fw_index = {name: i for i, name in enumerate(selected)}

        def cb(name, done, total):
            idx = fw_index.get(name, 0)
            overall = (idx + min(1.0, done / max(1, total))) / len(selected)
            progress.progress(min(1.0, overall), text=f"{name}: {done:,}/{total:,} samples")
            status.caption(f"Running {name} · shared prequential protocol")

        results = run_benchmark(df, cfg, frameworks=selected, progress=cb)
        progress.progress(1.0, text="Benchmark complete")
        _state()["run_results"] = results
        _state()["last_track_sustainability"] = bool(track)
        st.success("Run complete. Decision, Fairness, Explainability and Sustainability workspaces now use this evidence.")

    if _results():
        st.markdown('<div class="section-title">Latest comparison</div>', unsafe_allow_html=True)
        rdf = _results_df()
        st.dataframe(rdf, use_container_width=True, hide_index=True)
        c_left, c_right = st.columns([1, 1])
        with c_left:
            fig = px.bar(rdf, x="Framework", y="Accuracy", color="Framework", text_auto=".3f", title="Current-run prequential accuracy")
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=45, b=10))
            _plot(fig, "run_accuracy_comparison", 330)
        with c_right:
            fig = px.scatter(
                rdf, x="Runtime (s)", y="Accuracy", color="Framework", size="F1 macro",
                hover_data=["F1 macro", "p95 latency (ms)", "Throughput (samples/s)", "Energy (kWh)", "CO2 (kg)", "Drifts", "Backend"],
                title="Accuracy / runtime trade-off",
            )
            _plot(fig, "run_accuracy_runtime", 330)

        norm_cols = {"Accuracy": True, "F1 macro": True, "Runtime (s)": False, "Energy (kWh)": False, "CO2 (kg)": False}
        score = pd.DataFrame(index=rdf["Framework"])
        for col, benefit in norm_cols.items():
            vals = pd.to_numeric(rdf[col], errors="coerce")
            if vals.notna().sum() == 0:
                continue
            lo, hi = vals.min(), vals.max()
            if abs(float(hi - lo)) < 1e-12:
                n = pd.Series([0.5] * len(vals), index=vals.index)
            else:
                n = (vals - lo) / (hi - lo)
                if not benefit:
                    n = 1.0 - n
            score[col] = list(n.fillna(0.5))
        if not score.empty:
            fig = px.imshow(score, text_auto=".2f", zmin=0, zmax=1, aspect="auto", color_continuous_scale="Viridis", title="Current-run normalized objective profile · higher is better")
            _plot(fig, "run_normalized_objectives", 330)

        point_rows = []
        for r in _result_dicts():
            for p in r.get("points", []) or []:
                point_rows.append({
                    "Framework": r["framework"],
                    "Sample": p.get("sample"),
                    "Accuracy": p.get("accuracy"),
                    "Rolling accuracy": p.get("rolling_accuracy"),
                    "Macro-F1": p.get("f1_macro"),
                    "Rolling Macro-F1": p.get("rolling_f1_macro"),
                })
        if point_rows:
            tfig = px.line(pd.DataFrame(point_rows), x="Sample", y="Accuracy", color="Framework", markers=True, title="Temporal performance")
            tfig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
            _plot(tfig, "run_temporal_accuracy", 340)



def drift_temporal_page():
    hero(
        "Stage C",
        "Drift & Temporal Lab",
        "Inspect prequential performance as the stream evolves. Drift markers, recovery patterns and window-level comparisons are shown separately from final averages.",
    )
    _dataset_context(compact=True)
    if not _results():
        st.info("Run a benchmark first.")
        return

    results = _result_dicts()
    summary = []
    timeline_rows = []
    for r in results:
        pts = r.get("points", []) or []
        drift_events = r.get("drift_events", []) or []
        summary.append({
            "Framework": r["framework"],
            "Final accuracy": r.get("accuracy"),
            "F1 macro": r.get("f1_macro"),
            "Drifts": len(drift_events),
            "Samples": r.get("samples"),
            "Runtime (s)": r.get("runtime_sec"),
            "p95 latency (ms)": r.get("p95_prediction_latency_ms"),
            "Throughput (samples/s)": r.get("throughput_samples_sec"),
            "Recovery rate": (r.get("drift_summary") or {}).get("recovery_rate"),
            "Mean recovery (samples)": (r.get("drift_summary") or {}).get("mean_recovery_samples"),
            "Max accuracy drop": (r.get("drift_summary") or {}).get("max_accuracy_drop"),
        })
        for pt in pts:
            timeline_rows.append({
                "Framework": r["framework"],
                "Sample": pt.get("sample"),
                "Accuracy": pt.get("accuracy"),
                "Rolling accuracy": pt.get("rolling_accuracy"),
                "F1 macro": pt.get("f1_macro"),
                "Rolling F1 macro": pt.get("rolling_f1_macro"),
                "Mean latency (ms)": pt.get("mean_prediction_latency_ms"),
                "Throughput (samples/s)": pt.get("throughput_samples_sec"),
            })

    sdf = pd.DataFrame(summary)
    a, b, c, d, e = st.columns(5)
    best = sdf.sort_values("Final accuracy", ascending=False).iloc[0]
    a.metric("Best final accuracy", f"{best['Framework']} · {best['Final accuracy']:.3f}")
    b.metric("Total detected drifts", int(sdf["Drifts"].sum()))
    recovered = pd.to_numeric(sdf["Recovery rate"], errors="coerce").dropna()
    c.metric("Mean drift recovery", f"{recovered.mean():.2f}" if not recovered.empty else "N/A")
    p95 = pd.to_numeric(sdf["p95 latency (ms)"], errors="coerce").dropna()
    d.metric("Median p95 latency", f"{p95.median():.2f} ms" if not p95.empty else "N/A")
    e.metric("Shared protocol", "Test→train")

    timeline = pd.DataFrame(timeline_rows)
    if not timeline.empty:
        acc_col = "Rolling accuracy" if timeline["Rolling accuracy"].notna().any() else "Accuracy"
        fig = px.line(
            timeline, x="Sample", y=acc_col, color="Framework", markers=True,
            title=f"{acc_col} over stream windows",
        )
        for r in results:
            for pos in (r.get("drift_events") or []):
                fig.add_vline(x=pos, line_width=1, line_dash="dot", opacity=0.18)
        fig.update_layout(hovermode="x unified")
        _plot(fig, "temporal_accuracy_full", 420)

        pivot = timeline.pivot_table(index="Framework", columns="Sample", values=acc_col, aggfunc="last")
        if not pivot.empty:
            fig = px.imshow(
                pivot, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis",
                title="Window-level accuracy heatmap",
            )
            _plot(fig, "temporal_accuracy_heatmap", 360)

    left, right = st.columns([1.3, 1])
    with left:
        st.dataframe(sdf, use_container_width=True, hide_index=True)
    with right:
        drift_df = sdf[["Framework", "Drifts"]].copy()
        fig = px.bar(drift_df, x="Framework", y="Drifts", color="Framework", title="Detected drift events")
        fig.update_layout(showlegend=False)
        _plot(fig, "temporal_drift_counts", 320)
    if not timeline.empty and timeline["Rolling F1 macro"].notna().any():
        fig = px.line(
            timeline, x="Sample", y="Rolling F1 macro", color="Framework", markers=True,
            title="Rolling prequential Macro-F1",
        )
        fig.update_layout(hovermode="x unified")
        _plot(fig, "temporal_f1_full", 360)

    st.caption(
        "Phase 2 recovery is defined as the number of samples required for rolling accuracy to return "
        "within the configured tolerance of its pre-drift baseline. Unrecovered events remain explicit; "
        "journal claims should aggregate repeated seeds and report uncertainty."
    )


def decision_lab_page():
    hero("Stage D", "Decision Lab", "Convert user preferences into an auditable multi-objective ranking. Fairness is selectable as an optimization criterion rather than only a post-hoc report.")
    if not _results():
        st.info("Run a benchmark first.")
        return

    cgoal, cllm = st.columns([1.65, 1])
    with cllm:
        use_llm, llm_model, _ = _ollama_controls("decision_llm", compact=True)
    with cgoal:
        goal = st.text_input("Describe your goal", "Prioritize accuracy, but keep runtime and carbon low and consider fairness.", key="decision_goal")
    parsed, meta = parse_objective_text(goal, use_llm=use_llm, model=llm_model)
    defaults = parsed.normalized_weights()
    st.caption(f"Objective parser: {meta['source']}" + (f" · model: {meta.get('model')}" if meta.get("model") else "") + (f" · {meta.get('warnings')}" if meta.get("warnings") else ""))

    cols = st.columns(6)
    values = {}
    for c, key in zip(cols, ["accuracy", "runtime", "energy", "co2", "fairness", "interpretability"]):
        with c:
            values[key] = st.number_input(key.title(), 0.0, 1.0, float(defaults[key]), 0.05, key=f"decision_w_{key}")
    weights = ObjectiveWeights(**values)

    c1, c2 = st.columns(2)
    with c1:
        fairness_label = st.selectbox("Fairness criterion used in utility", list(FAIRNESS_OPTIONS.keys()), key="decision_fairness_metric")
        fairness_metric = FAIRNESS_OPTIONS[fairness_label]
    with c2:
        epsilon = st.slider("Near-Pareto epsilon", 0.0, 0.20, 0.05, 0.01, key="decision_epsilon")

    service = RecommendationService(epsilon=epsilon, fairness_metric=fairness_metric)
    frame, recs, corr, warnings = service.rank(_results(), weights)
    _state()["ranking"] = frame.to_dict(orient="records")
    _state()["fairness_metric"] = fairness_metric

    top = recs[0]
    a, b, c = st.columns([1.2, 1, 1])
    a.metric("Recommended framework", top.framework)
    b.metric("Weighted utility", f"{top.utility:.3f}")
    c.metric("Near-Pareto", "Yes" if top.near_pareto else "No")
    for warning in warnings:
        st.warning(warning)

    left, right = st.columns([1.45, 1])
    with left:
        show = frame.rename(columns={"framework": "Framework", "utility": "Utility", "near_pareto": "Near-Pareto", "rank": "Rank", "fairness_score": "Fairness score"})
        st.dataframe(show, use_container_width=True, hide_index=True)
        fig = px.scatter(frame, x="accuracy", y="fairness_score", size="utility", color="framework", hover_data=["runtime_sec", "near_pareto"], title=f"Accuracy / fairness trade-off · {fairness_label}")
        fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
        _plot(fig, "decision_accuracy_fairness", 390)
    with right:
        st.subheader("Objective correlation")
        if not corr.empty:
            fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(margin=dict(l=10, r=10, t=25, b=10))
            _plot(fig, "decision_correlation", 360)
        st.caption("Correlation is descriptive across the current candidates; it is not causal.")

    st.subheader("Uncertainty")
    unc_rows = []
    for r in _result_dicts():
        ci = (r.get("uncertainty") or {}).get("window_accuracy", {})
        unc_rows.append([r["framework"], ci.get("mean"), ci.get("lower"), ci.get("upper"), ci.get("n")])
    st.dataframe(pd.DataFrame(unc_rows, columns=["Framework", "Window mean", "95% CI lower", "95% CI upper", "Windows"]), use_container_width=True, hide_index=True)


def fairness_page():
    hero(
        "Stage E1",
        "Fairness Lab",
        "Streaming group fairness as temporal evidence: compare complementary disparity criteria, inspect worst windows, and keep unsupported metrics explicitly missing.",
    )
    _dataset_context(compact=True)
    if not _results():
        st.info("Run a benchmark first.")
        return
    if not _state().get("sensitive"):
        st.warning("No sensitive attribute was confirmed for the current run. Re-run from Run Studio with an explicit sensitive attribute to compute fairness evidence.")
        return

    results = _result_dicts()
    metric_map = {
        "Demographic parity": "dp_diff",
        "Equal opportunity": "equal_opportunity_diff",
        "Equalized odds": "equalized_odds_gap",
        "Predictive parity": "predictive_parity_diff",
        "Error-rate parity": "error_rate_gap",
    }
    rows = []
    for r in results:
        f = r.get("fairness", {}) or {}
        row = {"Framework": r["framework"]}
        for label, key in metric_map.items():
            row[label] = f.get(key)
        row["Status"] = f.get("status")
        row["Window N"] = f.get("window_n")
        rows.append(row)
    fair_df = pd.DataFrame(rows)
    metric_cols = list(metric_map.keys())

    available = fair_df[metric_cols].apply(pd.to_numeric, errors="coerce")
    mean_gap = available.mean(axis=1, skipna=True)
    best_idx = mean_gap.idxmin() if mean_gap.notna().any() else None
    worst_idx = available.max(axis=1, skipna=True).idxmax() if available.notna().any().any() else None
    coverage_n = int(available.notna().sum().sum())
    coverage_total = max(1, int(available.shape[0] * available.shape[1]))

    a, b, c, d = st.columns(4)
    if best_idx is not None:
        a.metric("Lowest mean disparity", fair_df.loc[best_idx, "Framework"], _metric_text(mean_gap.loc[best_idx], 3))
    else:
        a.metric("Lowest mean disparity", "N/A")
    b.metric("Metric coverage", f"{coverage_n}/{coverage_total}")
    c.metric("Sensitive attribute", str(_state().get("sensitive")))
    d.metric("Fairness window", str(next((r.get("fairness", {}).get("window_n") for r in results if r.get("fairness")), "N/A")))

    long = fair_df.melt(id_vars=["Framework", "Status", "Window N"], value_vars=metric_cols, var_name="Fairness metric", value_name="Gap").dropna()
    left, right = st.columns([1.35, 1])
    with left:
        if not long.empty:
            fig = px.bar(
                long, x="Framework", y="Gap", color="Fairness metric", barmode="group",
                title="Complementary fairness gaps · lower is better",
            )
            fig.update_layout(legend_title_text="Criterion")
            _plot(fig, "fairness_grouped_gaps", 390)
    with right:
        heat = fair_df.set_index("Framework")[metric_cols]
        fig = px.imshow(
            heat, text_auto=".3f", zmin=0, zmax=max(0.25, float(np.nanmax(heat.to_numpy(dtype=float))) if np.isfinite(heat.to_numpy(dtype=float)).any() else 1.0),
            aspect="auto", color_continuous_scale="YlOrRd", title="Disparity heatmap",
        )
        _plot(fig, "fairness_gap_heatmap", 390)

    # Parity radar: convert each available gap to a [0,1] parity score for visualization only.
    radar = fair_df.copy()
    for col in metric_cols:
        radar[col] = 1.0 - pd.to_numeric(radar[col], errors="coerce").clip(0.0, 1.0)
    fig = go.Figure()
    theta = metric_cols + [metric_cols[0]]
    for _, row in radar.iterrows():
        vals = [row.get(c) if pd.notna(row.get(c)) else 0.0 for c in metric_cols]
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=theta, fill="toself", name=row["Framework"], opacity=0.45))
    fig.update_layout(
        title="Parity profile · 1 − gap (visual aid only)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(l=40, r=40, t=70, b=40),
    )
    _plot(fig, "fairness_parity_radar", 430)

    st.markdown('<div class="section-title">Temporal fairness</div>', unsafe_allow_html=True)
    temporal_rows = []
    point_keys = {
        "dp_diff": "Demographic parity",
        "eo_diff": "Equal opportunity",
        "equalized_odds_gap": "Equalized odds",
        "predictive_parity_diff": "Predictive parity",
        "error_rate_gap": "Error-rate parity",
    }
    for r in results:
        for pt in (r.get("points") or []):
            for key, label in point_keys.items():
                val = pt.get(key)
                if val is not None:
                    temporal_rows.append({"Framework": r["framework"], "Sample": pt.get("sample"), "Metric": label, "Gap": val})
    temporal = pd.DataFrame(temporal_rows)
    if temporal.empty:
        st.info("Temporal fairness evidence is unavailable for this run.")
    else:
        selected_metric = st.selectbox("Temporal fairness criterion", sorted(temporal["Metric"].unique()), key="fair_temporal_metric")
        tmp = temporal[temporal["Metric"] == selected_metric]
        fig = px.line(tmp, x="Sample", y="Gap", color="Framework", markers=True, title=f"{selected_metric} over stream windows")
        fig.update_layout(hovermode="x unified")
        _plot(fig, "fairness_temporal_selected", 390)
        worst = tmp.groupby("Framework")["Gap"].max().sort_values(ascending=False).rename("Worst-window gap").reset_index()
        st.dataframe(worst, use_container_width=True, hide_index=True)

        summary_key = {v: k for k, v in point_keys.items()}.get(selected_metric)
        summary_rows = []
        if summary_key:
            for r in results:
                item = (((r.get("fairness") or {}).get("temporal_summary") or {}).get("metrics") or {}).get(summary_key, {})
                if item:
                    summary_rows.append({
                        "Framework": r["framework"],
                        "Mean gap": item.get("mean"),
                        "Time-weighted mean": item.get("time_weighted_mean"),
                        "P95 gap": item.get("p95"),
                        "Max gap": item.get("max"),
                        "Volatility": item.get("volatility"),
                        "Windows": item.get("n"),
                    })
        if summary_rows:
            st.caption("Phase-3 temporal aggregation (missing windows remain missing; they are never converted to zero).")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Framework detail</div>', unsafe_allow_html=True)
    tabs = st.tabs([r["framework"] for r in results])
    for idx, (tab, r) in enumerate(zip(tabs, results)):
        with tab:
            f = r.get("fairness", {}) or {}
            cols = st.columns(5)
            vals = [
                ("DP gap", f.get("dp_diff")), ("EO gap", f.get("equal_opportunity_diff")),
                ("EOdds gap", f.get("equalized_odds_gap")), ("PPV gap", f.get("predictive_parity_diff")),
                ("Error gap", f.get("error_rate_gap")),
            ]
            for c, (lab, val) in zip(cols, vals):
                c.metric(lab, _metric_text(val, 3))
            support = f.get("groups", {}) or {}
            st.caption(f"Status: {f.get('status')} · recent window: {f.get('window_n', 'N/A')} · group support: {support}")
            pred_diag = r.get("prediction_diagnostics", {}) or {}
            if pred_diag.get("near_constant_prediction"):
                st.warning(
                    "Prediction-degeneracy safeguard: this framework is producing near-constant predictions "
                    f"(majority fraction={_metric_text(pred_diag.get('majority_prediction_fraction'), 3)}). "
                    "Very small fairness gaps can be misleading when predictive utility is weak."
                )
            if support:
                support_df = pd.DataFrame([{"Group": str(g), "Window support": n} for g, n in support.items()])
                fig = px.bar(support_df, x="Group", y="Window support", title="Sensitive-group support in current fairness window")
                _plot(fig, f"fairness_support_{idx}", 280)

    st.info("Fairness gaps are descriptive criteria, not interchangeable definitions of fairness. A publication should report the criterion, sensitive attribute, positive label, group support, temporal aggregation, and worst-window behavior.")


def explainability_page():
    hero(
        "Stage E2",
        "Explainability Lab",
        "Recover the strongest ideas from the previous multi-level explainability dashboard: model-level evidence, hyperparameter context and system-level comparison — now separated from fairness and backed by explicit diagnostics.",
    )
    _dataset_context(compact=True)
    if not _results():
        st.info("Run a benchmark first.")
        return

    results = _result_dicts()
    use_llm, llm_model, status = _ollama_controls("xai_llm", compact=True)
    if status.get("reachable"):
        st.caption("Local Ollama is available for grounded explanation summaries. Raw dataset rows are not sent by default.")

    model_tab, hp_tab, system_tab = st.tabs([
        "Model-level explanations",
        "Hyperparameter-level context",
        "System-level explainability",
    ])

    with model_tab:
        framework = st.selectbox("Framework", [r["framework"] for r in results], key="xai_model_framework")
        r = next(x for x in results if x["framework"] == framework)
        e = r.get("explainability", {}) or {}
        if e.get("status") == "failed":
            st.warning(f"Explainability failed but the benchmark result was preserved: {e.get('error', 'unknown error')}")
        elif e.get("status") == "unsupported":
            st.warning("No requested XAI method produced a non-degenerate explanation signal. AwareML does not display all-zero importance as a valid explanation.")
            attempts = e.get("method_attempts", []) or []
            if attempts:
                st.dataframe(pd.DataFrame(attempts), use_container_width=True, hide_index=True)
        elif e.get("status") != "ok":
            st.info("Explanation diagnostics require at least 30 recent labeled observations.")
        else:
            a, b, c, d, ee = st.columns(5)
            a.metric("Stability", _metric_text(e.get("stability"), 3))
            b.metric("Consistency", _metric_text(e.get("consistency"), 3))
            c.metric("Sensitivity ↓", _metric_text(e.get("sensitivity"), 3))
            d.metric("Deletion fidelity", _metric_text(e.get("fidelity"), 3))
            ee.metric("Sparsity", _metric_text(e.get("sparsity"), 3))
            st.caption(f"Method actually used: {e.get('method', 'N/A')} · model backend: {r.get('backend')}")
            if e.get("replay_warning"):
                st.warning(e.get("replay_warning"))
            attempts = e.get("method_attempts", []) or []
            if attempts:
                with st.expander("XAI method audit trail", expanded=False):
                    st.dataframe(pd.DataFrame(attempts), use_container_width=True, hide_index=True)
                    if e.get("method_metadata"):
                        st.json(e.get("method_metadata"))

            imp = pd.DataFrame(e.get("feature_importance", []) or [])
            left, right = st.columns([1.5, 1])
            with left:
                if not imp.empty:
                    top = imp.head(20).sort_values("importance")
                    fig = px.bar(top, x="importance", y="feature", orientation="h", title="Feature importance on recent explanation window")
                    _plot(fig, "xai_model_feature_importance", 470)
            with right:
                if not imp.empty:
                    top_table = imp.head(12).copy()
                    top_table["rank"] = np.arange(1, len(top_table) + 1)
                    st.dataframe(top_table[["rank", "feature", "importance"]], use_container_width=True, hide_index=True)
                base = e.get("base_accuracy_on_explanation_window")
                reference = e.get("prequential_reference_accuracy")
                replay_gap = e.get("replay_accuracy_gap")
                deletion = e.get("deletion_accuracy")
                st.markdown(
                    f'<div class="evidence-card"><b>Diagnostic check</b><br>'
                    f'final-model replay accuracy: <b>{_metric_text(base, 3)}</b><br>'
                    f'prequential rolling reference: <b>{_metric_text(reference, 3)}</b><br>'
                    f'replay gap: <b>{_metric_text(replay_gap, 3)}</b><br>'
                    f'after deleting top features: <b>{_metric_text(deletion, 3)}</b><br>'
                    '<span>Deletion fidelity measures predictive degradation under perturbation; it does not establish causality.</span></div>',
                    unsafe_allow_html=True,
                )
            st.info(e.get("diagnostic_note", "Perturbation/resampling diagnostics are comparative, not causal guarantees."))

            if use_llm and st.button("Generate grounded model-level explanation", key="xai_model_llm"):
                facts = GroundedChat(model=llm_model).build_facts(results, _state().get("ranking"))
                answer, meta = GroundedChat(model=llm_model).answer(
                    f"Explain the model-level explainability evidence for {framework}. Mention top features, method, stability, consistency, sensitivity, deletion fidelity, sparsity and limitations.",
                    facts,
                    use_llm=True,
                )
                st.markdown(answer)
                st.caption(f"Answer source: {meta.get('source')}" + (f" · model: {meta.get('model')}" if meta.get("model") else ""))

    with hp_tab:
        st.markdown("### Hyperparameter-level context")
        st.caption(
            "The previous dashboard exposed hyperparameter values and tuning narratives. This extension preserves that layer, but does not label a single final configuration as temporal 'evolution' unless a framework actually records its tuning history."
        )
        scope = st.segmented_control("Scope", ["Single framework", "All frameworks"], default="Single framework", key="xai_hp_scope")
        selected = st.selectbox("Framework", [r["framework"] for r in results], key="xai_hp_fw") if scope == "Single framework" else None
        chosen = [r for r in results if selected is None or r["framework"] == selected]
        for idx, r in enumerate(chosen):
            params = r.get("parameters", {}) or {}
            st.markdown(f"#### {r['framework']}")
            if not params:
                st.info("No framework parameters were exposed for this run.")
                continue
            flat = []
            for k, v in params.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        flat.append({"Parameter": f"{k}.{sk}", "Value": sv})
                else:
                    flat.append({"Parameter": k, "Value": v})
            pdf = pd.DataFrame(flat)
            st.dataframe(pdf, use_container_width=True, hide_index=True)
            numeric = []
            for row in flat:
                try:
                    numeric.append({"Parameter": row["Parameter"], "Value": float(row["Value"])})
                except Exception:
                    pass
            if numeric:
                ndf = pd.DataFrame(numeric).head(20)
                fig = px.bar(ndf, x="Parameter", y="Value", title=f"Numeric hyperparameter snapshot · {r['framework']}")
                fig.update_xaxes(tickangle=-35)
                _plot(fig, f"xai_hp_values_{idx}_{r['framework']}", 350)
        st.warning("To make this a true hyperparameter-evolution study, the runner should log parameter changes per adaptation window together with the resulting performance/fairness/energy delta. Phase 3 keeps the UI scientifically honest until that trace exists.")

    with system_tab:
        st.markdown("### Cross-framework explanation agreement")
        diag_rows = []
        feature_maps = {}
        for r in results:
            e = r.get("explainability", {}) or {}
            diag_rows.append({
                "Framework": r["framework"],
                "Stability": e.get("stability"),
                "Consistency": e.get("consistency"),
                "Sensitivity": e.get("sensitivity"),
                "Fidelity": e.get("fidelity"),
                "Sparsity": e.get("sparsity"),
            })
            feature_maps[r["framework"]] = _top_feature_dict(r, k=8)
        diag = pd.DataFrame(diag_rows)
        long = diag.melt(id_vars="Framework", var_name="Diagnostic", value_name="Score").dropna()
        if not long.empty:
            fig = px.bar(long, x="Framework", y="Score", color="Diagnostic", barmode="group", title="Explanation-quality diagnostics across frameworks")
            _plot(fig, "xai_system_diagnostics", 390)

        names = [r["framework"] for r in results]
        overlap = pd.DataFrame(index=names, columns=names, dtype=float)
        for a in names:
            for b in names:
                overlap.loc[a, b] = _jaccard_top_features(feature_maps.get(a, {}), feature_maps.get(b, {}))
        if not overlap.empty:
            fig = px.imshow(overlap, text_auto=".2f", zmin=0, zmax=1, color_continuous_scale="Blues", title="Top-feature agreement · Jaccard overlap")
            _plot(fig, "xai_system_feature_overlap", 390)

        rows = []
        for fw, fmap in feature_maps.items():
            for feat, imp in fmap.items():
                rows.append({"Framework": fw, "Feature": feat, "Importance": imp})
        if rows:
            fi_long = pd.DataFrame(rows)
            pivot = fi_long.pivot_table(index="Feature", columns="Framework", values="Importance", fill_value=0.0)
            pivot["mean_importance"] = pivot.mean(axis=1)
            pivot = pivot.sort_values("mean_importance", ascending=False).head(15).drop(columns=["mean_importance"])
            fig = px.imshow(pivot, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis", title="Unified feature-importance view across frameworks")
            _plot(fig, "xai_system_feature_heatmap", 430)

        if use_llm and st.button("Generate grounded system-level XAI summary", key="xai_system_llm"):
            facts = GroundedChat(model=llm_model).build_facts(results, _state().get("ranking"))
            answer, meta = GroundedChat(model=llm_model).answer(
                "Compare explainability across all frameworks. Discuss where top features agree or disagree and summarize stability, consistency, sensitivity, fidelity and limitations without causal claims.",
                facts,
                use_llm=True,
            )
            st.markdown(answer)
            st.caption(f"Answer source: {meta.get('source')}" + (f" · model: {meta.get('model')}" if meta.get("model") else ""))


def sustainability_page():
    hero(
        "Reviewer protocol",
        "Sustainability Lab",
        "Treat energy and carbon as measured experimental variables. Compare efficiency, runtime and carbon intensity while preserving missingness and measurement provenance.",
    )
    _dataset_context(compact=True)
    st.markdown('<div class="research-warning"><b>Protocol rule:</b> missing CodeCarbon output stays <code>N/A</code>. A zero is never invented to make a chart look complete.</div>', unsafe_allow_html=True)
    st.write("")
    if not _results():
        st.info("No run evidence yet. Enable ‘Measure energy & CO₂ with CodeCarbon’ in Run Studio.")
        return

    rows = []
    for r in _result_dicts():
        s = r.get("sustainability", {}) or {}
        rows.append({
            "Framework": r["framework"], "Status": s.get("status"), "Energy kWh": r.get("energy_kwh"),
            "CO2 kg": r.get("co2_kg"), "Runtime s": r.get("runtime_sec"), "Samples": r.get("samples"),
            "Measurement duration s": s.get("duration_sec"), "Backend": s.get("measurement_backend"),
            "CodeCarbon": s.get("codecarbon_version"), "CPU": s.get("cpu"), "GPU": s.get("gpu"), "RAM GB": s.get("ram_gb"),
        })
    sdf = pd.DataFrame(rows)
    measured_energy = sdf["Energy kWh"].notna().sum()
    measured_co2 = sdf["CO2 kg"].notna().sum()
    a, b, c, d = st.columns(4)
    a.metric("Energy measured", f"{measured_energy}/{len(sdf)}")
    b.metric("CO₂ measured", f"{measured_co2}/{len(sdf)}")
    c.metric("Total measured energy", _metric_text(pd.to_numeric(sdf["Energy kWh"], errors="coerce").sum(min_count=1), 6, " kWh"))
    d.metric("Total measured CO₂", _metric_text(pd.to_numeric(sdf["CO2 kg"], errors="coerce").sum(min_count=1), 6, " kg"))

    st.dataframe(sdf, use_container_width=True, hide_index=True)
    measured = sdf.dropna(subset=["Energy kWh", "CO2 kg"], how="all")
    if measured.empty:
        st.warning("This run contains no measured energy/CO₂ values. Re-run from Run Studio with CodeCarbon measurement enabled and inspect the diagnostics below.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            e = measured.dropna(subset=["Energy kWh"])
            if not e.empty:
                fig = px.bar(e, x="Framework", y="Energy kWh", color="Framework", text_auto=".3g", title="Measured energy consumption")
                fig.update_layout(showlegend=False)
                _plot(fig, "sustain_energy", 340)
        with c2:
            c = measured.dropna(subset=["CO2 kg"])
            if not c.empty:
                fig = px.bar(c, x="Framework", y="CO2 kg", color="Framework", text_auto=".3g", title="Measured carbon emissions")
                fig.update_layout(showlegend=False)
                _plot(fig, "sustain_co2", 340)

        st.markdown('<div class="section-title">Efficiency relationships</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            both = measured.dropna(subset=["Runtime s", "Energy kWh"])
            if len(both) >= 2:
                fig = px.scatter(both, x="Runtime s", y="Energy kWh", color="Framework", size="Samples", hover_data=["CO2 kg"], title="Runtime vs measured energy")
                _plot(fig, "sustain_runtime_energy", 360)
        with right:
            both = measured.dropna(subset=["Energy kWh", "CO2 kg"])
            if len(both) >= 2:
                corr = both[["Energy kWh", "CO2 kg"]].corr(method="spearman").iloc[0, 1]
                fig = px.scatter(both, x="Energy kWh", y="CO2 kg", color="Framework", text="Framework", title=f"Energy vs CO₂ · Spearman ρ={corr:.2f}")
                fig.update_traces(textposition="top center")
                _plot(fig, "sustain_energy_co2", 360)
                st.caption("A high energy/CO₂ correlation means assigning both large utility weights can double-count a similar resource signal.")

    with st.expander("Measurement diagnostics", expanded=False):
        for r in _result_dicts():
            s = r.get("sustainability", {}) or {}
            st.markdown(f"**{r['framework']} — {s.get('status')}**")
            notes = s.get("notes") or ["No diagnostic notes recorded."]
            for note in notes:
                st.caption(f"• {note}")

    st.subheader("Paper-grade protocol checklist")
    st.markdown("""
- Use one warm-up plus at least five measured repetitions for reported energy/CO₂ claims.
- Keep machine, sample cap and time budget fixed across frameworks within a comparison.
- Record CPU/GPU, RAM, OS, Python, CodeCarbon version, region and run timestamp.
- Report mean, standard deviation, median and bootstrap 95% confidence interval across repetitions.
- Test energy/CO₂ correlation before assigning both large independent utility weights.
- Separate runtime efficiency from regional carbon-intensity effects.
""")


def recommender_lab_page():
    hero(
        "Stage B",
        "Recommender Lab",
        "Use the active dataset once. Compare an LLM-assisted natural-language recommender with an explicit ML meta-recommender trained from historical AwareML runs.",
    )
    df = _dataset_context(compact=True)
    if df is None:
        st.warning("Load a dataset in Run Studio first. This page intentionally has no second dataset uploader.")
        return
    target = _state().get("target")
    if not target or target not in df.columns:
        st.warning("Confirm the target column in Run Studio first.")
        return

    meta_path = locate_meta_logs()
    meta_df = load_meta_logs(str(meta_path) if meta_path else None)
    coverage = meta_log_coverage(meta_df)
    profile = profile_from_dataframe(df, target)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical meta-runs", f"{coverage.get('rows', 0):,}")
    c2.metric("Historical datasets", f"{coverage.get('datasets', 0):,}")
    c3.metric("Energy coverage", f"{coverage.get('energy_coverage', 0.0):.0%}")
    c4.metric("CO₂ coverage", f"{coverage.get('co2_coverage', 0.0):.0%}")

    if meta_df.empty:
        st.error("No historical meta_logs.json was found in data/meta/ or the project root.")
        return

    try:
        if meta_path is None:
            raise RuntimeError("meta_logs.json not found")
        stat = meta_path.stat()
        recommender, _ = _cached_historical_recommender(str(meta_path), int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))))
    except Exception as exc:
        st.error(f"Historical ML recommender could not be trained: {exc}")
        return

    tab_llm, tab_ml, tab_eval = st.tabs([
        "LLM-Assisted Recommender",
        "ML Recommender",
        "Meta-Evidence & Validation",
    ])

    with tab_llm:
        st.markdown("### Natural-language objective → validated evidence ranking")
        st.caption(
            "The local LLM parses the user's goal; the framework choice is then checked against the historical meta-model. "
            "If Ollama is unavailable or returns malformed output, the deterministic parser is used."
        )
        top, controls = st.columns([1.7, 1])
        with controls:
            use_llm, llm_model, status = _ollama_controls("rec_llm", compact=True)
            if status.get("reachable"):
                st.success("Ollama API is already running on 127.0.0.1:11434.")
        with top:
            goal = st.text_area(
                "AutoML goal",
                value="High predictive performance with fast adaptation to drift, while keeping runtime, energy and carbon reasonably low.",
                height=120,
                key="recommender_goal",
            )
        if st.button("Generate LLM-assisted recommendation", type="primary", use_container_width=True, key="rec_llm_generate"):
            parsed, parser_meta = parse_objective_text(goal, use_llm=use_llm, model=llm_model)
            w = parsed.normalized_weights()
            meta_weights = {
                "accuracy": w.get("accuracy", 0.0),
                "runtime": w.get("runtime", 0.0),
                "energy": w.get("energy", 0.0),
                "co2": w.get("co2", 0.0),
            }
            candidates, rec_meta = recommender.predict_candidates(**profile, weights=meta_weights)
            _state()["pre_run_recommendation"] = candidates.to_dict(orient="records")
            _state()["pre_run_recommendation_meta"] = rec_meta
            _state()["pre_run_objective"] = {"goal": goal, "parser": parser_meta, "weights": w}

        rows = _state().get("pre_run_recommendation")
        if rows:
            cand = pd.DataFrame(rows)
            obj = _state().get("pre_run_objective") or {}
            parser_meta = obj.get("parser") or {}
            st.caption(
                f"Objective parser: {parser_meta.get('source', 'unknown')}"
                + (f" · model: {parser_meta.get('model')}" if parser_meta.get("model") else "")
            )
            best = cand.iloc[0]
            a, b, c, d = st.columns([1.2, 1, 1, 1])
            a.metric("Recommended framework", best["framework"])
            b.metric("Expected accuracy", _metric_text(best.get("accuracy"), 2, "%"))
            c.metric("Expected runtime", _metric_text(best.get("runtime_sec"), 2, " s"))
            d.metric("Evidence utility", _metric_text(best.get("utility"), 3))
            st.markdown(
                f'<div class="evidence-card"><b>Suggested algorithm:</b> {best.get("algorithm", "N/A")} &nbsp; '
                f'· <b>closest historical dataset:</b> {best.get("nearest_dataset", "N/A")}<br>'
                '<span>The recommendation is a pre-run estimate from historical evidence; the Run Studio benchmark remains the verification step.</span></div>',
                unsafe_allow_html=True,
            )
            view = cand[["rank", "framework", "algorithm", "accuracy", "runtime_sec", "energy_kwh", "co2_ug", "utility", "nearest_dataset"]].copy()
            view.columns = ["Rank", "Framework", "Algorithm", "Expected accuracy", "Expected runtime (s)", "Expected energy (kWh)", "Expected CO₂ (µg)", "Utility", "Nearest dataset"]
            st.dataframe(view, use_container_width=True, hide_index=True)

            chart = cand[["framework", "accuracy_score", "runtime_score", "energy_score", "co2_score"]].melt(
                id_vars="framework", var_name="objective", value_name="normalized_score"
            )
            fig = px.bar(chart, x="framework", y="normalized_score", color="objective", barmode="group", title="Evidence-normalized candidate profile")
            fig.update_layout(yaxis_range=[0, 1], legend_title_text="Objective")
            _plot(fig, "recommender_llm_evidence", 360)

            if use_llm and st.button("Explain recommendation with local Ollama", key="rec_llm_explain"):
                facts = {
                    "dataset_profile": profile,
                    "goal": obj.get("goal"),
                    "parsed_weights": obj.get("weights"),
                    "recommended": best.to_dict(),
                    "top_candidates": cand.head(3).to_dict(orient="records"),
                    "meta_coverage": coverage,
                }
                q = "Explain why the recommended framework fits the user's goal. Mention uncertainty, nearest historical evidence, and that this is a pre-run estimate."
                answer, answer_meta = GroundedChat(model=llm_model).answer(q, {"frameworks": {}, "pre_run_recommendation": facts}, use_llm=True)
                if answer_meta.get("source") == "deterministic-fallback":
                    st.warning(answer_meta.get("warning", "Ollama request failed; deterministic fallback used."))
                st.markdown(answer)

    with tab_ml:
        st.markdown("### Explicit multi-objective ML recommender")
        st.caption("This is the second recommender from the previous dashboard, kept separate from the LLM-assisted path. It uses explicit preference weights and historical meta-logs.")
        cols = st.columns(4)
        weights = {
            "accuracy": cols[0].slider("Accuracy weight", 0.0, 1.0, 0.55, 0.05, key="mlrec_acc"),
            "runtime": cols[1].slider("Runtime weight", 0.0, 1.0, 0.15, 0.05, key="mlrec_rt"),
            "energy": cols[2].slider("Energy weight", 0.0, 1.0, 0.15, 0.05, key="mlrec_en"),
            "co2": cols[3].slider("CO₂ weight", 0.0, 1.0, 0.15, 0.05, key="mlrec_co2"),
        }
        if st.button("Recommend with ML meta-model", type="primary", use_container_width=True, key="mlrec_run"):
            candidates, rec_meta = recommender.predict_candidates(**profile, weights=weights)
            st.session_state["mlrec_candidates"] = candidates.to_dict(orient="records")
            st.session_state["mlrec_meta"] = rec_meta
        rows = st.session_state.get("mlrec_candidates")
        if rows:
            cand = pd.DataFrame(rows)
            best = cand.iloc[0]
            a, b, c, d, e = st.columns(5)
            a.metric("Top framework", best["framework"])
            b.metric("Accuracy", _metric_text(best.get("accuracy"), 2, "%"))
            c.metric("Runtime", _metric_text(best.get("runtime_sec"), 2, " s"))
            d.metric("Energy", _metric_text(best.get("energy_kwh"), 6, " kWh"))
            e.metric("Utility", _metric_text(best.get("utility"), 3))

            left, right = st.columns([1.45, 1])
            with left:
                plot_df = cand.copy()
                fig = px.scatter(
                    plot_df, x="runtime_sec", y="accuracy", size="utility", color="framework",
                    hover_data=["algorithm", "energy_kwh", "co2_ug", "nearest_dataset"],
                    title="Predicted performance / runtime frontier",
                )
                _plot(fig, "mlrec_frontier", 390)
            with right:
                score_cols = ["accuracy_score", "runtime_score", "energy_score", "co2_score"]
                heat = cand.set_index("framework")[score_cols].rename(columns=lambda c: c.replace("_score", "").title())
                fig = px.imshow(heat, text_auto=".2f", zmin=0, zmax=1, aspect="auto", color_continuous_scale="Viridis", title="Normalized objective score")
                _plot(fig, "mlrec_heatmap", 390)

            st.dataframe(
                cand[["rank", "framework", "algorithm", "accuracy", "runtime_sec", "energy_kwh", "co2_ug", "utility", "nearest_dataset"]],
                use_container_width=True, hide_index=True,
            )
            with st.expander("Suggested hyperparameters & nearest historical evidence", expanded=False):
                st.json(best.get("best_hyperparams", {}))
                meta = st.session_state.get("mlrec_meta") or {}
                neighbors = (meta.get("neighbors") or {}).get(str(best["framework"]), [])
                if neighbors:
                    st.dataframe(pd.DataFrame(neighbors), use_container_width=True, hide_index=True)

    with tab_eval:
        st.markdown("### Historical meta-evidence quality")
        counts = pd.DataFrame(list((coverage.get("frameworks") or {}).items()), columns=["Framework", "Runs"])
        if not counts.empty:
            left, right = st.columns([1, 1.25])
            with left:
                fig = px.bar(counts, x="Framework", y="Runs", color="Framework", title="Meta-log coverage by framework")
                fig.update_layout(showlegend=False)
                _plot(fig, "meta_coverage_framework", 340)
            with right:
                metric_rows = []
                for fw, block in meta_df.groupby("framework"):
                    metric_rows.append({
                        "Framework": fw,
                        "Datasets": int(block["dataset_name"].nunique()),
                        "Accuracy mean": float(pd.to_numeric(block["accuracy"], errors="coerce").mean()),
                        "Runtime median": float(pd.to_numeric(block["runtime_sec"], errors="coerce").median()),
                        "Energy measured": float(pd.to_numeric(block["energy_consumption_kwh"], errors="coerce").notna().mean()),
                        "CO₂ measured": float(pd.to_numeric(block.get("co2_ug"), errors="coerce").notna().mean()),
                    })
                st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
        st.warning(
            "Research note: historical zeros that indicate unmeasured carbon are treated as missing before ML fitting. "
            "The recommender should be evaluated with leave-one-dataset-out validation before publication claims."
        )
        st.markdown("**Recommended evaluation:** Top-1 framework selection · normalized regret · rank correlation · per-dataset coverage · calibration of predicted utility.")


def trust_page():
    hero("Controlled study", "Trust Calibration", "Hold explanation style constant while recommendation reliability changes, then measure whether trust follows evidence.")
    st.markdown('<div class="research-warning"><b>Research mode only.</b> Wrong/weak recommendations here are deliberate experimental manipulations and never replace the operational recommendation.</div>', unsafe_allow_html=True)
    if not _state().get("ranking"):
        st.info("Create a ranking in Decision Lab first.")
        return
    if "trust_case" not in st.session_state:
        st.session_state.trust_case = None
    c1, c2 = st.columns([1, 1.4])
    with c1:
        condition = st.selectbox("Condition assignment", ["Randomized", "correct", "weak", "wrong"], key="trust_condition")
        if st.button("Generate matched trial", type="primary", key="trust_generate"):
            study = TrustCalibrationStudy(seed=int(time.time()) % 100000)
            st.session_state.trust_case = study.build_case(_state()["ranking"], None if condition == "Randomized" else condition)
    case = st.session_state.trust_case
    if case:
        with c2:
            st.markdown("### Recommendation shown to participant")
            st.markdown(case.explanation)
            st.caption("Explanation wording/structure is held constant across reliability conditions.")
        st.divider()
        trust = st.slider("How much do you trust this recommendation?", 1, 7, 4, key="trust_score")
        correctness = st.slider("How likely is it to be correct?", 1, 7, 4, key="trust_correct")
        confidence = st.slider("How confident are you in your decision?", 1, 7, 4, key="trust_conf")
        accept = st.radio("Would you accept this recommendation?", ["Yes", "No"], horizontal=True, key="trust_accept")
        session = st.text_input("Participant/session code", value="pilot-session", key="trust_session")
        if st.button("Save trial response", key="trust_save"):
            StudyStore().log("trust", session, "trial_response", {**case.to_dict(), "trust": trust, "perceived_correctness": correctness, "decision_confidence": confidence, "accepted": accept == "Yes"})
            st.success("Stored with a hashed session identifier.")
            with st.expander("Experimenter-only condition check"):
                st.json({"condition": case.condition, "oracle": case.oracle_framework, "shown": case.shown_framework, "reliability": case.reliability})


def information_seeking_page():
    hero("Qualitative study", "Information-Seeking Lab", "Observe what users do after a recommendation: request evidence, challenge it, compare alternatives, clarify concepts, or stop probing.")
    if not _results():
        st.info("Run a benchmark first so chat has grounded evidence.")
        return
    use_llm, llm_model, _ = _ollama_controls("info_llm", compact=True)
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    if not _state().get("chat_session_id"):
        _state()["chat_session_id"] = str(uuid.uuid4())
    chat = GroundedChat(model=llm_model)
    facts = chat.build_facts(_result_dicts(), _state().get("ranking"))
    for item in st.session_state.chat_log:
        with st.chat_message(item["role"]):
            st.markdown(item["text"])
    q = st.chat_input("Ask about the evidence…")
    if q:
        st.session_state.chat_log.append({"role": "user", "text": q})
        category = classify_follow_up(q)
        started = time.perf_counter()
        answer, meta = chat.answer(q, facts, use_llm=use_llm)
        latency = time.perf_counter() - started
        st.session_state.chat_log.append({"role": "assistant", "text": answer})
        StudyStore().log("information_seeking", _state()["chat_session_id"], "chat_turn", {
            "question": q, "category": category, "answer_source": meta.get("source"), "model": meta.get("model"),
            "response_time_sec": latency, "turn": len(st.session_state.chat_log) // 2,
        })
        st.rerun()
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Think-aloud prompts")
        for prompt in THINK_ALOUD_PROMPTS:
            st.markdown(f"- {prompt}")
    with c2:
        st.subheader("Session summary")
        users = [x for x in st.session_state.chat_log if x["role"] == "user"]
        cats = [classify_follow_up(x["text"]) for x in users]
        st.metric("Follow-up questions", len(users))
        st.write(pd.Series(cats).value_counts().rename_axis("behavior").to_frame("count") if cats else "No questions yet")
        accepted = st.checkbox("Participant accepted the first answer without further probing", value=False, key="info_accepted")
        if st.button("Save session endpoint", key="info_save"):
            StudyStore().log("information_seeking", _state()["chat_session_id"], "session_end", {"first_answer_accepted": accepted, "follow_up_depth": len(users)})
            st.success("Session endpoint stored.")


def protocol_page():
    hero("Reproducibility", "Research Protocol", "Turn reviewer concerns and fairness-aware AutoML ideas into explicit streaming experiments rather than hidden implementation choices.")
    tabs = st.tabs(["Streaming evaluation", "Human-centric LLM + AutoML", "Fairness-aware optimization", "Recommendation", "Sustainability", "Human studies", "LLM & privacy"])
    with tabs[0]:
        st.markdown("**Primary rule:** prequential test-then-train, preserve temporal order, and use the same budget for all frameworks. Report sample cap, window size, seed, drift detector and missing-value treatment.")
    with tabs[1]:
        st.markdown("""
**Human-centric static AutoML → streaming AwareML**

| Human-centric LLM+AutoML pattern | Streaming translation in AwareML |
|---|---|
| Natural-language configuration | Natural-language objective parsing in Decision Lab with typed, deterministic fallback |
| Human-readable configuration artifact | Export the exact streaming protocol as YAML from Run Studio |
| Local LLM through Ollama | Discover installed Ollama models and let the user select any local model |
| Interactive config refinement | Keep target, sensitive attribute, positive label, budgets and objectives explicitly user-confirmable |
| LLM contextual help | Ground answers only in derived run facts; do not send raw stream rows by default |
| Static train/test evaluation | Prequential test→observe→train evaluation plus drift-aware temporal evidence |

**Extension-study opportunity:** compare *manual configuration*, *deterministic assistant*, and *Ollama-assisted configuration* on task completion, configuration errors, time-to-valid-run, objective regret, trust calibration and follow-up information seeking. This makes the human-centric idea a controlled streaming study rather than only a conversational feature.
""")
    with tabs[2]:
        st.markdown("""
**Static fairness-aware AutoML idea → AwareML streaming translation**

| Static idea | Streaming extension in AwareML |
|---|---|
| Put fairness inside optimization | Fairness receives an explicit utility weight in Decision Lab |
| Complementary fairness criteria | DP, EO, Equalized Odds, Predictive Parity and Error-rate gaps are selectable |
| Study fairness/performance trade-off | Accuracy-vs-fairness Pareto view on the current stream |
| Avoid assuming one metric captures fairness | Criterion is recorded with every ranking; composite mode aggregates available gaps |
| Compare complete solutions | Compare all five streaming AutoML backends under identical temporal budgets |

A next paper should additionally evaluate **fairness adaptation over drift**, not only a final gap: recovery time after drift, worst-window disparity, variance of disparity, and accuracy/fairness regret over time.
""")
    with tabs[3]:
        st.markdown("Hold out entire datasets, not random rows from the same dataset. Report Top-1 selection accuracy, normalized regret and rank correlation against multiple baselines. Define utility and epsilon-Pareto mathematically.")
    with tabs[4]:
        st.markdown("Repeat measured runs, capture hardware/software/region context, and treat missing measurements as missing. Analyze energy/CO₂ correlation before assigning both independent weights.")
    with tabs[5]:
        st.markdown("Trust calibration: matched explanation style with correct/weak/wrong reliability. Information seeking: combine interaction logs with think-aloud/interviews and code follow-up behavior.")
    with tabs[6]:
        st.markdown("Only derived experiment facts enter the LLM context by default. Discover locally installed Ollama models, validate natural-language objectives against a typed schema, and fall back deterministically when Ollama is unavailable or malformed.")
    matrix_path = Path("docs/REVIEWER_RESPONSE_MATRIX.md")
    if matrix_path.exists():
        st.download_button("Download reviewer-response matrix", matrix_path.read_text(encoding="utf-8"), file_name="REVIEWER_RESPONSE_MATRIX.md")


PAGE_REGISTRY = {
    "Overview": overview_page,
    "Run Studio": run_studio_page,
    "Recommender Lab": recommender_lab_page,
    "Decision Lab": decision_lab_page,
    "Drift & Temporal Lab": drift_temporal_page,
    "Fairness Lab": fairness_page,
    "Explainability Lab": explainability_page,
    "Sustainability Lab": sustainability_page,
    "Trust Calibration": trust_page,
    "Information-Seeking Lab": information_seeking_page,
    "Research Protocol": protocol_page,
}
