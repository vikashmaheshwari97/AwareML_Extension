from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .plots import FRAMEWORK_COLORS, apply_research_layout
from .page_utils import plot


def _points_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for p in result.get("points") or []:
        if hasattr(p, "__dict__"):
            p = dict(p.__dict__)
        if isinstance(p, dict):
            rows.append(dict(p))
    return pd.DataFrame(rows)


def _nearest_value(frame: pd.DataFrame, sample: int, column: str, direction: str = "after"):
    if frame.empty or "sample" not in frame or column not in frame:
        return None
    f = frame.copy()
    f["sample"] = pd.to_numeric(f["sample"], errors="coerce")
    f[column] = pd.to_numeric(f[column], errors="coerce")
    f = f.dropna(subset=["sample", column])
    if f.empty:
        return None
    if direction == "after":
        cand = f[f["sample"] >= sample].sort_values("sample")
    else:
        cand = f[f["sample"] <= sample].sort_values("sample", ascending=False)
    if cand.empty:
        return None
    return float(cand.iloc[0][column])


def adaptation_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results or []:
        events = list(result.get("refit_events") or [])
        if not events:
            events = list((result.get("drift_summary") or {}).get("refit_events") or [])
        for event in events:
            if not isinstance(event, dict):
                event = {"sample": event}
            rows.append({
                "Framework": result.get("framework"),
                "Sample": event.get("sample"),
                "Action": event.get("action") or event.get("policy") or "recorded adaptation",
                "Status": event.get("status") or "recorded",
                "Replayed samples": event.get("replayed_samples"),
                "Trigger": ", ".join(event.get("trigger_sources") or []),
                "Duration (s)": event.get("duration_sec"),
            })
    return pd.DataFrame(rows)


def detector_health_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results or []:
        ds = result.get("drift_summary") or {}
        det = ds.get("detector") or {}
        adaptations = list(result.get("refit_events") or ds.get("refit_events") or [])
        # v3 stored `warnings` as a per-sample warning count. v4 separates
        # warning *steps* from warning *episodes* so a long persistent warning
        # no longer looks like thousands of independent drift episodes.
        has_v4_warning_semantics = ("warning_steps" in det) or ("warning_episodes" in det)
        legacy_warning_steps = det.get("warnings") if not has_v4_warning_semantics else None
        rows.append({
            "Framework": result.get("framework"),
            "Detector": det.get("name") or ds.get("detector_name") or "ADWIN / legacy",
            "Mode": det.get("mode") or "legacy",
            "Warning episodes": det.get("warning_episodes") if has_v4_warning_semantics else None,
            "Warning steps": det.get("warning_steps") if has_v4_warning_semantics else legacy_warning_steps,
            "Confirmed changes": det.get("confirmed_drifts", len(result.get("drift_events") or [])),
            "Confirmation rule": det.get("confirmation_rule") or ("legacy v3 warning-step telemetry" if legacy_warning_steps is not None else None),
            "Explicit adaptations": len(adaptations),
            "Recovery-applicable": ds.get("n_recovery_applicable"),
            "Recovered": ds.get("n_recovered"),
            "Recovery rate": ds.get("recovery_rate"),
            "Max accuracy drop": ds.get("max_accuracy_drop"),
        })
    return pd.DataFrame(rows)


def drift_episode_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results or []:
        fw = str(result.get("framework"))
        pts = _points_frame(result)
        ds = result.get("drift_summary") or {}
        adaptations = list(result.get("refit_events") or ds.get("refit_events") or [])
        adapt_samples = []
        for a in adaptations:
            if isinstance(a, dict):
                a = a.get("sample")
            try:
                adapt_samples.append(int(a))
            except Exception:
                pass
        for ep_i, ep in enumerate(ds.get("episodes") or [], 1):
            if not isinstance(ep, dict):
                continue
            drift_sample = ep.get("sample_index") or ep.get("drift_sample")
            try:
                drift_sample = int(drift_sample)
            except Exception:
                continue
            baseline = ep.get("baseline_accuracy")
            trough = ep.get("min_accuracy_after")
            recovery_sample = ep.get("recovered_at_sample")
            post_sample = recovery_sample
            explicit_adaptation = None
            after = [a for a in adapt_samples if a >= drift_sample]
            if after:
                explicit_adaptation = min(after)
                # Read a post-adaptation point at the next recorded temporal window.
                post_sample = explicit_adaptation
            if post_sample is None:
                assessment = ep.get("assessment_end_sample")
                post_sample = assessment if assessment is not None else drift_sample
            try:
                post_sample = int(post_sample)
            except Exception:
                post_sample = drift_sample
            post_acc = _nearest_value(pts, post_sample, "rolling_accuracy", direction="after")
            if post_acc is None and recovery_sample is not None:
                post_acc = _nearest_value(pts, int(recovery_sample), "rolling_accuracy", direction="after")
            b = None if baseline is None else float(baseline)
            t = None if trough is None else float(trough)
            ratio = None
            if b is not None and t is not None and post_acc is not None and b > t + 1e-12:
                ratio = (float(post_acc) - t) / (b - t)
            rows.append({
                "Framework": fw,
                "Episode": ep_i,
                "Drift sample": drift_sample,
                "Baseline accuracy": b,
                "Trough accuracy": t,
                "Post-event accuracy": post_acc,
                "Recovery ratio": ratio,
                "Recovered": recovery_sample is not None,
                "Recovery sample": recovery_sample,
                "Recovery lag": ep.get("recovery_samples"),
                "Explicit adaptation sample": explicit_adaptation,
                "Adaptation recorded": explicit_adaptation is not None,
            })
    return pd.DataFrame(rows)


def render_drift_episode_figures(results: Sequence[Mapping[str, Any]]) -> None:
    frame = drift_episode_frame(results)
    if frame.empty:
        st.info(
            "No scientifically assessable predictive-performance change episode is available for the current dataset/run. "
            "This means the active error/performance detector did not confirm an episode under its configured rule. "
            "It does not imply that fairness was temporally constant or that the raw feature distribution was identical."
        )
        return

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for fw, grp in frame.groupby("Framework", sort=False):
            color = FRAMEWORK_COLORS.get(str(fw), "#64748b")
            fig.add_trace(go.Scatter(
                x=grp["Episode"],
                y=grp["Post-event accuracy"],
                mode="markers+lines",
                name=str(fw),
                marker=dict(
                    symbol=["diamond" if bool(v) else "circle" for v in grp["Adaptation recorded"]],
                    size=12,
                    color=color,
                    line=dict(width=1.5, color="#111827"),
                ),
                line=dict(color=color, width=2),
                customdata=np.stack([
                    grp["Drift sample"].fillna(-1),
                    grp["Baseline accuracy"].fillna(np.nan),
                    grp["Trough accuracy"].fillna(np.nan),
                    grp["Recovery lag"].fillna(np.nan),
                ], axis=1),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>Episode %{x}<br>Post-event accuracy %{y:.3f}"
                    "<br>Drift sample %{customdata[0]}<br>Baseline %{customdata[1]:.3f}"
                    "<br>Trough %{customdata[2]:.3f}<br>Recovery lag %{customdata[3]}<extra></extra>"
                ),
            ))
        apply_research_layout(fig, height=430, legend="bottom", title="Post-drift / post-adaptation accuracy")
        fig.update_layout(xaxis_title="Detected drift episode on this dataset", yaxis_title="Rolling accuracy")
        plot(fig, "obs_post_drift_accuracy")

    with right:
        valid = frame.dropna(subset=["Recovery ratio"]).copy()
        if valid.empty:
            st.info("Drift-loss recovery ratio is unavailable because the run lacks a measurable degradation/post-event pair.")
        else:
            fig = go.Figure()
            for fw, grp in valid.groupby("Framework", sort=False):
                color = FRAMEWORK_COLORS.get(str(fw), "#64748b")
                fig.add_trace(go.Scatter(
                    x=grp["Episode"],
                    y=grp["Recovery ratio"],
                    mode="markers+lines",
                    name=str(fw),
                    marker=dict(size=12, symbol="diamond", color=color, line=dict(width=1.5, color="#111827")),
                    line=dict(color=color, width=2),
                ))
            fig.add_hline(y=0.80, line_dash="dot", line_color="rgba(37,99,235,.65)", annotation_text="0.80 recovery reference")
            apply_research_layout(fig, height=430, legend="bottom", title="Drift-loss recovery")
            fig.update_layout(xaxis_title="Detected drift episode on this dataset", yaxis_title="Recovered fraction of observed loss")
            plot(fig, "obs_drift_loss_recovery")
    st.caption(
        "Diamond markers indicate an explicitly recorded adaptation action when available. Recovery ratio = "
        "(post-event rolling accuracy − observed trough) / (pre-drift baseline − observed trough). "
        "Values above 1 indicate overshoot. No synthetic refit or recovery points are generated."
    )


def render_window_accuracy_heatmap(results: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for result in results or []:
        for p in result.get("points") or []:
            if hasattr(p, "__dict__"):
                p = p.__dict__
            if not isinstance(p, dict):
                continue
            if p.get("sample") is None or p.get("rolling_accuracy") is None:
                continue
            rows.append({
                "Framework": result.get("framework"),
                "Sample": p.get("sample"),
                "Rolling accuracy": p.get("rolling_accuracy"),
            })
    if not rows:
        return
    frame = pd.DataFrame(rows)
    heat = frame.pivot_table(index="Framework", columns="Sample", values="Rolling accuracy", aggfunc="last")
    z = heat.to_numpy(dtype=float)
    if not np.isfinite(z).any():
        return
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[str(x) for x in heat.columns],
        y=[str(x) for x in heat.index],
        colorscale="Viridis",
        colorbar=dict(title="Rolling accuracy"),
        hovertemplate="Framework %{y}<br>Sample %{x}<br>Rolling accuracy %{z:.3f}<extra></extra>",
    ))
    apply_research_layout(fig, height=390, legend="none", title="Window-level rolling accuracy")
    fig.update_layout(xaxis_title="Stream sample", yaxis_title="Framework")
    plot(fig, "obs_window_accuracy_heatmap")
