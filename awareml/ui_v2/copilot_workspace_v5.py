from __future__ import annotations

import streamlit as st

from .recommender_bridge import bundle_status, sync_copilot_evidence


def _css() -> None:
    st.markdown(
        """
        <style>
        .copv5-hero{
            border:1px solid rgba(148,163,184,.36);border-radius:20px;
            padding:24px 26px 20px;background:
            radial-gradient(circle at 92% 5%,rgba(59,130,246,.14),transparent 30%),
            linear-gradient(135deg,rgba(255,255,255,.98),rgba(244,248,255,.96));
            box-shadow:0 10px 30px rgba(15,23,42,.055);margin-bottom:14px}
        .copv5-kicker{color:#2563eb;text-transform:uppercase;letter-spacing:.12em;
            font-weight:800;font-size:11px;margin-bottom:7px}
        .copv5-title{font-size:34px;font-weight:820;line-height:1.15;
            color:#0f274d;margin-bottom:8px}
        .copv5-sub{max-width:1080px;color:#334155;font-size:14px;line-height:1.6}
        .copv5-path{border:1px solid rgba(148,163,184,.38);border-radius:16px;
            background:#fff;padding:16px 17px;min-height:184px;
            box-shadow:0 6px 20px rgba(15,23,42,.035)}
        .copv5-num{color:#2563eb;font-size:12px;font-weight:800;
            letter-spacing:.08em;margin-bottom:7px}
        .copv5-path-title{color:#0f172a;font-size:17px;font-weight:800;margin-bottom:8px}
        .copv5-copy{color:#475569;font-size:12px;line-height:1.5;min-height:54px}
        .copv5-status{display:inline-block;border-radius:999px;border:1px solid #cbd5e1;
            padding:4px 9px;margin-top:11px;font-size:11px;font-weight:700;
            color:#334155;background:#f8fafc}
        .copv5-current{border-left:4px solid #2563eb;border-radius:11px;
            background:#eef5ff;padding:12px 15px;margin:8px 0 16px;color:#1e293b;
            line-height:1.5}
        .copv5-guide{border:1px solid rgba(148,163,184,.40);border-radius:15px;
            background:#fff;padding:15px 16px;min-height:126px}
        .copv5-label{color:#64748b;font-size:12px;margin-bottom:7px}
        .copv5-value{color:#0f172a;font-size:24px;line-height:1.2;font-weight:820}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_copilot_workspace_header(state):
    _css()
    status = sync_copilot_evidence(state)
    frame, target = state.get("dataset"), state.get("target")
    has_dataset = frame is not None and bool(target) and target in frame.columns
    active = str(state.get("copilot_workspace_path") or "goal")
    if active not in {"goal", "historical", "dataset"}:
        active = "goal"
    if status["ready"]:
        prior_copy = "{} development datasets · {} validated development runs · {}-seed aggregation → global framework starting point.".format(
            status["development_datasets"], status["underlying_runs"], len(status["seeds"]))
        dataset_copy = "First {:,} labeled observations + trained objective models → dataset-specific ranking of {} frameworks.".format(
            int(status["manifest"]["profile_size"]), status["framework_count"])
        prior_status = "Available"
        dataset_status = "Ready" if has_dataset else "Waiting for dataset + target"
    else:
        prior_copy = "Seed-aggregated development evidence → global framework starting point after validation and training."
        dataset_copy = "Observed labeled prefix + trained objective models → dataset-specific ranking once the model is available."
        prior_status = "Training pending" if status["state"] == "training_pending" else "Evidence unavailable"
        dataset_status = prior_status
        if not has_dataset:
            dataset_status += " · Dataset + target required"

    st.markdown(
        '<div class="copv5-hero"><div class="copv5-kicker">Human-centred evidence orchestration</div>'
        '<div class="copv5-title">AwareML Copilot Workspace</div>'
        '<div class="copv5-sub">One deployment goal, three evidence levels. Interpret priorities, review '
        'development evidence, and use the observed labeled prefix for dataset-aware recommendations.</div></div>',
        unsafe_allow_html=True,
    )
    cards = [
        ("goal", "01 · Goal interpretation",
         "Natural-language goal → Accuracy / Runtime / Energy / CO₂ priorities, plus separate HCAI oversight requirements.", "Ready"),
        ("historical", "02 · Historical preference prior", prior_copy, prior_status),
        ("dataset", "03 · Dataset-aware ML Recommender", dataset_copy, dataset_status),
    ]
    for column, (key, title, copy, readiness) in zip(st.columns(3), cards):
        with column:
            with st.container(border=True):
                if st.button(title, key="copilot_workspace_open_" + key, use_container_width=True,
                             type="primary" if active == key else "secondary"):
                    state["copilot_workspace_path"] = key
                    st.rerun()
                st.caption(copy)
                st.markdown("**{}{}**".format(readiness, " · Active" if active == key else ""))
    title = "Dataset-aware decision support" if has_dataset else "Dataset-free planning"
    copy = ("Goal interpretation is available. Dataset-aware ranking additionally requires a trained model bundle and a labeled dataset prefix."
            if not status["ready"] else
            "The historical prior summarizes development evidence. Dataset-aware ranking uses the observed prefix and the active priorities.")
    st.markdown('<div class="copv5-current"><b>Current evidence mode: {}</b><br>{}</div>'.format(title, copy), unsafe_allow_html=True)
    return active


def render_goal_framework_guidance(state, interpretation, proposal=None):
    from .copilot_recommender import render_current_goal_guidance
    return render_current_goal_guidance(state, interpretation, proposal)
