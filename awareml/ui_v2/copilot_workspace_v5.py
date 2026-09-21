from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd
import streamlit as st

from awareml.recommender.historical_preference import (
    HistoricalPreferenceRecommender,
    normalize_preference_weights,
)

OBJECTIVE_ORDER = ("accuracy", "runtime", "energy", "co2")
OBJECTIVE_LABELS = {
    "accuracy": "Accuracy",
    "runtime": "Runtime",
    "energy": "Energy",
    "co2": "CO₂",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    try:
        return dict(value)
    except Exception:
        return {}


def _dataset_ready_from_state(state: Mapping[str, Any]) -> bool:
    df = state.get("dataset")
    target = state.get("target")
    return bool(df is not None and target and target in getattr(df, "columns", []))


def _active_weights(state: Mapping[str, Any], interpretation: Any) -> Dict[str, float]:
    override = state.get("copilot_human_corrected_weights")
    if isinstance(override, dict) and override:
        return normalize_preference_weights(override)
    data = _as_dict(interpretation)
    return normalize_preference_weights(_as_dict(data.get("primary_weights")))


def _weights_text(weights: Mapping[str, float]) -> str:
    return " · ".join(
        "{} {:.0%}".format(OBJECTIVE_LABELS[k], float(weights.get(k, 0.0)))
        for k in OBJECTIVE_ORDER
    )


@st.cache_data(show_spinner=False)
def _historical_snapshot(
    accuracy: float,
    runtime: float,
    energy: float,
    co2: float,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    result = HistoricalPreferenceRecommender().recommend(
        weights={
            "accuracy": accuracy,
            "runtime": runtime,
            "energy": energy,
            "co2": co2,
        },
        seed_mode=HistoricalPreferenceRecommender.STABLE,
    )
    return result.as_dict(), result.ranking.copy()


def _get_historical(
    state: Dict[str, Any],
    interpretation: Any,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, float]]:
    weights = _active_weights(state, interpretation)
    result, ranking = _historical_snapshot(
        float(weights["accuracy"]),
        float(weights["runtime"]),
        float(weights["energy"]),
        float(weights["co2"]),
    )
    state["copilot_auto_historical_result"] = dict(result)
    state["copilot_auto_historical_ranking"] = ranking.copy()
    state["copilot_auto_historical_weights"] = dict(weights)
    return result, ranking, weights


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


def render_copilot_workspace_header(state: Mapping[str, Any]) -> str:
    _css()
    has_dataset = _dataset_ready_from_state(state)
    has_goal = bool(
        state.get("copilot_context_free_interpretation")
        or state.get("copilot_proposal")
    )
    active = str(state.get("copilot_workspace_path") or "goal")
    if active not in {"goal", "historical", "dataset"}:
        active = "goal"

    dataset_status = (
        "Ready · dataset + target available"
        if has_dataset else "Waiting for dataset + target"
    )
    mode_title = "Dataset-aware decision support" if has_dataset else "Dataset-free planning"
    mode_copy = (
        "Goal interpretation, the historical prior and dataset-aware recommendation "
        "are available for the uploaded dataset."
        if has_dataset else
        "Goal interpretation and the historical prior work without a dataset. Upload "
        "a dataset in Run Studio only when you want a dataset-specific recommendation."
    )

    st.markdown(
        """
        <div class="copv5-hero">
          <div class="copv5-kicker">Human-centred evidence orchestration</div>
          <div class="copv5-title">AwareML Copilot Workspace</div>
          <div class="copv5-sub">
            One deployment goal, three evidence levels. Goal interpretation is
            dataset-free; dataset-aware ranking is activated only when a dataset
            and target are available.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = [
        ("goal", "01 · Goal interpretation",
         "Natural-language goal → Accuracy / Runtime / Energy / CO₂ priorities, plus separate HCAI oversight requirements.",
         "Ready"),
        ("historical", "02 · Historical preference prior",
         "47 development datasets · 705 validated runs · stable 3-seed aggregation → global framework starting point.",
         "Ready from current goal" if has_goal else "Available"),
        ("dataset", "03 · Dataset-aware ML Recommender",
         "Dataset meta-profile + validated learned objective models → dataset-specific ranking of five frameworks.",
         dataset_status),
    ]
    cols = st.columns(3)
    for col, (key, title, copy, status) in zip(cols, cards):
        with col:
            with st.container(border=True):
                if st.button(
                    title,
                    key="copilot_workspace_open_{}".format(key),
                    use_container_width=True,
                    type="primary" if active == key else "secondary",
                ):
                    state["copilot_workspace_path"] = key
                    st.rerun()
                st.caption(copy)
                st.markdown("**{}{}**".format(status, " · Active" if active == key else ""))

    st.markdown(
        '<div class="copv5-current"><b>Current evidence mode: {}</b><br>{}</div>'.format(
            mode_title, mode_copy
        ),
        unsafe_allow_html=True,
    )
    return active

def _historical_cards(result: Mapping[str, Any], ranking: pd.DataFrame) -> None:
    top = ranking.iloc[0]
    cols = st.columns(4)
    values = [
        ("Framework starting point", str(top["framework"])),
        ("Historical rank", "#1 of {}".format(len(ranking))),
        ("Historical preference score", "{:.3f}".format(float(top["historical_utility"]))),
        (
            "Cross-dataset wins",
            "{} / {}".format(int(top["win_count"]), int(top["support_datasets"])),
        ),
    ]
    for col, (label, value) in zip(cols, values):
        with col:
            st.markdown(
                '<div class="copv5-guide">'
                '<div class="copv5-label">{}</div>'
                '<div class="copv5-value">{}</div>'
                '</div>'.format(label, value),
                unsafe_allow_html=True,
            )


def render_goal_framework_guidance(
    state: Dict[str, Any],
    interpretation: Any,
    proposal: Optional[Mapping[str, Any]] = None,
) -> None:
    try:
        result, ranking, weights = _get_historical(state, interpretation)
    except Exception as exc:
        st.warning(
            "Historical framework guidance could not be produced from the frozen "
            "development evidence: {}".format(exc)
        )
        return

    if ranking.empty:
        st.warning("Historical framework guidance is unavailable.")
        return

    historical_winner = str(ranking.iloc[0]["framework"])
    has_dataset = _dataset_ready_from_state(state)

    if not has_dataset or proposal is None:
        st.markdown("## 2 · Framework guidance from the evidence available now")
        st.info(
            "**Historical starting point · no dataset required.** "
            "AwareML uses the priorities inferred from your scenario to rank the five "
            "frameworks across the frozen 47-dataset / 705-run development evidence."
        )
        _historical_cards(result, ranking)
        st.caption(
            "This is a transparent historical aggregation, not a dataset-specific ML "
            "prediction and not a probability/confidence score."
        )

        st.markdown("### Why this framework is suggested")
        st.write(
            "**{}** has the highest global historical preference score under **{}**. "
            "Its validated default algorithm is **{}**. Use this as a planning starting "
            "point until dataset evidence becomes available.".format(
                historical_winner,
                _weights_text(weights),
                str(result.get("algorithm") or "N/A"),
            )
        )

        cols = [
            c for c in (
                "rank", "framework", "historical_utility", "win_rate",
                "top3_rate", "win_count", "support_datasets"
            )
            if c in ranking.columns
        ]
        with st.expander("Compare the historical alternatives", expanded=False):
            st.dataframe(ranking[cols], use_container_width=True, hide_index=True)
            st.caption(
                "Stable research view: each dataset/framework is represented by the "
                "mean of seeds 42/43/44 before cross-dataset aggregation."
            )

        st.markdown("### What improves when you add a dataset")
        c1, c2, c3 = st.columns(3)
        for col, title, value, body in (
            (
                c1,
                "Historical prior",
                "Available now",
                "Global evidence answers what generally worked across development datasets.",
            ),
            (
                c2,
                "Dataset-aware recommender",
                "Next evidence level",
                "Dataset meta-features answer what is predicted to work for this dataset.",
            ),
            (
                c3,
                "Human decision",
                "Always required",
                "You can correct priorities and review the recommendation before execution.",
            ),
        ):
            with col:
                with st.container(border=True):
                    st.markdown("**{}**".format(title))
                    st.markdown("### {}".format(value))
                    st.caption(body)
        return

    proposal_dict = _as_dict(proposal)
    dataset_winner = str(
        proposal_dict.get("ml_recommender_framework")
        or _as_dict(proposal_dict.get("proposed_config")).get("framework")
        or "N/A"
    )
    dataset_utility = proposal_dict.get("ml_recommender_utility")

    st.markdown("## 2 · Compare the available evidence paths")
    st.caption(
        "The same priorities can produce different framework guidance because the "
        "historical prior is global while ML Recommender conditions on this dataset."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.caption("GLOBAL HISTORICAL PRIOR")
            st.markdown("### {}".format(historical_winner))
            st.write(
                "Historical score: **{:.3f}**".format(
                    float(ranking.iloc[0]["historical_utility"])
                )
            )
            st.caption("47 datasets · 705 runs · transparent aggregation")
    with c2:
        with st.container(border=True):
            st.caption("DATASET-AWARE ML RECOMMENDER V2")
            st.markdown("### {}".format(dataset_winner))
            if dataset_utility is not None:
                st.write("Ranking utility: **{:.4f}**".format(float(dataset_utility)))
            st.caption("Current dataset meta-profile · learned pre-run prediction")
    with c3:
        with st.container(border=True):
            same = historical_winner == dataset_winner
            st.caption("EVIDENCE RELATIONSHIP")
            st.markdown("### {}".format("Agreement" if same else "Different · expected"))
            st.write("Both paths use **{}**.".format(_weights_text(weights)))
            st.caption(
                "Do not compare score magnitudes directly; the ranking mechanisms differ."
            )

    if historical_winner != dataset_winner:
        st.info(
            "The historical prior suggests **{}**, while ML Recommender predicts **{}** "
            "for the loaded dataset. This is expected: the first is global evidence and "
            "the second uses dataset-specific meta-features.".format(
                historical_winner, dataset_winner
            )
        )
    else:
        st.success(
            "Historical and dataset-aware evidence both favor **{}**. The two evidence "
            "paths remain methodologically distinct even when they agree.".format(
                dataset_winner
            )
        )
