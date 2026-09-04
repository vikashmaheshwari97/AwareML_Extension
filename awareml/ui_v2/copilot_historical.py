from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from awareml.recommender.historical_preference import (
    HistoricalPreferenceError,
    HistoricalPreferenceRecommender,
    normalize_preference_weights,
)

from .state import ROOT, ensure_research_state


PRESET_WEIGHTS = {
    "Accuracy first": {"accuracy": 55, "runtime": 15, "energy": 15, "co2": 15},
    "Balanced": {"accuracy": 25, "runtime": 25, "energy": 25, "co2": 25},
    "Fast response": {"accuracy": 25, "runtime": 50, "energy": 15, "co2": 10},
    "Low energy": {"accuracy": 25, "runtime": 15, "energy": 50, "co2": 10},
    "Low CO₂": {"accuracy": 25, "runtime": 15, "energy": 10, "co2": 50},
}


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)
    except Exception:
        return {}


def _latest_copilot_weights(state: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    interpretation = state.get("copilot_context_free_interpretation")
    if interpretation is None:
        proposal = _as_dict(state.get("copilot_proposal"))
        interpretation = proposal.get("interpretation")
    data = _as_dict(interpretation)
    weights = _as_dict(data.get("primary_weights"))
    if not weights:
        return None
    normalized = normalize_preference_weights(weights)
    if sum(normalized.values()) <= 0:
        return None
    return normalized


def _set_slider_weights(weights: Mapping[str, float]) -> None:
    normalized = normalize_preference_weights(weights)
    st.session_state["histmeta_accuracy"] = int(round(normalized["accuracy"] * 100))
    st.session_state["histmeta_runtime"] = int(round(normalized["runtime"] * 100))
    st.session_state["histmeta_energy"] = int(round(normalized["energy"] * 100))
    st.session_state["histmeta_co2"] = int(round(normalized["co2"] * 100))


def _preference_badges(weights: Mapping[str, float]) -> str:
    labels = {
        "accuracy": "Accuracy ↑",
        "runtime": "Runtime ↓",
        "energy": "Energy ↓",
        "co2": "CO₂ ↓",
    }
    return " · ".join(
        "{} {:.0%}".format(labels[key], float(weights.get(key, 0.0)))
        for key in ("accuracy", "runtime", "energy", "co2")
        if float(weights.get(key, 0.0)) > 0
    )


def _format_metric(key: str, value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "N/A"
    if key == "accuracy":
        return "{:.4f}".format(val)
    if key == "runtime":
        return "{:.3f} s".format(val)
    if key == "energy":
        return "{:.6f} kWh".format(val)
    if key == "co2":
        return "{:.6f} kg".format(val)
    return "{:.4f}".format(val)


def _winner_explanation(ranking: pd.DataFrame, weights: Mapping[str, float]) -> str:
    if ranking.empty:
        return "Historical evidence is unavailable."
    top = ranking.iloc[0]
    active = []
    for key, label in [
        ("accuracy", "accuracy"),
        ("runtime", "runtime"),
        ("energy", "energy use"),
        ("co2", "CO₂ emissions"),
    ]:
        if float(weights.get(key, 0.0)) > 0:
            active.append("{} {:.0%}".format(label, float(weights[key])))
    return (
        "Across the 47 historical development datasets, **{}** has the highest robust "
        "preference score for this mix of priorities ({}). It is therefore a useful "
        "dataset-free starting point—not a prediction for a specific new dataset."
    ).format(str(top["framework"]), ", ".join(active))


def _contribution_table(ranking: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    row = ranking.iloc[0]
    rows = []
    for key, label in [
        ("accuracy", "Accuracy ↑"),
        ("runtime", "Runtime ↓"),
        ("energy", "Energy ↓"),
        ("co2", "CO₂ ↓"),
    ]:
        weight = float(weights.get(key, 0.0))
        desirability = float(row.get(f"{key}_desirability", 0.0) or 0.0)
        rows.append(
            {
                "Priority": label,
                "Your weight": "{:.0%}".format(weight),
                "Historical desirability": round(desirability, 3),
                "Weighted contribution": round(weight * desirability, 3),
            }
        )
    return pd.DataFrame(rows)


def _ranking_display(ranking: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "rank",
        "framework",
        "historical_utility",
        "win_rate",
        "top3_rate",
        "utility_iqr",
        "accuracy_median",
        "runtime_median_sec",
        "energy_median_kwh",
        "co2_median_kg",
        "support_datasets",
    ]
    display = ranking[[c for c in cols if c in ranking.columns]].copy()
    return display.rename(
        columns={
            "rank": "Rank",
            "framework": "Framework",
            "historical_utility": "Historical preference score",
            "win_rate": "Cross-dataset win rate",
            "top3_rate": "Top-3 rate",
            "utility_iqr": "Utility IQR",
            "accuracy_median": "Median accuracy",
            "runtime_median_sec": "Median runtime (s)",
            "energy_median_kwh": "Median energy (kWh)",
            "co2_median_kg": "Median CO₂ (kg)",
            "support_datasets": "Datasets",
        }
    )


def _append_review(state: Mapping[str, Any], result: Mapping[str, Any], decision: str, note: str) -> Dict[str, Any]:
    path = Path(ROOT) / "artifacts" / "copilot" / "historical_meta_reviews.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "review_id": "HMR-{}".format(uuid.uuid4().hex[:10].upper()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "note": note or None,
        "winner": result.get("winner"),
        "weights": dict(result.get("weights") or {}),
        "seed_mode": result.get("seed_mode"),
        "dataset_count": result.get("dataset_count"),
        "run_count": result.get("run_count"),
        "dataset_specific": False,
        "source": "historical_meta_recommender_705",
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def _render_result(state: Dict[str, Any]) -> None:
    result = state.get("historical_meta_result")
    ranking = state.get("historical_meta_ranking")
    if not isinstance(result, dict) or not isinstance(ranking, pd.DataFrame) or ranking.empty:
        return

    weights = dict(result.get("weights") or {})
    winner = str(result.get("winner") or ranking.iloc[0]["framework"])
    algorithm = str(result.get("algorithm") or "N/A")
    seed_mode = str(result.get("seed_mode") or "")
    top = ranking.iloc[0]

    st.markdown("## Historical AutoML Planner Summary")
    st.info(
        "This is a **dataset-free historical recommendation**. It summarizes what worked across the frozen 47-dataset development meta-evidence. "
        "It is not a dataset-specific prediction and does not replace ML Recommender V2."
    )

    cards = st.columns(4)
    cards[0].metric("Historical recommendation", winner)
    cards[1].metric("Validated default algorithm", algorithm)
    cards[2].metric("Historical preference score", "{:.3f}".format(float(top["historical_utility"])))
    cards[3].metric("Cross-dataset wins", "{} / {}".format(int(top["win_count"]), int(top["support_datasets"])))
    st.caption(
        "Historical preference score is a normalized cross-dataset ranking score under your current priorities. It is not a probability or model confidence."
    )

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("### Why this framework?")
        st.markdown(_winner_explanation(ranking, weights))
        st.markdown("**Your priorities:** {}".format(_preference_badges(weights)))
        st.dataframe(_contribution_table(ranking, weights), use_container_width=True, hide_index=True)
        st.caption(
            "Historical desirability is normalized within each development dataset before aggregation, so every dataset contributes equally to the comparison."
        )

    with right:
        st.markdown("### What this recommendation knows")
        knowledge = pd.DataFrame(
            [
                {"Evidence": "Development datasets", "Value": str(int(result.get("dataset_count", 47)))},
                {"Evidence": "Historical framework runs", "Value": str(int(result.get("run_count", 705)))},
                {"Evidence": "Framework candidates", "Value": "5"},
                {
                    "Evidence": "Seed handling",
                    "Value": "3-seed aggregate" if seed_mode == HistoricalPreferenceRecommender.STABLE else "best observed single seed (exploratory)",
                },
                {"Evidence": "Current dataset rows used", "Value": "No"},
            ]
        )
        st.dataframe(knowledge, use_container_width=True, hide_index=True)

        params = dict(result.get("parameters") or {})
        with st.expander("Validated default configuration for the recommended framework", expanded=False):
            st.write("**Framework:** {}".format(winner))
            st.write("**Algorithm:** {}".format(algorithm))
            if params:
                st.json(params, expanded=True)
            else:
                st.caption("No default parameter dictionary is registered for this framework.")
            st.caption(
                "Algorithm/defaults come from AwareML's validated configuration synthesizer, not from an LLM guess."
            )

    if seed_mode == HistoricalPreferenceRecommender.BEST_SEED:
        st.warning(
            "Exploratory best-seed mode is optimistic. For each dataset/framework pair it chooses **one actual seed** that best satisfies the whole preference vector. "
            "It never combines accuracy from one seed with runtime/energy/CO₂ from another seed. Use the stable 3-seed aggregate for primary reporting."
        )
        usage = (result.get("seed_usage") or {}).get(winner) or {}
        if usage:
            st.caption(
                "Selected-seed counts for {} across the 47 datasets: {}".format(
                    winner,
                    " · ".join("seed {}: {}".format(seed, count) for seed, count in sorted(usage.items())),
                )
            )

    if float(weights.get("energy", 0.0)) > 0 and float(weights.get("co2", 0.0)) > 0:
        st.warning(
            "Energy and CO₂ are strongly correlated in the Phase-13 development analysis. This recommender preserves both preferences exactly as you set them and adds no extra sustainability bonus; large weights on both can still emphasize the same efficiency signal."
        )

    st.markdown("### Historical ranking")
    st.dataframe(_ranking_display(ranking), use_container_width=True, hide_index=True)

    chart = ranking.sort_values("historical_utility", ascending=True).copy()
    fig = px.bar(
        chart,
        x="historical_utility",
        y="framework",
        orientation="h",
        text="historical_utility",
        labels={"historical_utility": "Historical preference score", "framework": "Framework"},
        title="Preference-only historical ranking",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="inside")
    fig.update_layout(height=330, margin=dict(l=80, r=20, t=48, b=48), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="histmeta_ranking_chart")

    if state.get("dataset") is None:
        st.success(
            "Use this as an early planning prior. When you have a dataset, load it in Run Studio and return to **Goal Copilot** for a dataset-aware ML Recommender V2 prediction."
        )
    else:
        st.info(
            "A dataset is currently loaded, but this tab intentionally ignores it. For the primary dataset-aware recommendation, switch to **Goal Copilot**."
        )

    st.markdown("### Your review")
    st.caption("The historical suggestion is advisory. Record whether it is useful as a starting point; your choice does not alter the frozen meta-evidence.")
    decision = st.segmented_control(
        "How do you want to treat this historical suggestion?",
        ["Use as starting point", "Keep as alternative", "Reject"],
        default="Use as starting point",
        key="histmeta_review_decision",
    ) or "Use as starting point"
    note = st.text_input(
        "Optional review note",
        key="histmeta_review_note",
        placeholder="Example: Useful before dataset upload, but I want a dataset-specific check before execution.",
    )
    if st.button("Save historical recommendation review", key="histmeta_save_review"):
        record = _append_review(state, result, decision, note)
        state["historical_meta_review"] = record
        st.success("Historical recommendation review saved as {}.".format(record["review_id"]))
    if isinstance(state.get("historical_meta_review"), dict):
        review = state["historical_meta_review"]
        st.caption("Saved review: {} · {}".format(review.get("review_id"), review.get("decision")))


def render_historical_meta_recommender_tab() -> None:
    state = ensure_research_state()

    st.markdown("# Historical Meta-Recommender")
    st.markdown(
        "Get a **framework starting point before uploading a dataset**. This mode uses the frozen historical meta-evidence rather than asking the LLM to invent a framework."
    )

    source_cards = st.columns(4)
    source_cards[0].metric("Historical runs", "705")
    source_cards[1].metric("Development datasets", "47")
    source_cards[2].metric("Frameworks", "5")
    source_cards[3].metric("Seeds per dataset/framework", "3")

    st.markdown(
        """
        <div class="r9-callout">
          <b>How this differs from Goal Copilot:</b><br>
          <b>Goal Copilot + ML Recommender V2</b> is dataset-aware and uses a dataset meta-profile.<br>
          <b>Historical Meta-Recommender</b> is dataset-free and asks: “Across the 47 development datasets, which framework best matched these preferences?”<br><br>
          The LLM does <b>not</b> choose the framework in this tab.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 1 · Choose what matters")
    st.caption("Use a quick profile or tune the four priorities yourself. Values are normalized automatically before ranking.")

    preset_cols = st.columns(5)
    for col, (label, values) in zip(preset_cols, PRESET_WEIGHTS.items()):
        with col:
            if st.button(label, key="histmeta_preset_{}".format(label.lower().replace(" ", "_")), use_container_width=True):
                _set_slider_weights(values)
                st.rerun()

    latest = _latest_copilot_weights(state)
    if latest:
        if st.button("Use latest Goal Copilot priorities", key="histmeta_import_copilot"):
            _set_slider_weights(latest)
            st.rerun()
        st.caption("Latest Goal Copilot priorities available: {}".format(_preference_badges(latest)))

    defaults = PRESET_WEIGHTS["Accuracy first"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        accuracy = st.slider("Accuracy ↑", 0, 100, int(st.session_state.get("histmeta_accuracy", defaults["accuracy"])), key="histmeta_accuracy")
    with c2:
        runtime = st.slider("Runtime ↓", 0, 100, int(st.session_state.get("histmeta_runtime", defaults["runtime"])), key="histmeta_runtime")
    with c3:
        energy = st.slider("Low energy ↓", 0, 100, int(st.session_state.get("histmeta_energy", defaults["energy"])), key="histmeta_energy")
    with c4:
        co2 = st.slider("Low CO₂ ↓", 0, 100, int(st.session_state.get("histmeta_co2", defaults["co2"])), key="histmeta_co2")

    weights = normalize_preference_weights(
        {"accuracy": accuracy, "runtime": runtime, "energy": energy, "co2": co2}
    )
    st.info("Effective normalized priorities: {}".format(_preference_badges(weights)))

    st.markdown("## 2 · Choose how to use the three seeds")
    mode_label = st.segmented_control(
        "Seed evidence",
        ["Stable 3-seed aggregate", "Best observed single seed · exploratory"],
        default="Stable 3-seed aggregate",
        key="histmeta_seed_mode",
        help="Stable mode averages all three seeds before ranking. Exploratory mode chooses one whole observed seed per dataset/framework using the complete preference vector.",
    ) or "Stable 3-seed aggregate"

    if mode_label == "Stable 3-seed aggregate":
        seed_mode = HistoricalPreferenceRecommender.STABLE
        st.success(
            "Recommended for research reporting: each dataset/framework is represented by the mean of all three seeds, then the five frameworks are compared within each dataset and aggregated across 47 datasets."
        )
    else:
        seed_mode = HistoricalPreferenceRecommender.BEST_SEED
        st.warning(
            "Exploratory upper-bound view: chooses one actual seed per dataset/framework using the full Accuracy/Runtime/Energy/CO₂ preference score. It does not cherry-pick different metrics from different seeds."
        )

    with st.expander("How the 705 runs are converted into one recommendation", expanded=False):
        st.markdown(
            """
            **Stable mode**
            1. Start from 47 datasets × 5 frameworks × 3 seeds = **705 runs**.
            2. Aggregate the three seeds for every dataset/framework pair → **235 stable profiles**.
            3. Within each dataset, convert Accuracy/Runtime/Energy/CO₂ to 0–1 desirability scores.
            4. Apply your normalized preference weights.
            5. Give every dataset equal weight and aggregate performance across all 47 datasets.
            6. Rank the five frameworks by robust historical preference score.

            **Exploratory best-seed mode** uses the same 705 runs, but selects **one entire observed seed** for each dataset/framework based on the complete weighted preference score before cross-dataset aggregation. It never constructs a synthetic run by mixing metrics from different seeds.
            """
        )

    st.markdown("## 3 · Generate the historical recommendation")
    if st.button("Recommend from 705 historical runs", type="primary", use_container_width=True, key="histmeta_recommend"):
        try:
            recommender = HistoricalPreferenceRecommender()
            with st.spinner("Ranking historical meta-evidence across 47 datasets..."):
                result = recommender.recommend(weights, seed_mode=seed_mode)
            state["historical_meta_result"] = result.as_dict()
            state["historical_meta_ranking"] = result.ranking
            state["historical_meta_result"]["seed_usage"] = result.seed_usage
            st.success("Historical recommendation generated from frozen meta-evidence.")
        except (HistoricalPreferenceError, FileNotFoundError, ValueError) as exc:
            st.error("Historical meta-recommendation is unavailable: {}".format(exc))
        except Exception as exc:
            st.error("Historical meta-recommendation failed safely: {}".format(type(exc).__name__))

    stored = state.get("historical_meta_result")
    if isinstance(stored, dict):
        stored_weights = normalize_preference_weights(stored.get("weights") or {})
        changed = any(abs(float(stored_weights[k]) - float(weights[k])) > 1e-9 for k in weights)
        changed = changed or str(stored.get("seed_mode")) != str(seed_mode)
        if changed:
            st.info("Your preferences or seed policy changed. Click **Recommend from 705 historical runs** to refresh the recommendation; the previous result is hidden to avoid showing stale advice.")
            return

    _render_result(state)
