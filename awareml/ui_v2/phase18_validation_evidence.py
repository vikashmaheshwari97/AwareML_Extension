from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .state import ROOT


EVIDENCE_DIR = ROOT / "data" / "journal" / "phase18_ui_evidence"
PREFERENCE_FILE = EVIDENCE_DIR / "phase18_preference_eval_primary_3100.csv"
WINNER_FILE = EVIDENCE_DIR / "phase18_predicted_vs_observed_objective_winners.csv"
SUMMARY_FILE = EVIDENCE_DIR / "phase18_preference_summary_by_dataset.csv"

PREFERENCE_COLUMNS = [
    "dataset_id", "cohort", "preference_id", "k_prime", "active_objectives",
    "w_accuracy", "w_runtime", "w_energy", "w_co2",
    "oracle_framework", "recommended_framework", "top1_match", "top3_contains_oracle",
    "oracle_real_utility", "recommended_real_utility", "normalized_regret",
    "normalized_regret_alt", "spearman_utility",
]
SUMMARY_COLUMNS = [
    "dataset_id", "cohort", "cases", "top1", "top3",
    "normalized_regret_mean", "spearman_mean",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_headerless(path: Path, columns) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, header=None, names=list(columns))


def load_phase18_ui_evidence(root: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(root) if root is not None else EVIDENCE_DIR
    pref_path = base / PREFERENCE_FILE.name
    winner_path = base / WINNER_FILE.name
    summary_path = base / SUMMARY_FILE.name

    pref = _read_headerless(pref_path, PREFERENCE_COLUMNS)
    summary = _read_headerless(summary_path, SUMMARY_COLUMNS)
    winners = pd.read_csv(winner_path)

    numeric_pref = [
        "w_accuracy", "w_runtime", "w_energy", "w_co2", "top1_match",
        "top3_contains_oracle", "oracle_real_utility", "recommended_real_utility",
        "normalized_regret", "normalized_regret_alt", "spearman_utility",
    ]
    for column in numeric_pref:
        pref[column] = pd.to_numeric(pref[column], errors="raise")
    for column in ("cases", "top1", "top3", "normalized_regret_mean", "spearman_mean"):
        summary[column] = pd.to_numeric(summary[column], errors="raise")
    winners["top1_match"] = winners["top1_match"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if winners["top1_match"].isna().any():
        raise ValueError("Objective-winner evidence contains an invalid top1_match value.")

    if len(pref) != 3100 or int(pref["dataset_id"].nunique()) != 31:
        raise ValueError("Phase-18 preference evidence must contain 3,100 cases across 31 datasets.")
    if len(summary) != 31 or int(summary["dataset_id"].nunique()) != 31:
        raise ValueError("Phase-18 per-dataset summary must contain exactly 31 datasets.")
    if len(winners) != 124 or int(winners["dataset_id"].nunique()) != 31:
        raise ValueError("Phase-18 objective-winner evidence must contain 124 rows across 31 datasets.")
    if set(pref["dataset_id"].astype(str)) != set(summary["dataset_id"].astype(str)):
        raise ValueError("Preference and summary dataset IDs do not match.")
    if set(pref["dataset_id"].astype(str)) != set(winners["dataset_id"].astype(str)):
        raise ValueError("Preference and objective-winner dataset IDs do not match.")

    return pref, winners, summary


def phase18_validation_snapshot(root: Optional[Path] = None) -> Dict[str, Any]:
    pref, winners, summary = load_phase18_ui_evidence(root=root)
    objective = (
        winners.groupby("objective", sort=False)["top1_match"]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .rename(columns={"size": "cases", "sum": "matches", "mean": "top1"})
    )
    cohort = (
        pref.groupby("cohort", sort=True)
        .agg(
            cases=("top1_match", "size"),
            datasets=("dataset_id", "nunique"),
            top1=("top1_match", "mean"),
            top3=("top3_contains_oracle", "mean"),
            normalized_regret=("normalized_regret", "mean"),
            spearman=("spearman_utility", "mean"),
        )
        .reset_index()
    )
    return {
        "datasets": int(pref["dataset_id"].nunique()),
        "preference_cases": int(len(pref)),
        "top1_matches": int(pref["top1_match"].sum()),
        "top1": float(pref["top1_match"].mean()),
        "top3_successes": int(pref["top3_contains_oracle"].sum()),
        "top3": float(pref["top3_contains_oracle"].mean()),
        "normalized_regret_mean": float(pref["normalized_regret"].mean()),
        "normalized_regret_median": float(pref["normalized_regret"].median()),
        "normalized_regret_p90": float(pref["normalized_regret"].quantile(0.90)),
        "spearman_mean": float(pref["spearman_utility"].mean()),
        "objective_winners": objective.to_dict(orient="records"),
        "cohort_summary": cohort.to_dict(orient="records"),
        "summary_rows": int(len(summary)),
        "winner_source_cohorts": sorted(winners["cohort"].astype(str).unique().tolist()),
    }


def _pct(value: Any) -> str:
    try:
        return "{:.2%}".format(float(value))
    except Exception:
        return "N/A"

def render_phase18_heldout_validation(
    expanded: bool = True,
    embedded: bool = False,
    collapsible: bool = False,
) -> None:
    try:
        pref, winners, summary = load_phase18_ui_evidence()
        snap = phase18_validation_snapshot()
    except Exception as exc:
        st.warning("31-dataset held-out validation evidence is unavailable: {}".format(exc))
        return

    def render_body(show_heading: bool = True) -> None:
        if show_heading:
            st.markdown("### 31-dataset held-out validation · supplied evidence")

        st.caption(
            "These aggregate results are computed only from the three 31-dataset evidence files supplied for this UI. "
            "They are displayed read-only, kept separate from the interactive recommendation, and are not confidence "
            "scores for the current recommendation."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Held-out datasets", snap["datasets"])
        c2.metric("Preference cases", "{:,}".format(snap["preference_cases"]))
        c3.metric("Top-1 oracle agreement", _pct(snap["top1"]))
        c4.metric("Top-3 oracle coverage", _pct(snap["top3"]))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Exact Top-1 matches", "{:,}".format(snap["top1_matches"]))
        c6.metric("Top-3 successes", "{:,}".format(snap["top3_successes"]))
        c7.metric("Mean normalized regret", "{:.4f}".format(snap["normalized_regret_mean"]))
        c8.metric("Mean Spearman", "{:.4f}".format(snap["spearman_mean"]))

        st.info(
            "Interpretation: Top-1/Top-3 compare the frozen recommender with an independently computed empirical "
            "oracle under the same preference vector. Regret measures utility loss when the exact oracle is missed; "
            "Spearman measures full-ranking agreement."
        )

        objective = pd.DataFrame(snap["objective_winners"])
        objective["Objective"] = objective["objective"].astype(str).map(
            {"accuracy": "Accuracy", "runtime": "Runtime", "energy": "Energy", "co2": "CO₂"}
        ).fillna(objective["objective"].astype(str))
        objective["Agreement"] = objective.apply(
            lambda r: "{} / {} ({:.2%})".format(
                int(r["matches"]), int(r["cases"]), float(r["top1"])
            ),
            axis=1,
        )

        left, right = st.columns([0.9, 1.1])
        with left:
            st.markdown("**Objective-level winner agreement**")
            st.dataframe(
                objective[["Objective", "matches", "cases", "top1"]].rename(
                    columns={"matches": "Matches", "cases": "Datasets", "top1": "Top-1 rate"}
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Top-1 rate": st.column_config.NumberColumn(format="%.2f")
                },
            )

        with right:
            fig = px.bar(
                objective,
                x="top1",
                y="Objective",
                orientation="h",
                text="Agreement",
                range_x=[0, 1],
                labels={"top1": "Exact winner agreement", "Objective": ""},
                color="Objective",
                color_discrete_sequence=["#2563eb", "#7c3aed", "#059669", "#0f766e"],
            )
            fig.update_traces(textposition="inside")
            fig.update_layout(
                height=310,
                showlegend=False,
                margin=dict(l=20, r=20, t=15, b=40),
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="phase18_objective_winner_agreement",
            )

        cohort = pd.DataFrame(snap["cohort_summary"])
        if not cohort.empty:
            display = cohort.rename(
                columns={
                    "cohort": "Cohort",
                    "cases": "Cases",
                    "datasets": "Datasets",
                    "top1": "Top-1",
                    "top3": "Top-3",
                    "normalized_regret": "Mean regret",
                    "spearman": "Mean Spearman",
                }
            )
            st.markdown("**Preference-case results by evaluation cohort**")
            st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("**Per-dataset summary (31 datasets)**")
        table = summary.rename(
            columns={
                "dataset_id": "Dataset",
                "cohort": "Cohort",
                "cases": "Cases",
                "top1": "Top-1",
                "top3": "Top-3",
                "normalized_regret_mean": "Mean regret",
                "spearman_mean": "Mean Spearman",
            }
        )
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            height=360,
        )

        st.markdown("**Evidence provenance**")
        provenance = pd.DataFrame(
            [
                {
                    "File": PREFERENCE_FILE.name,
                    "Rows": len(pref),
                    "SHA-256": _sha256(PREFERENCE_FILE),
                },
                {
                    "File": WINNER_FILE.name,
                    "Rows": len(winners),
                    "SHA-256": _sha256(WINNER_FILE),
                },
                {
                    "File": SUMMARY_FILE.name,
                    "Rows": len(summary),
                    "SHA-256": _sha256(SUMMARY_FILE),
                },
            ]
        )
        st.dataframe(
            provenance,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "The objective-winner file partitions the 31 datasets into source cohorts {}. "
            "Aggregates above are keyed by dataset_id and include all 31 unique datasets.".format(
                ", ".join(snap["winner_source_cohorts"])
            )
        )

    if collapsible:
        with st.expander(
            "31-dataset held-out validation · supplied evidence",
            expanded=expanded,
        ):
            render_body(show_heading=False)
        return

    if embedded:
        with st.container():
            render_body(show_heading=True)
        return

    render_body(show_heading=True)
