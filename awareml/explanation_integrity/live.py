from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from .schemas import EvidenceCase


def _result_dicts(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for result in state.get("run_results") or []:
        if hasattr(result, "to_dict"):
            rows.append(dict(result.to_dict()))
        elif isinstance(result, Mapping):
            rows.append(dict(result))
    return rows


def _ranking_records(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    ranked = state.get("v2_candidates")
    if isinstance(ranked, pd.DataFrame):
        return ranked.to_dict(orient="records")
    if isinstance(ranked, list):
        return [dict(row) for row in ranked if isinstance(row, Mapping)]

    ranking = state.get("ranking")
    if isinstance(ranking, pd.DataFrame):
        return ranking.to_dict(orient="records")
    if isinstance(ranking, list):
        return [dict(row) for row in ranking if isinstance(row, Mapping)]
    return []


def build_live_stage_b_case(state: Mapping[str, Any]) -> Optional[EvidenceCase]:
    ranking = _ranking_records(state)
    if not ranking:
        return None

    ranking = sorted(
        ranking,
        key=lambda row: (
            float(row.get("rank")) if row.get("rank") is not None else 1e9,
            -float(row.get("utility")) if row.get("utility") is not None else 0.0,
        ),
    )
    candidates = {}
    for row in ranking:
        framework = str(row.get("framework"))
        candidates[framework] = {
            key: row.get(key)
            for key in [
                "rank",
                "utility",
                "accuracy",
                "runtime",
                "energy",
                "co2",
                "accuracy_lower",
                "accuracy_upper",
                "runtime_lower",
                "runtime_upper",
                "energy_lower",
                "energy_upper",
                "co2_lower",
                "co2_upper",
            ]
            if key in row
        }

    top = str(ranking[0].get("framework"))
    return EvidenceCase(
        case_id="LIVE_STAGE_B",
        dataset_id=str(state.get("dataset_name") or "active_dataset"),
        source_stage="B",
        source_name="Stage B · setup/recommendation explanation",
        prompt=(
            "Explain the current pre-run recommendation using the predicted "
            "framework evidence and ranking."
        ),
        evidence={
            "recommendation": {
                "top_framework": top,
                "ranking_mode": state.get("ranking_mode"),
                "weights": state.get("preference_weights"),
            },
            "candidates": candidates,
            "ranking": ranking,
        },
        metadata={
            "exploratory_live_probe": True,
            "journal_evidence": False,
            "probe_scope": "exploratory_live_dataset_only",
        },
    )


def _focus_result(state: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    rows = _result_dicts(state)
    if not rows:
        return None
    selected = state.get("selected_framework")
    if selected:
        for row in rows:
            if str(row.get("framework")) == str(selected):
                return row
    return rows[0]


def build_live_stage_e_case(state: Mapping[str, Any]) -> Optional[EvidenceCase]:
    rows = _result_dicts(state)
    if not rows:
        return None

    frameworks = {}
    for row in rows:
        framework = str(row.get("framework"))
        frameworks[framework] = {
            "fairness": dict(row.get("fairness") or {})
        }

    focus = state.get("selected_framework")
    if focus not in frameworks:
        focus = next(iter(frameworks), None)

    return EvidenceCase(
        case_id="LIVE_STAGE_E",
        dataset_id=str(state.get("dataset_name") or "active_dataset"),
        source_stage="E",
        source_name="Stage E · fairness explanation",
        prompt=(
            "Explain the current fairness evidence for {}. Keep unavailable "
            "values as N/A and include calibration fairness where available."
        ).format(focus or "the selected framework"),
        evidence={"frameworks": frameworks},
        metadata={
            "focus_framework": focus,
            "exploratory_live_probe": True,
            "journal_evidence": False,
            "probe_scope": "exploratory_live_dataset_only",
        },
    )


def build_live_stage_xai_case(state: Mapping[str, Any]) -> Optional[EvidenceCase]:
    row = _focus_result(state)
    if not row:
        return None
    exp = dict(row.get("explainability") or {})
    if not exp:
        return None

    feature_importance = exp.get("feature_importance")
    shap_values = exp.get("shap_values")

    if not isinstance(shap_values, Mapping) and isinstance(feature_importance, Mapping):
        # Existing AwareML adapters often expose one derived feature-importance
        # map rather than literal per-instance SHAP values. Do not relabel it as
        # SHAP; the UI will disclose the method/status.
        shap_values = None

    top_features = exp.get("top_features")
    if not top_features and isinstance(shap_values, Mapping):
        top_features = [
            {"feature": str(name), "shap_value": value}
            for name, value in sorted(
                shap_values.items(),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )[:10]
        ]

    evidence = {
        "framework": row.get("framework"),
        "explainability": {
            "status": exp.get("status"),
            "method": exp.get("method"),
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "top_features": top_features,
            "fidelity": exp.get("fidelity"),
            "stability": exp.get("stability"),
            "consistency": exp.get("consistency"),
        },
    }
    return EvidenceCase(
        case_id="LIVE_STAGE_F_XAI",
        dataset_id=str(state.get("dataset_name") or "active_dataset"),
        source_stage="F_XAI",
        source_name="Stage F · SHAP/XAI explanation",
        prompt=(
            "Explain the current feature-attribution evidence for {}. "
            "Only call values SHAP when the structured evidence identifies "
            "them as SHAP."
        ).format(row.get("framework")),
        evidence=evidence,
        metadata={
            "framework": row.get("framework"),
            "exploratory_live_probe": True,
            "journal_evidence": False,
            "probe_scope": "exploratory_live_dataset_only",
        },
    )


def build_live_stage_chat_case(state: Mapping[str, Any]) -> Optional[EvidenceCase]:
    rows = _result_dicts(state)
    if not rows:
        return None

    frameworks = {}
    for row in rows:
        framework = str(row.get("framework"))
        frameworks[framework] = {
            "accuracy": row.get("accuracy"),
            "f1_macro": row.get("f1_macro"),
            "runtime_sec": row.get("runtime_sec"),
            "energy_kwh": row.get("energy_kwh"),
            "co2_kg": row.get("co2_kg"),
            "fairness": row.get("fairness") or {},
            "explainability": row.get("explainability") or {},
        }

    ranking = _ranking_records(state)
    return EvidenceCase(
        case_id="LIVE_STAGE_F_CHAT",
        dataset_id=str(state.get("dataset_name") or "active_dataset"),
        source_stage="F_CHAT",
        source_name="Stage F · conversational answer",
        prompt=(
            "Answer this concrete question using only the supplied active-run "
            "evidence: Which framework is currently ranked first, and which "
            "reported accuracy, runtime, energy and CO2 values support that "
            "ranking? If the ranking evidence is unavailable or incomplete, "
            "say so explicitly rather than asking the user for another question."
        ),
        evidence={
            "frameworks": frameworks,
            "ranking": ranking,
        },
        metadata={
            "exploratory_live_probe": True,
            "journal_evidence": False,
            "probe_scope": "exploratory_live_dataset_only",
        },
    )


def build_live_cases(state: Mapping[str, Any]) -> List[EvidenceCase]:
    builders = [
        build_live_stage_b_case,
        build_live_stage_e_case,
        build_live_stage_xai_case,
        build_live_stage_chat_case,
    ]
    cases = []
    for builder in builders:
        try:
            case = builder(state)
        except Exception:
            case = None
        if case is not None:
            cases.append(case)
    return cases
