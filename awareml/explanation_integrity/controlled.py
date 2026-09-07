from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .faithfulness import ControlledExplanationGenerator
from .schemas import EvidenceCase, Intervention


FRAMEWORKS = [
    "AutoStreamML",
    "AutoClass",
    "EvoAutoML",
    "OAML",
    "ChaCha",
]

CONTROLLED_CONTEXTS = [
    {
        "dataset_id": "controlled_stream_alpha",
        "focus_framework": "AutoClass",
        "top_framework": "AutoClass",
        "top_feature": "age",
        "second_feature": "hours_per_week",
        "base": 0.00,
    },
    {
        "dataset_id": "controlled_stream_beta",
        "focus_framework": "OAML",
        "top_framework": "OAML",
        "top_feature": "sensor_load",
        "second_feature": "temperature",
        "base": 0.02,
    },
    {
        "dataset_id": "controlled_stream_gamma",
        "focus_framework": "EvoAutoML",
        "top_framework": "EvoAutoML",
        "top_feature": "transaction_velocity",
        "second_feature": "account_age",
        "base": 0.04,
    },
    {
        "dataset_id": "controlled_stream_delta",
        "focus_framework": "ChaCha",
        "top_framework": "ChaCha",
        "top_feature": "traffic_density",
        "second_feature": "weather_index",
        "base": 0.06,
    },
    {
        "dataset_id": "controlled_stream_epsilon",
        "focus_framework": "AutoClass",
        "top_framework": "AutoClass",
        "top_feature": "signal_quality",
        "second_feature": "device_age",
        "base": 0.08,
    },
    {
        "dataset_id": "controlled_stream_zeta",
        "focus_framework": "OAML",
        "top_framework": "OAML",
        "top_feature": "credit_utilization",
        "second_feature": "income_ratio",
        "base": 0.10,
    },
]


def _framework_rows(context_index: int, top_framework: str):
    rows = []
    # Deterministic but non-identical values across contexts.
    for idx, framework in enumerate(FRAMEWORKS):
        accuracy = 0.68 + 0.025 * idx + 0.004 * context_index
        runtime = 18.0 - 1.7 * idx + 0.35 * context_index
        energy = 0.0018 - 0.00016 * idx + 0.00003 * context_index
        co2 = energy * 0.4167

        rows.append({
            "framework": framework,
            "accuracy": round(accuracy, 6),
            "runtime": round(runtime, 6),
            "runtime_sec": round(runtime, 6),
            "energy": round(energy, 8),
            "energy_kwh": round(energy, 8),
            "co2": round(co2, 8),
            "co2_kg": round(co2, 8),
        })

    # Give the intended winner a modest accuracy advantage and good efficiency.
    top_idx = FRAMEWORKS.index(top_framework)
    rows[top_idx]["accuracy"] = round(0.86 + 0.006 * context_index, 6)
    rows[top_idx]["runtime"] = round(8.4 + 0.25 * context_index, 6)
    rows[top_idx]["runtime_sec"] = rows[top_idx]["runtime"]
    rows[top_idx]["energy"] = round(0.00092 + 0.00002 * context_index, 8)
    rows[top_idx]["energy_kwh"] = rows[top_idx]["energy"]
    rows[top_idx]["co2"] = round(rows[top_idx]["energy"] * 0.4167, 8)
    rows[top_idx]["co2_kg"] = rows[top_idx]["co2"]

    # Rank by a fixed controlled utility.
    for row in rows:
        row["utility"] = (
            0.60 * row["accuracy"]
            - 0.015 * row["runtime"]
            - 25.0 * row["energy"]
        )
    ranked = sorted(rows, key=lambda row: row["utility"], reverse=True)

    # Force intended winner first if tiny numerical differences compete.
    ranked.sort(
        key=lambda row: (
            0 if row["framework"] == top_framework else 1,
            -row["utility"],
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["utility"] = round(float(row["utility"]), 8)
    return ranked



def _recompute_point_utility(
    accuracy: float,
    runtime: float,
    energy: float,
) -> float:
    """Controlled-fixture utility used by _framework_rows."""
    return round(
        0.60 * float(accuracy)
        - 0.015 * float(runtime)
        - 25.0 * float(energy),
        8,
    )


def _decision_flip_linked_changes(
    ranking: List[Dict[str, Any]],
    changed_framework: str,
    changed_accuracy: float,
    candidate_prefix: str,
    ranking_prefix: str = "evidence.ranking",
) -> Tuple[Dict[str, Any], str]:
    """Recompute the derived utility/rank after an accuracy perturbation.

    This keeps counterfactual evidence coherent: if a primitive objective changes,
    its derived utility and ranking are updated too. The original benchmark only
    changed accuracy while leaving utility/rank frozen, which made the intervention
    inappropriate for decision-faithfulness testing.
    """
    rows = [dict(row) for row in ranking]

    for row in rows:
        if str(row["framework"]) == str(changed_framework):
            row["accuracy"] = float(changed_accuracy)
            row["utility"] = _recompute_point_utility(
                row["accuracy"],
                row["runtime"],
                row["energy"],
            )

    ordered = sorted(
        rows,
        key=lambda row: float(row["utility"]),
        reverse=True,
    )
    ranks = {
        str(row["framework"]): rank
        for rank, row in enumerate(ordered, start=1)
    }
    new_top = str(ordered[0]["framework"])

    linked: Dict[str, Any] = {}

    for idx, original_row in enumerate(ranking):
        framework = str(original_row["framework"])
        updated = next(
            row for row in rows
            if str(row["framework"]) == framework
        )

        linked[
            "{}.{}.rank".format(ranking_prefix, idx)
        ] = int(ranks[framework])
        linked[
            "{}.{}.utility".format(ranking_prefix, idx)
        ] = float(updated["utility"])

        if framework == str(changed_framework):
            linked[
                "{}.{}.accuracy".format(ranking_prefix, idx)
            ] = float(changed_accuracy)

        linked[
            "{}.{}.rank".format(candidate_prefix, framework)
        ] = int(ranks[framework])
        linked[
            "{}.{}.utility".format(candidate_prefix, framework)
        ] = float(updated["utility"])

    linked["evidence.recommendation.top_framework"] = new_top
    return linked, new_top


def _stage_b_case(index: int, context: Mapping[str, Any]) -> EvidenceCase:
    ranking = _framework_rows(index, str(context["top_framework"]))
    candidates = {
        row["framework"]: {
            "rank": row["rank"],
            "utility": row["utility"],
            "accuracy": row["accuracy"],
            "runtime": row["runtime"],
            "energy": row["energy"],
            "co2": row["co2"],
        }
        for row in ranking
    }
    top = ranking[0]["framework"]

    focus_key = "evidence.candidates.{}.accuracy".format(top)
    original = candidates[top]["accuracy"]
    counterfactual = round(max(0.50, float(original) - 0.17), 6)

    linked_changes, new_top = _decision_flip_linked_changes(
        ranking=ranking,
        changed_framework=top,
        changed_accuracy=counterfactual,
        candidate_prefix="evidence.candidates",
    )

    return EvidenceCase(
        case_id="P15_{:02d}_B".format(index + 1),
        dataset_id=str(context["dataset_id"]),
        source_stage="B",
        source_name="Stage B · setup/recommendation explanation",
        prompt=(
            "Explain why the current pre-run framework is recommended using "
            "only the supplied predicted evidence."
        ),
        evidence={
            "recommendation": {
                "top_framework": top,
                "ranking_mode": "point",
            },
            "candidates": candidates,
            "ranking": ranking,
        },
        metadata={
            "controlled_fixture": True,
            "focus_framework": top,
            "intervention": {
                "kind": "decision_flip_accuracy_counterfactual",
                "evidence_key": focus_key,
                "counterfactual_value": counterfactual,
                "relevant": True,
                "linked_changes": linked_changes,
                "expected_new_decision": new_top,
            },
        },
    )


def _stage_e_case(index: int, context: Mapping[str, Any]) -> EvidenceCase:
    ranking = _framework_rows(index, str(context["top_framework"]))
    frameworks = {}
    for idx, row in enumerate(ranking):
        offset = 0.008 * idx + float(context["base"])
        frameworks[row["framework"]] = {
            "fairness": {
                "dp_diff": round(0.035 + offset, 6),
                "equal_opportunity_diff": round(0.028 + offset * 0.8, 6),
                "equalized_odds_gap": round(0.044 + offset * 0.7, 6),
                "group_brier_score_gap": round(0.012 + offset * 0.20, 6),
                "group_ece_gap": round(0.018 + offset * 0.25, 6),
            }
        }

    focus = str(context["focus_framework"])
    original = frameworks[focus]["fairness"]["dp_diff"]
    counterfactual = round(min(0.35, float(original) + 0.12), 6)

    return EvidenceCase(
        case_id="P15_{:02d}_E".format(index + 1),
        dataset_id=str(context["dataset_id"]),
        source_stage="E",
        source_name="Stage E · fairness explanation",
        prompt=(
            "Explain the fairness disparities for the focus framework, "
            "including calibration fairness where available."
        ),
        evidence={
            "frameworks": frameworks,
        },
        metadata={
            "controlled_fixture": True,
            "focus_framework": focus,
            "intervention": {
                "kind": "fairness_counterfactual",
                "evidence_key": (
                    "evidence.frameworks.{}.fairness.dp_diff".format(focus)
                ),
                "counterfactual_value": counterfactual,
                "relevant": True,
            },
        },
    )


def _stage_xai_case(index: int, context: Mapping[str, Any]) -> EvidenceCase:
    top = str(context["top_feature"])
    second = str(context["second_feature"])
    third = "baseline_feature_{}".format(index + 1)
    shap_values = {
        top: round(0.31 + 0.012 * index, 6),
        second: round(-0.19 - 0.008 * index, 6),
        third: round(0.07 + 0.003 * index, 6),
    }
    top_features = [
        {"feature": name, "shap_value": value}
        for name, value in sorted(
            shap_values.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
    ]

    # Reduce the old top feature below the second feature's magnitude.
    counterfactual = round(0.05 + 0.002 * index, 6)

    return EvidenceCase(
        case_id="P15_{:02d}_F_XAI".format(index + 1),
        dataset_id=str(context["dataset_id"]),
        source_stage="F_XAI",
        source_name="Stage F · SHAP/XAI explanation",
        prompt=(
            "Explain which feature is most influential and report the "
            "supporting SHAP values."
        ),
        evidence={
            "explainability": {
                "method": "SHAP",
                "shap_values": shap_values,
                "top_features": top_features,
                "top_feature": top,
            }
        },
        metadata={
            "controlled_fixture": True,
            "focus_feature": top,
            "intervention": {
                "kind": "shap_counterfactual",
                "evidence_key": (
                    "evidence.explainability.shap_values.{}".format(top)
                ),
                "counterfactual_value": counterfactual,
                "relevant": True,
                "linked_changes": {
                    "evidence.explainability.top_feature": second,
                    "evidence.explainability.top_features.0.feature": second,
                    "evidence.explainability.top_features.0.shap_value": shap_values[second],
                    "evidence.explainability.top_features.1.feature": top,
                    "evidence.explainability.top_features.1.shap_value": counterfactual,
                },
                "expected_new_decision": second,
            },
        },
    )


def _stage_chat_case(index: int, context: Mapping[str, Any]) -> EvidenceCase:
    ranking = _framework_rows(index, str(context["top_framework"]))
    frameworks = {
        row["framework"]: {
            "accuracy": row["accuracy"],
            "runtime_sec": row["runtime_sec"],
            "energy_kwh": row["energy_kwh"],
            "co2_kg": row["co2_kg"],
        }
        for row in ranking
    }

    top = str(ranking[0]["framework"])
    original = float(frameworks[top]["accuracy"])

    # Use the same 0.17 accuracy decrease as Stage B so the controlled utility
    # recomputation genuinely flips the first-ranked framework.
    counterfactual = round(max(0.50, original - 0.17), 6)

    # Build linked ranking changes directly from the same controlled utility.
    updated_rows = [dict(row) for row in ranking]
    for row in updated_rows:
        if str(row["framework"]) == top:
            row["accuracy"] = counterfactual
            row["utility"] = _recompute_point_utility(
                row["accuracy"],
                row["runtime"],
                row["energy"],
            )

    ordered = sorted(
        updated_rows,
        key=lambda row: float(row["utility"]),
        reverse=True,
    )
    ranks = {
        str(row["framework"]): rank
        for rank, row in enumerate(ordered, start=1)
    }
    new_top = str(ordered[0]["framework"])

    linked_changes: Dict[str, Any] = {}
    for idx, original_row in enumerate(ranking):
        framework = str(original_row["framework"])
        updated = next(
            row for row in updated_rows
            if str(row["framework"]) == framework
        )
        linked_changes[
            "evidence.ranking.{}.rank".format(idx)
        ] = int(ranks[framework])
        linked_changes[
            "evidence.ranking.{}.utility".format(idx)
        ] = float(updated["utility"])

    return EvidenceCase(
        case_id="P15_{:02d}_F_CHAT".format(index + 1),
        dataset_id=str(context["dataset_id"]),
        source_stage="F_CHAT",
        source_name="Stage F · conversational answer",
        prompt=(
            "Which framework is ranked first, and what measured/predicted "
            "evidence supports that answer?"
        ),
        evidence={
            "frameworks": frameworks,
            "ranking": [
                {
                    "framework": row["framework"],
                    "rank": row["rank"],
                    "utility": row["utility"],
                }
                for row in ranking
            ],
        },
        metadata={
            "controlled_fixture": True,
            "focus_framework": top,
            "intervention": {
                "kind": "decision_flip_conversational_counterfactual",
                "evidence_key": (
                    "evidence.frameworks.{}.accuracy".format(top)
                ),
                "counterfactual_value": counterfactual,
                "relevant": True,
                "linked_changes": linked_changes,
                "expected_new_decision": new_top,
            },
        },
    )


def build_controlled_cases() -> List[EvidenceCase]:
    cases: List[EvidenceCase] = []
    for index, context in enumerate(CONTROLLED_CONTEXTS):
        batch = [
            _stage_b_case(index, context),
            _stage_e_case(index, context),
            _stage_xai_case(index, context),
            _stage_chat_case(index, context),
        ]
        for case in batch:
            case.evidence["control_metadata"] = {
                "display_token": "CONTROL_A"
            }
        cases.extend(batch)
    return cases


def intervention_from_case(case: EvidenceCase) -> Intervention:
    spec = dict(case.metadata.get("intervention") or {})
    key = str(spec["evidence_key"])

    # Resolve the original value from the evidence path without depending on
    # private helper functions.
    from .evidence import flatten_evidence, normalize_key

    flat = flatten_evidence(case.evidence)
    normalized = normalize_key(key)
    original = flat[normalized]

    return Intervention(
        intervention_id="I_" + case.case_id,
        kind=str(spec.get("kind") or "counterfactual"),
        evidence_key=normalized,
        original_value=original,
        counterfactual_value=spec.get("counterfactual_value"),
        relevant=bool(spec.get("relevant", True)),
        expected_decision_change=(
            True if spec.get("expected_new_decision") is not None else None
        ),
        metadata={
            "linked_changes": dict(spec.get("linked_changes") or {}),
            "expected_new_decision": spec.get("expected_new_decision"),
        },
    )


def correct_explanation(case: EvidenceCase) -> str:
    return ControlledExplanationGenerator().generate(case)[0]


def incorrect_explanation(case: EvidenceCase, pair_index: int) -> Tuple[str, str]:
    """Create one known error without consulting an LLM."""
    text = correct_explanation(case)
    mode = pair_index % 4

    if case.source_stage == "B":
        top = str(case.evidence["recommendation"]["top_framework"])
        value = float(case.evidence["candidates"][top]["accuracy"])
        if mode == 0:
            wrong = round(max(0.0, value - 0.19), 6)
            return (
                text.replace(
                    "{} accuracy = {}".format(top, "{:.6g}".format(value)),
                    "{} accuracy = {}".format(top, "{:.6g}".format(wrong)),
                    1,
                ),
                "numeric_error",
            )
        if mode == 1:
            other = next(name for name in FRAMEWORKS if name != top)
            return (
                text.replace(
                    "{} is ranked #1".format(top),
                    "{} is ranked #1".format(other),
                    1,
                ),
                "ranking_error",
            )
        if mode == 2:
            return (
                text + " The framework is guaranteed to remain best on every future dataset.",
                "unsupported_claim",
            )
        return (
            text.replace(
                "[evidence.recommendation.top_framework]",
                "[evidence.recommendation.nonexistent]",
                1,
            ),
            "invalid_citation",
        )

    if case.source_stage == "E":
        focus = str(case.metadata["focus_framework"])
        value = float(
            case.evidence["frameworks"][focus]["fairness"]["dp_diff"]
        )
        if mode in (0, 2):
            wrong = round(value + 0.11, 6)
            return (
                text.replace(
                    "{} DP = {}".format(focus, "{:.6g}".format(value)),
                    "{} DP = {}".format(focus, "{:.6g}".format(wrong)),
                    1,
                ),
                "numeric_error",
            )
        if mode == 1:
            return (
                text + " A zero disparity on one criterion proves the classifier is fair.",
                "unsupported_claim",
            )
        return (
            text.replace(
                ".fairness.dp_diff]",
                ".fairness.not_a_metric]",
                1,
            ),
            "invalid_citation",
        )

    if case.source_stage == "F_XAI":
        exp = case.evidence["explainability"]
        top = str(exp["top_feature"])
        second = str(case.metadata.get("focus_feature") or top)
        ranked = [
            str(item["feature"]) for item in exp["top_features"]
        ]
        other = ranked[1] if len(ranked) > 1 else top
        value = float(exp["shap_values"][top])
        if mode == 0:
            wrong = round(value + 0.24, 6)
            return (
                text.replace(
                    "{} SHAP = {}".format(top, "{:.6g}".format(value)),
                    "{} SHAP = {}".format(top, "{:.6g}".format(wrong)),
                    1,
                ),
                "numeric_error",
            )
        if mode == 1:
            return (
                text.replace(
                    "{} is the top feature".format(top),
                    "{} is the top feature".format(other),
                    1,
                ),
                "feature_ranking_error",
            )
        if mode == 2:
            return (
                text + " SHAP proves that changing this feature will causally change the prediction.",
                "unsupported_claim",
            )
        return (
            text.replace(
                "[evidence.explainability.top_feature]",
                "[evidence.explainability.fake_feature]",
                1,
            ),
            "invalid_citation",
        )

    # Conversational answer.
    ranking = case.evidence["ranking"]
    top = str(ranking[0]["framework"])
    value = float(case.evidence["frameworks"][top]["accuracy"])
    if mode == 0:
        wrong = round(max(0.0, value - 0.17), 6)
        return (
            text.replace(
                "{} accuracy = {}".format(top, "{:.6g}".format(value)),
                "{} accuracy = {}".format(top, "{:.6g}".format(wrong)),
                1,
            ),
            "numeric_error",
        )
    if mode == 1:
        other = str(ranking[1]["framework"])
        return (
            text.replace(
                "{} is ranked #1".format(top),
                "{} is ranked #1".format(other),
                1,
            ),
            "ranking_error",
        )
    if mode == 2:
        return (
            text + " This ranking will necessarily hold after any concept drift.",
            "unsupported_claim",
        )
    return (
        text.replace(
            "[evidence.ranking.0.rank]",
            "[evidence.ranking.99.rank]",
            1,
        ),
        "invalid_citation",
    )
