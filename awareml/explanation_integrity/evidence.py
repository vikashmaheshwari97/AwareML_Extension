from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


METRIC_ALIASES = {
    "accuracy": (
        "accuracy",
        "acc",
    ),
    "runtime": (
        "runtime",
        "runtime_sec",
        "time",
        "latency",
    ),
    "energy": (
        "energy",
        "energy_kwh",
    ),
    "co2": (
        "co2",
        "co2_kg",
        "carbon",
        "carbon_emissions",
    ),
    "dp": (
        "dp",
        "dp_diff",
        "demographic_parity",
        "demographic_parity_diff",
        "statistical_parity_difference",
        "spd",
    ),
    "eo": (
        "eo",
        "equal_opportunity",
        "equal_opportunity_diff",
    ),
    "eodds": (
        "eodds",
        "equalized_odds",
        "equalized_odds_gap",
        "equalised_odds",
    ),
    "brier_gap": (
        "brier",
        "brier_gap",
        "group_brier_score_gap",
        "group_brier_gap",
    ),
    "ece_gap": (
        "ece",
        "ece_gap",
        "group_ece_gap",
    ),
    "shap": (
        "shap",
        "shap_value",
        "shap_values",
    ),
    "feature_importance": (
        "feature_importance",
        "importance",
    ),
    "rank": (
        "rank",
        "ranking",
    ),
    "utility": (
        "utility",
        "preference_utility",
    ),
    "ranking_mode": (
        "ranking_mode",
    ),
}

LOWER_IS_BETTER = {
    "runtime",
    "energy",
    "co2",
    "dp",
    "eo",
    "eodds",
    "brier_gap",
    "ece_gap",
}


def plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    return value


def sanitize(value: Any) -> Any:
    value = plain(value)
    if isinstance(value, Mapping):
        return {
            str(key): sanitize(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(child) for child in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def flatten_evidence(
    evidence: Mapping[str, Any],
    prefix: str = "evidence",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, "{}.{}".format(path, key))
            return
        if isinstance(value, list):
            out[path] = value
            # Flatten small dictionaries inside lists so ranking rows stay addressable.
            for index, child in enumerate(value):
                if isinstance(child, Mapping):
                    visit(child, "{}.{}".format(path, index))
            return
        out[path] = value

    visit(sanitize(dict(evidence)), prefix)
    return out


def normalize_key(key: str) -> str:
    key = str(key).strip()
    if key.startswith("[") and key.endswith("]"):
        key = key[1:-1].strip()
    if key.startswith("frameworks."):
        return "evidence." + key
    if key.startswith("ranking"):
        return "evidence." + key
    if key.startswith("evidence."):
        return key
    return "evidence." + key



def resolve_flat_key(
    flat: Mapping[str, Any],
    key: str,
) -> Optional[str]:
    """Resolve evidence citations across existing AwareML stage prefixes.

    Phase-8 rationale citations use evidence.before.*, while newer live probes
    use evidence.* and GroundedChat often emits frameworks.*. The verifier
    treats these as citation-format aliases only; the underlying fact must still
    exist in the supplied structured evidence.
    """
    normalized = normalize_key(key)
    if normalized in flat:
        return normalized

    parts = normalized.split(".")
    if len(parts) > 2 and parts[0] == "evidence" and parts[1] in {
        "before",
        "during",
        "after",
    }:
        stripped = ".".join([parts[0]] + parts[2:])
        if stripped in flat:
            return stripped

    # Conversely, if a case stores an explicit stage wrapper, try each one.
    if len(parts) > 1 and parts[0] == "evidence":
        tail = ".".join(parts[1:])
        for stage in ("before", "during", "after"):
            candidate = "evidence.{}.{}".format(stage, tail)
            if candidate in flat:
                return candidate
    return None

def canonical_metric(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    cleaned = (
        str(token)
        .lower()
        .replace("co₂", "co2")
        .replace("coâ‚‚", "co2")
        .replace("-", "_")
        .replace(" ", "_")
        .strip("_")
    )
    for metric, aliases in METRIC_ALIASES.items():
        if cleaned == metric or cleaned in aliases:
            return metric
    if "demographic" in cleaned and "parity" in cleaned:
        return "dp"
    if "statistical" in cleaned and "parity" in cleaned:
        return "dp"
    if "equal" in cleaned and "opportunity" in cleaned:
        return "eo"
    if "equal" in cleaned and "odds" in cleaned:
        return "eodds"
    if "brier" in cleaned:
        return "brier_gap"
    if "ece" in cleaned or "calibration" in cleaned:
        return "ece_gap"
    if "runtime" in cleaned or "latency" in cleaned:
        return "runtime"
    if "energy" in cleaned:
        return "energy"
    if "carbon" in cleaned or "co2" in cleaned:
        return "co2"
    if "accuracy" in cleaned:
        return "accuracy"
    if "shap" in cleaned:
        return "shap"
    if "rank" in cleaned:
        return "rank"
    return None


def key_metric(key: str) -> Optional[str]:
    lowered = normalize_key(key).lower()
    segments = [
        segment
        for segment in lowered.replace("[", ".").replace("]", ".").split(".")
        if segment
    ]

    # Prefer the most specific/right-most path segment. This is essential for
    # keys such as evidence.ranking.0.utility: the container name "ranking"
    # must not cause that key to be misclassified as a rank metric.
    for segment in reversed(segments):
        resolved = canonical_metric(segment)
        if resolved is not None:
            return resolved

    # Then match aliases as complete path segments. This prevents short aliases
    # such as "acc" from incorrectly matching feature names like account_age.
    for metric, aliases in METRIC_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if alias in segments:
                return metric

    return canonical_metric(lowered)


def key_entity(key: str) -> Optional[str]:
    key = normalize_key(key)
    parts = key.split(".")
    if "frameworks" in parts:
        idx = parts.index("frameworks")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "candidates" in parts:
        idx = parts.index("candidates")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def evidence_numeric_candidates(
    flat: Mapping[str, Any],
    metric: str,
    entity: Optional[str] = None,
    feature: Optional[str] = None,
) -> List[Tuple[str, float]]:
    metric = canonical_metric(metric) or str(metric)
    candidates = []

    for key, value in flat.items():
        if key_metric(key) != metric:
            continue

        if entity:
            key_lower = key.lower()
            entity_lower = str(entity).lower()

            # Direct framework/candidate paths carry the entity in the key.
            direct_match = entity_lower in key_lower

            # Ranking rows carry the framework name in a sibling field:
            # evidence.ranking.<i>.framework. Resolve that association here.
            ranking_match = False
            parts = key.split(".")
            if (
                len(parts) >= 4
                and parts[0] == "evidence"
                and parts[1] == "ranking"
                and parts[2].isdigit()
            ):
                sibling = "evidence.ranking.{}.framework".format(parts[2])
                ranking_match = (
                    str(flat.get(sibling, "")).lower() == entity_lower
                )

            if not (direct_match or ranking_match):
                continue

        if feature:
            key_lower = key.lower()
            feature_lower = str(feature).lower()
            direct_feature_match = feature_lower in key_lower

            importance_list_match = False
            parts = key.split(".")
            if (
                len(parts) >= 5
                and parts[0] == "evidence"
                and "feature_importance" in parts
            ):
                try:
                    idx = parts.index("feature_importance")
                except ValueError:
                    idx = -1
                if (
                    idx >= 0
                    and idx + 1 < len(parts)
                    and parts[idx + 1].isdigit()
                ):
                    prefix = ".".join(parts[: idx + 2])
                    sibling_feature = flat.get(prefix + ".feature")
                    sibling_name = flat.get(prefix + ".name")
                    importance_list_match = (
                        str(sibling_feature or sibling_name or "").lower()
                        == feature_lower
                    )

            if not (direct_feature_match or importance_list_match):
                continue

        try:
            numeric = float(value)
        except Exception:
            continue
        if not np.isfinite(numeric):
            continue

        candidates.append((key, numeric))

    # Point-estimate claims must prefer canonical point fields over
    # confidence/uncertainty interval fields such as accuracy_lower,
    # accuracy_upper, runtime_lower, runtime_upper, etc.
    #
    # Without this preference a sentence such as "accuracy is 0.779" can
    # produce three equally plausible structured candidates:
    #   ...accuracy
    #   ...accuracy_lower
    #   ...accuracy_upper
    # which incorrectly turns a correct point claim into "unsupported".
    exact_aliases = {
        str(alias).lower()
        for alias in METRIC_ALIASES.get(metric, ())
    }
    exact_aliases.add(str(metric).lower())

    exact_point_candidates = []
    for key, numeric in candidates:
        terminal = normalize_key(key).split(".")[-1].lower()
        if terminal in exact_aliases:
            exact_point_candidates.append((key, numeric))

    if exact_point_candidates:
        return exact_point_candidates

    return candidates


def ranking_rows(evidence: Mapping[str, Any]) -> List[Dict[str, Any]]:
    evidence = sanitize(evidence)
    rows = evidence.get("ranking") if isinstance(evidence, Mapping) else None
    if isinstance(rows, list):
        clean = [dict(row) for row in rows if isinstance(row, Mapping)]
        if clean:
            return clean

    # Common after-run/recommender representation.
    frameworks = evidence.get("frameworks") if isinstance(evidence, Mapping) else None
    if isinstance(frameworks, Mapping):
        rows = []
        for name, payload in frameworks.items():
            if not isinstance(payload, Mapping):
                continue
            rank = payload.get("rank")
            utility = payload.get("utility")
            if rank is not None or utility is not None:
                rows.append({
                    "framework": str(name),
                    "rank": rank,
                    "utility": utility,
                })
        if rows:
            return rows
    return []


def top_ranked_framework(evidence: Mapping[str, Any]) -> Optional[str]:
    # Direct recommendation field first.
    recommendation = evidence.get("recommendation") if isinstance(evidence, Mapping) else None
    if isinstance(recommendation, Mapping):
        top = recommendation.get("top_framework")
        if top:
            return str(top)

    rows = ranking_rows(evidence)
    if not rows:
        return None

    def sort_key(row: Mapping[str, Any]):
        rank = row.get("rank")
        utility = row.get("utility")
        try:
            rank_value = float(rank)
        except Exception:
            rank_value = 1e9
        try:
            utility_value = float(utility)
        except Exception:
            utility_value = -1e9
        return (rank_value, -utility_value)

    row = sorted(rows, key=sort_key)[0]
    framework = row.get("framework")
    return str(framework) if framework is not None else None


def top_feature(evidence: Mapping[str, Any]) -> Optional[str]:
    # Preferred explicit top_features representation.
    exp = evidence.get("explainability") if isinstance(evidence, Mapping) else None
    if isinstance(exp, Mapping):
        top = exp.get("top_features")
        if isinstance(top, list) and top:
            first = top[0]
            if isinstance(first, Mapping):
                name = first.get("feature") or first.get("name")
                if name:
                    return str(name)
            if isinstance(first, (list, tuple)) and first:
                return str(first[0])
            if isinstance(first, str):
                return first

        shap = exp.get("shap_values")
        if isinstance(shap, Mapping) and shap:
            ranked = sorted(
                shap.items(),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )
            return str(ranked[0][0])

        importance = exp.get("feature_importance")
        if isinstance(importance, Mapping) and importance:
            ranked = sorted(
                importance.items(),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )
            return str(ranked[0][0])

        if isinstance(importance, list) and importance:
            rows = []
            for item in importance:
                if not isinstance(item, Mapping):
                    continue
                name = item.get("feature") or item.get("name")
                value = item.get("importance")
                if name is None or value is None:
                    continue
                try:
                    numeric = float(value)
                except Exception:
                    continue
                rows.append((str(name), numeric))
            if rows:
                rows.sort(key=lambda item: abs(item[1]), reverse=True)
                return rows[0][0]

    shap = evidence.get("shap_values") if isinstance(evidence, Mapping) else None
    if isinstance(shap, Mapping) and shap:
        ranked = sorted(
            shap.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
        return str(ranked[0][0])
    return None


def numeric_close(
    observed: float,
    expected: float,
    abs_tol: float = 1e-4,
    rel_tol: float = 0.01,
) -> bool:
    try:
        observed = float(observed)
        expected = float(expected)
    except Exception:
        return False
    tolerance = max(float(abs_tol), abs(expected) * float(rel_tol))
    return abs(observed - expected) <= tolerance
