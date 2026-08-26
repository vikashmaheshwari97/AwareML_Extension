from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


FORBIDDEN_RAW_KEYS = {
    "raw_rows",
    "rows_raw",
    "dataframe",
    "raw_dataframe",
    "dataset_rows",
    "records_raw",
    "participant_rows",
}


def _plain(value: Any):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sanitize_evidence(
    value: Any,
    max_list_items: int = 25,
):
    value = _plain(value)

    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            key_str = str(key)
            if (
                key_str.lower()
                in FORBIDDEN_RAW_KEYS
            ):
                continue
            out[key_str] = sanitize_evidence(
                child,
                max_list_items=max_list_items,
            )
        return out

    if isinstance(value, (list, tuple)):
        return [
            sanitize_evidence(
                child,
                max_list_items=max_list_items,
            )
            for child in list(value)[
                :max_list_items
            ]
        ]

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or value is None:
        return value

    return str(value)


def _flatten(
    value: Any,
    prefix: str,
    out: Dict[str, Any],
):
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = (
                "{}.{}".format(
                    prefix,
                    key,
                )
            )
            _flatten(
                child,
                next_prefix,
                out,
            )
    elif isinstance(value, list):
        # Lists remain addressable as one evidence item rather than exposing
        # arbitrary row-like indexing.
        out[prefix] = value
    else:
        out[prefix] = value


class EvidenceBundle:
    def __init__(
        self,
        stage: str,
        facts: Dict[str, Any],
    ):
        self.stage = str(stage)
        self.facts = sanitize_evidence(
            facts
        )
        self._flat = {}
        _flatten(
            self.facts,
            "evidence.{}".format(
                self.stage
            ),
            self._flat,
        )

    @property
    def valid_keys(self) -> List[str]:
        return sorted(
            self._flat.keys()
        )

    def resolve(self, key: str):
        return self._flat.get(key)

    def prompt_payload(self):
        return {
            "stage": self.stage,
            "facts": self.facts,
            "valid_evidence_keys": (
                self.valid_keys
            ),
            "privacy": {
                "raw_dataset_rows_included": False,
                "raw_participant_rows_included": False,
            },
        }


def build_before_evidence(
    interpretation,
    profile: Dict[str, Any],
    ranked: pd.DataFrame,
    ranking_meta: Dict[str, Any],
) -> EvidenceBundle:
    candidates = {}
    for _, row in ranked.iterrows():
        fw = str(row["framework"])
        candidates[fw] = {
            "rank": int(
                row.get("rank", 0)
            ),
            "utility": (
                float(row["utility"])
                if pd.notna(
                    row.get("utility")
                )
                else None
            ),
            "pareto_efficient": bool(
                row.get(
                    "pareto_efficient",
                    False,
                )
            ),
        }

        for objective in [
            "accuracy",
            "runtime",
            "energy",
            "co2",
        ]:
            value = row.get(objective)
            candidates[fw][
                objective
            ] = (
                float(value)
                if value is not None
                and pd.notna(value)
                else None
            )
            for bound in [
                "lower",
                "upper",
            ]:
                col = "{}_{}".format(
                    objective,
                    bound,
                )
                value = row.get(col)
                candidates[fw][col] = (
                    float(value)
                    if value is not None
                    and pd.notna(value)
                    else None
                )

    facts = {
        "goal_interpretation": (
            interpretation.model_dump()
            if hasattr(
                interpretation,
                "model_dump",
            )
            else interpretation
        ),
        "dataset_profile": profile,
        "recommendation": {
            "top_framework": str(
                ranked.iloc[0][
                    "framework"
                ]
            ),
            "ranking_mode": (
                ranking_meta.get(
                    "ranking_mode"
                )
            ),
            "utility_margin": (
                ranking_meta.get(
                    "utility_margin"
                )
            ),
            "weights": (
                ranking_meta.get(
                    "weights"
                )
            ),
            "warnings": (
                ranking_meta.get(
                    "warnings"
                )
            ),
        },
        "candidates": candidates,
    }
    return EvidenceBundle(
        "before",
        facts,
    )


def _point_to_dict(point):
    point = _plain(point)
    if isinstance(point, dict):
        return point
    return {}


def build_during_evidence(
    result: Any,
) -> EvidenceBundle:
    result = _plain(result)
    if not isinstance(result, dict):
        raise TypeError(
            "During-run evidence expects a result dictionary/dataclass."
        )

    points = result.get(
        "points"
    ) or []
    last_point = (
        _point_to_dict(points[-1])
        if points
        else {}
    )

    drift_events = (
        result.get("drift_events")
        or []
    )

    explainability = (
        result.get(
            "explainability"
        )
        or {}
    )
    feature_importance = (
        explainability.get(
            "feature_importance"
        )
        if isinstance(
            explainability,
            dict,
        )
        else None
    )

    top_features = []
    if isinstance(
        feature_importance,
        dict,
    ):
        top_features = sorted(
            [
                (
                    str(key),
                    float(value),
                )
                for key, value in (
                    feature_importance.items()
                )
                if value is not None
            ],
            key=lambda item: abs(
                item[1]
            ),
            reverse=True,
        )[:10]

    facts = {
        "framework": result.get(
            "framework"
        ),
        "current": {
            "sample": (
                last_point.get("sample")
                if last_point
                else result.get(
                    "samples"
                )
            ),
            "accuracy": (
                last_point.get(
                    "accuracy"
                )
                if last_point
                else result.get(
                    "accuracy"
                )
            ),
            "f1_macro": (
                last_point.get(
                    "f1_macro"
                )
                if last_point
                else result.get(
                    "f1_macro"
                )
            ),
            "rolling_accuracy": (
                last_point.get(
                    "rolling_accuracy"
                )
            ),
            "rolling_f1_macro": (
                last_point.get(
                    "rolling_f1_macro"
                )
            ),
        },
        "drift": {
            "count": len(
                drift_events
            ),
            "events": drift_events,
            "latest_event": (
                drift_events[-1]
                if drift_events
                else None
            ),
            "summary": (
                result.get(
                    "drift_summary"
                )
                or {}
            ),
        },
        "fairness": (
            result.get("fairness")
            or {}
        ),
        "explainability": {
            "status": (
                explainability.get(
                    "status"
                )
                if isinstance(
                    explainability,
                    dict,
                )
                else None
            ),
            "method": (
                explainability.get(
                    "method"
                )
                if isinstance(
                    explainability,
                    dict,
                )
                else None
            ),
            "top_features": (
                top_features
            ),
            "stability": (
                explainability.get(
                    "stability"
                )
                if isinstance(
                    explainability,
                    dict,
                )
                else None
            ),
            "fidelity": (
                explainability.get(
                    "fidelity"
                )
                if isinstance(
                    explainability,
                    dict,
                )
                else None
            ),
        },
        "sustainability": {
            "energy_kwh": (
                result.get(
                    "energy_kwh"
                )
            ),
            "co2_kg": (
                result.get(
                    "co2_kg"
                )
            ),
            "details": (
                result.get(
                    "sustainability"
                )
                or {}
            ),
        },
    }

    return EvidenceBundle(
        "during",
        facts,
    )


def build_after_evidence(
    results: Iterable[Any],
    ranking: Optional[
        Iterable[Dict[str, Any]]
    ] = None,
) -> EvidenceBundle:
    frameworks = {}

    for raw in results:
        row = _plain(raw)
        if not isinstance(row, dict):
            continue
        framework = str(
            row.get(
                "framework",
                "unknown",
            )
        )
        frameworks[
            framework
        ] = {
            "accuracy": row.get(
                "accuracy"
            ),
            "f1_macro": row.get(
                "f1_macro"
            ),
            "runtime_sec": row.get(
                "runtime_sec"
            ),
            "energy_kwh": row.get(
                "energy_kwh"
            ),
            "co2_kg": row.get(
                "co2_kg"
            ),
            "drift_summary": (
                row.get(
                    "drift_summary"
                )
                or {}
            ),
            "fairness": (
                row.get(
                    "fairness"
                )
                or {}
            ),
            "explainability": (
                row.get(
                    "explainability"
                )
                or {}
            ),
        }

    facts = {
        "frameworks": frameworks,
        "ranking": list(
            ranking or []
        ),
    }
    return EvidenceBundle(
        "after",
        facts,
    )
