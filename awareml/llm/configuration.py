from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .schemas import (
    CopilotConfiguration,
    GoalInterpretation,
)


FRAMEWORK_DEFAULTS = {
    "AutoStreamML": {
        "algorithm": (
            "adaptive_population_ensemble"
        ),
        "parameters": {
            "exploration_window": 500,
            "budget": 8,
            "ensemble_size": 3,
        },
    },
    "AutoClass": {
        "algorithm": (
            "adaptive_population_search"
        ),
        "parameters": {
            "exploration_window": 500,
            "population_size": 8,
        },
    },
    "EvoAutoML": {
        "algorithm": (
            "EvolutionaryBaggingClassifier"
        ),
        "parameters": {
            "population_size": 10,
            "sampling_rate": 500,
        },
    },
    "OAML": {
        "algorithm": (
            "online_streaming_mode"
        ),
        "parameters": {
            "mode": "online",
            "grace_period": 100,
        },
    },
    "ChaCha": {
        "algorithm": (
            "FLAML_AutoVW"
        ),
        "parameters": {
            "multiclass_strategy": (
                "awareml_one_vs_rest_native_autovw"
            ),
            "exploration": 1.2,
        },
    },
}


def synthesize_configuration(
    interpretation: GoalInterpretation,
    ranked: pd.DataFrame,
    sensitive_attribute: Optional[
        str
    ] = None,
) -> Tuple[
    CopilotConfiguration,
    List[str],
]:
    if ranked is None or ranked.empty:
        raise ValueError(
            "Empirical ML recommender ranking is required."
        )

    framework = str(
        ranked.iloc[0][
            "framework"
        ]
    )
    defaults = (
        FRAMEWORK_DEFAULTS.get(
            framework
        )
    )
    if defaults is None:
        raise ValueError(
            "Unsupported framework: {}".format(
                framework
            )
        )

    warnings = []

    if (
        interpretation
        .drift_sensitivity
        == "high"
    ):
        window_size = 500
    elif (
        interpretation
        .drift_sensitivity
        == "low"
    ):
        window_size = 1500
    else:
        window_size = 1000

    time_budget = (
        float(
            interpretation
            .max_runtime_sec
        )
        if (
            interpretation
            .max_runtime_sec
            is not None
        )
        else 60.0
    )

    fairness_status = "disabled"
    if (
        interpretation
        .fairness_required
    ):
        if sensitive_attribute:
            fairness_status = "enabled"
        else:
            fairness_status = (
                "requires_sensitive_attribute"
            )
            warnings.append(
                "Fairness was requested but no sensitive attribute "
                "is currently selected. The user must review this before run."
            )

    weights = (
        interpretation
        .primary_weights
        .normalized_dict()
    )

    config = CopilotConfiguration(
        framework=framework,
        algorithm=defaults[
            "algorithm"
        ],
        framework_parameters=dict(
            defaults["parameters"]
        ),
        window_size=window_size,
        time_budget_sec=time_budget,
        drift={
            "monitoring_enabled": True,
            "detector": "ADWIN",
            "sensitivity": (
                interpretation
                .drift_sensitivity
            ),
            "recovery_tracking": True,
        },
        fairness={
            "requested": (
                interpretation
                .fairness_required
            ),
            "status": fairness_status,
            "sensitive_attribute": (
                sensitive_attribute
            ),
            "metric": (
                interpretation
                .fairness_metric
                or "composite"
            ),
            "audit_only": True,
        },
        explainability={
            "level": (
                interpretation
                .explainability_level
            ),
            "method": "auto",
            "fallback_order": [
                "SHAP",
                "LIME",
                "permutation",
            ],
            "unsupported_is_not_zero": True,
        },
        sustainability={
            "enabled": (
                weights["energy"] > 0
                or weights["co2"] > 0
            ),
            "track_energy": True,
            "track_co2": True,
        },
        primary_weights=(
            interpretation
            .primary_weights
        ),
    )

    return config, warnings
