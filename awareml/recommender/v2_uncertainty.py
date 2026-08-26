from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_COVERAGES = (0.80, 0.90, 0.95)


def _higher_quantile(values: np.ndarray, q: float) -> float:
    """Compatibility helper for NumPy 1.23."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute a quantile from no residuals.")
    try:
        return float(
            np.quantile(
                arr,
                q,
                method="higher",
            )
        )
    except TypeError:
        return float(
            np.quantile(
                arr,
                q,
                interpolation="higher",
            )
        )


def empirical_residual_calibration(
    predictions: pd.DataFrame,
    coverages: Iterable[float] = DEFAULT_COVERAGES,
) -> Dict[str, object]:
    """Calibrate symmetric residual intervals from dataset-level OOF errors.

    These are empirical cross-validated residual intervals. They are useful for
    uncertainty communication but are not claimed to be exact finite-sample
    conformal guarantees because the residuals come from LODO CV rather than an
    independent calibration split.
    """
    required = {"y_true", "y_pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(
            "Predictions missing columns: {}".format(missing)
        )

    residuals = np.abs(
        predictions["y_true"].to_numpy(dtype=float)
        - predictions["y_pred"].to_numpy(dtype=float)
    )
    if not np.isfinite(residuals).all():
        raise ValueError("Residuals contain non-finite values.")

    n = int(len(residuals))
    intervals = {}

    for coverage in coverages:
        coverage = float(coverage)
        if not (0.0 < coverage < 1.0):
            raise ValueError(
                "Coverage must be in (0,1), got {}.".format(
                    coverage
                )
            )

        # Conservative finite-sample quantile index. The interval is still
        # described as empirical CV calibration, not strict conformal coverage.
        adjusted = min(
            1.0,
            math.ceil((n + 1) * coverage) / float(n),
        )
        q = _higher_quantile(
            residuals,
            adjusted,
        )
        observed = float(
            np.mean(residuals <= q)
        )

        intervals[
            "{:.2f}".format(coverage)
        ] = {
            "requested_coverage": coverage,
            "adjusted_quantile_level": adjusted,
            "absolute_residual_quantile": q,
            "oof_observed_coverage": observed,
        }

    return {
        "method": "empirical_lodo_absolute_residual",
        "calibration_rows": n,
        "mean_absolute_residual": float(
            np.mean(residuals)
        ),
        "median_absolute_residual": float(
            np.median(residuals)
        ),
        "intervals": intervals,
        "claim": (
            "Empirical dataset-level cross-validated residual interval; "
            "not an exact conformal coverage guarantee."
        ),
    }


def interval_for_prediction(
    prediction: float,
    target: str,
    calibration: Mapping[str, object],
    coverage: float = 0.90,
):
    key = "{:.2f}".format(float(coverage))
    intervals = calibration.get("intervals") or {}
    if key not in intervals:
        raise KeyError(
            "Coverage {} is unavailable.".format(key)
        )

    radius = float(
        intervals[key][
            "absolute_residual_quantile"
        ]
    )
    lower = float(prediction) - radius
    upper = float(prediction) + radius

    if target == "accuracy":
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    else:
        lower = max(0.0, lower)

    return float(lower), float(upper)
