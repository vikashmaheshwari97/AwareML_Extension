from __future__ import annotations

import logging
import warnings


class _DropKnownNoise(logging.Filter):
    """Drop only known repetitive console noise; preserve other warnings/errors."""

    BLOCKED = (
        "Multiple instances of codecarbon are allowed to run at the same time",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(token in message for token in self.BLOCKED)


def configure_runtime_hygiene() -> None:
    # LIME/feature-selection can invoke sklearn's least-angle code on degenerate
    # local perturbation matrices. These two RuntimeWarnings are repetitive and
    # already surfaced to the user as failed_or_degenerate XAI diagnostics.
    warnings.filterwarnings(
        "ignore",
        message="divide by zero encountered in log",
        category=RuntimeWarning,
        module=r"sklearn\.linear_model\._least_angle",
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in divide",
        category=RuntimeWarning,
        module=r"sklearn\.linear_model\._least_angle",
    )

    filt = _DropKnownNoise()
    for logger_name in ("codecarbon", "codecarbon.emissions_tracker", ""):
        logger = logging.getLogger(logger_name)
        logger.addFilter(filt)
        for handler in logger.handlers:
            handler.addFilter(filt)
