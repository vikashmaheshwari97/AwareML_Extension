from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Iterable, Optional
import math
import numpy as np


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _labels_equal(a: Any, b: Any) -> bool:
    if b is None:
        return False
    a = _scalar(a)
    b = _scalar(b)
    try:
        return bool(a == b)
    except Exception:
        return False


class OnlineClassificationMetrics:
    """Cumulative prequential classification metrics.

    ``missing_prediction_policy='incorrect'`` is the journal-grade default:
    a framework that cannot emit a prediction during warm-up does not receive a
    free pass. This keeps test-then-train comparisons conservative and aligned
    across the five backends.
    """

    def __init__(self, missing_prediction_policy: str = "incorrect"):
        if missing_prediction_policy not in {"incorrect", "skip"}:
            raise ValueError("missing_prediction_policy must be 'incorrect' or 'skip'.")
        self.missing_prediction_policy = missing_prediction_policy
        self.correct = 0
        self.total = 0
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)
        self.labels = set()

    def update(self, y_true: Any, y_pred: Any):
        y_true = _scalar(y_true)
        y_pred = _scalar(y_pred)
        if y_pred is None and self.missing_prediction_policy == "skip":
            return
        self.total += 1
        self.labels.add(y_true)
        if y_pred is not None:
            self.labels.add(y_pred)
        if _labels_equal(y_true, y_pred):
            self.correct += 1
            self.tp[y_true] += 1
        else:
            if y_pred is not None:
                self.fp[y_pred] += 1
            self.fn[y_true] += 1

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def f1_macro(self) -> float:
        vals = []
        for label in self.labels:
            tp, fp, fn = self.tp[label], self.fp[label], self.fn[label]
            denom = 2 * tp + fp + fn
            vals.append((2 * tp / denom) if denom else 0.0)
        return sum(vals) / len(vals) if vals else 0.0


class RollingClassificationMetrics:
    """Window-bounded Accuracy and Macro-F1 for streaming diagnostics."""

    def __init__(self, window_size: int = 500, missing_prediction_policy: str = "incorrect"):
        self.window_size = max(1, int(window_size))
        self.missing_prediction_policy = missing_prediction_policy
        self.buffer: Deque[tuple[Any, Any]] = deque(maxlen=self.window_size)

    def update(self, y_true: Any, y_pred: Any) -> None:
        if y_pred is None and self.missing_prediction_policy == "skip":
            return
        self.buffer.append((_scalar(y_true), _scalar(y_pred)))

    def _metrics(self) -> OnlineClassificationMetrics:
        m = OnlineClassificationMetrics(missing_prediction_policy=self.missing_prediction_policy)
        for yt, yp in self.buffer:
            m.update(yt, yp)
        return m

    @property
    def accuracy(self) -> float:
        return self._metrics().accuracy if self.buffer else 0.0

    @property
    def f1_macro(self) -> float:
        return self._metrics().f1_macro if self.buffer else 0.0

    @property
    def n(self) -> int:
        return len(self.buffer)


class LatencyTracker:
    """Streaming latency tracker storing milliseconds per prediction."""

    def __init__(self, maxlen: Optional[int] = None):
        self.values: Deque[float] = deque(maxlen=maxlen)

    def update(self, latency_ms: float) -> None:
        value = float(latency_ms)
        if math.isfinite(value) and value >= 0:
            self.values.append(value)

    @property
    def mean_ms(self) -> Optional[float]:
        return float(np.mean(self.values)) if self.values else None

    @property
    def p95_ms(self) -> Optional[float]:
        return float(np.percentile(list(self.values), 95)) if self.values else None

    def recent_mean_ms(self, n: int) -> Optional[float]:
        if not self.values:
            return None
        vals = list(self.values)[-max(1, int(n)):]
        return float(np.mean(vals))


class PredictionDiagnosticsTracker:
    """Track prediction coverage and detect degenerate/near-constant outputs.

    The diagnostics are descriptive safeguards rather than performance metrics.
    They prevent a weak near-constant classifier from looking attractive merely
    because group-rate fairness gaps are numerically small.
    """

    def __init__(self, positive_label: Any = 1, near_constant_threshold: float = 0.95):
        if not 0.5 <= float(near_constant_threshold) <= 1.0:
            raise ValueError("near_constant_threshold must be in [0.5, 1.0].")
        self.positive_label = _scalar(positive_label)
        self.near_constant_threshold = float(near_constant_threshold)
        self.total_requests = 0
        self.missing_predictions = 0
        self.counts = defaultdict(int)

    def update(self, y_pred: Any) -> None:
        self.total_requests += 1
        if y_pred is None:
            self.missing_predictions += 1
            return
        value = _scalar(y_pred)
        self.counts[value] += 1

    def summary(self) -> dict[str, Any]:
        observed = sum(self.counts.values())
        majority_count = max(self.counts.values()) if self.counts else 0
        majority_fraction = (majority_count / observed) if observed else None
        positive_count = 0
        for label, count in self.counts.items():
            if _labels_equal(label, self.positive_label):
                positive_count += count
        positive_rate = (positive_count / observed) if observed else None
        counts = {str(k): int(v) for k, v in sorted(self.counts.items(), key=lambda kv: str(kv[0]))}
        near_constant = bool(
            observed > 0 and majority_fraction is not None and majority_fraction >= self.near_constant_threshold
        )
        return {
            "n_prediction_requests": int(self.total_requests),
            "n_predictions_observed": int(observed),
            "n_missing_predictions": int(self.missing_predictions),
            "prediction_coverage": (observed / self.total_requests) if self.total_requests else None,
            "unique_predicted_labels": int(len(self.counts)),
            "predicted_label_counts": counts,
            "majority_prediction_fraction": majority_fraction,
            "positive_prediction_rate": positive_rate,
            "near_constant_threshold": self.near_constant_threshold,
            "near_constant_prediction": near_constant,
            "warning": (
                "Predictions are near-constant; interpret small fairness gaps cautiously."
                if near_constant else None
            ),
        }


@dataclass
class DriftEpisode:
    sample_index: int
    baseline_accuracy: Optional[float]
    immediate_accuracy: Optional[float]
    assessment_end_sample: int
    min_accuracy_after: Optional[float] = None
    degradation_observed: Optional[bool] = None
    recovered_at_sample: Optional[int] = None

    @property
    def recovery_samples(self) -> Optional[int]:
        if self.recovered_at_sample is None:
            return None
        return max(0, int(self.recovered_at_sample) - int(self.sample_index))

    @property
    def accuracy_drop(self) -> Optional[float]:
        if self.baseline_accuracy is None or self.min_accuracy_after is None:
            return None
        return max(0.0, float(self.baseline_accuracy) - float(self.min_accuracy_after))


class DriftRecoveryTracker:
    """Estimate post-drift degradation and recovery in sample units.

    Phase 3 avoids the optimistic ``recovered one sample later`` artefact. A
    detected drift is observed for ``min_assessment_samples`` before recovery
    is eligible. If rolling accuracy never degrades beyond the configured
    tolerance during that assessment horizon, recovery is reported as *not
    applicable* rather than as an instantaneous recovery.
    """

    def __init__(
        self,
        tolerance: float = 0.02,
        max_recovery_samples: Optional[int] = None,
        min_assessment_samples: int = 25,
    ):
        self.tolerance = max(0.0, float(tolerance))
        self.max_recovery_samples = None if max_recovery_samples is None else max(1, int(max_recovery_samples))
        self.min_assessment_samples = max(1, int(min_assessment_samples))
        self.episodes: list[DriftEpisode] = []

    def on_drift(
        self,
        sample_index: int,
        baseline_accuracy: Optional[float],
        immediate_accuracy: Optional[float] = None,
    ) -> DriftEpisode:
        baseline = None if baseline_accuracy is None else float(baseline_accuracy)
        immediate = baseline if immediate_accuracy is None else float(immediate_accuracy)
        ep = DriftEpisode(
            sample_index=int(sample_index),
            baseline_accuracy=baseline,
            immediate_accuracy=immediate,
            assessment_end_sample=int(sample_index) + self.min_assessment_samples,
            min_accuracy_after=immediate,
        )
        self.episodes.append(ep)
        return ep

    def update(self, sample_index: int, rolling_accuracy: Optional[float]) -> None:
        if rolling_accuracy is None:
            return
        idx = int(sample_index)
        acc = float(rolling_accuracy)
        for ep in self.episodes:
            if ep.recovered_at_sample is not None:
                continue
            if idx < ep.sample_index:
                continue

            if ep.baseline_accuracy is None:
                ep.degradation_observed = None
                continue

            threshold = max(0.0, ep.baseline_accuracy - self.tolerance)

            # The degradation decision is based on a fixed post-drift assessment
            # horizon. Once that horizon closes with no material degradation,
            # freeze the episode. This prevents later, unrelated fluctuations
            # from changing ``min_accuracy_after`` while leaving
            # ``degradation_observed=False`` -- the inconsistency exposed by the
            # Phase-3 ChaCha Adult run.
            if ep.degradation_observed is False:
                continue

            if ep.min_accuracy_after is None or acc < ep.min_accuracy_after:
                ep.min_accuracy_after = acc

            if idx < ep.assessment_end_sample:
                continue

            if ep.degradation_observed is None:
                ep.degradation_observed = bool(
                    ep.min_accuracy_after is not None and ep.min_accuracy_after < threshold
                )
                if ep.degradation_observed is False:
                    # Freeze a no-degradation episode at the end of its assessment
                    # horizon. Later concept changes belong to later drift events.
                    continue

            # Material degradation was observed during the assessment horizon.
            # Continue tracking until the rolling metric returns within tolerance.
            if acc >= threshold:
                ep.recovered_at_sample = idx
                continue

            if self.max_recovery_samples is not None and idx - ep.sample_index >= self.max_recovery_samples:
                # Explicitly leave unrecovered at the configured horizon.
                continue

    def summary(self) -> dict[str, Any]:
        applicable = [ep for ep in self.episodes if ep.degradation_observed is True]
        no_degradation = [ep for ep in self.episodes if ep.degradation_observed is False]
        recovered = [ep.recovery_samples for ep in applicable if ep.recovery_samples is not None]
        drops = [ep.accuracy_drop for ep in self.episodes if ep.accuracy_drop is not None]
        return {
            "n_drift_events": len(self.episodes),
            "n_recovery_applicable": len(applicable),
            "n_no_observed_degradation": len(no_degradation),
            "n_recovered": len(recovered),
            "recovery_rate": (len(recovered) / len(applicable)) if applicable else None,
            "mean_recovery_samples": float(np.mean(recovered)) if recovered else None,
            "median_recovery_samples": float(np.median(recovered)) if recovered else None,
            "max_accuracy_drop": float(max(drops)) if drops else None,
            "mean_accuracy_drop": float(np.mean(drops)) if drops else None,
            "tolerance": self.tolerance,
            "min_assessment_samples": self.min_assessment_samples,
            "episodes": [
                {
                    "sample_index": ep.sample_index,
                    "baseline_accuracy": ep.baseline_accuracy,
                    "immediate_accuracy": ep.immediate_accuracy,
                    "assessment_end_sample": ep.assessment_end_sample,
                    "min_accuracy_after": ep.min_accuracy_after,
                    "accuracy_drop": ep.accuracy_drop,
                    "degradation_observed": ep.degradation_observed,
                    "recovered_at_sample": ep.recovered_at_sample,
                    "recovery_samples": ep.recovery_samples,
                }
                for ep in self.episodes
            ],
        }
