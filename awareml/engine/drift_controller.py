from __future__ import annotations

"""Drift detection + optional generic reset/replay adaptation for interactive runs.

The controller is deliberately separate from the frozen Phase-18 protocol.  The
``adwin`` mode reproduces the historical single-detector behaviour as closely as
possible, while ``hybrid`` adds a second detector and a performance-degradation
confirmation signal.  Adaptation is opt-in at the Run Studio level.
"""

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Optional

import numpy as np

try:
    from river.drift import ADWIN, PageHinkley
except Exception:  # pragma: no cover - handled at runtime
    ADWIN = None
    PageHinkley = None


@dataclass
class DriftSignal:
    sample_index: int
    detected: bool
    warning: bool
    adwin: bool
    page_hinkley: bool
    degradation: bool
    degradation_gap: Optional[float]
    score: Optional[float]
    sources: list[str]


class HybridDriftController:
    """Bounded-state detector with warm-up and thrash control.

    ``hybrid`` prefers agreement between statistical detectors, or a statistical
    detector plus a material fast-vs-slow performance degradation.  In v4, a
    material degradation that persists for a bounded confirmation horizon can
    also become a confirmed *performance-change* episode.  This is not proof of
    a change in the data-generating concept; the UI keeps that distinction clear.
    """

    name = "hybrid-adwin-pagehinkley"

    def __init__(
        self,
        mode: str = "adwin",
        window_size: int = 500,
        warmup_samples: Optional[int] = None,
        min_separation: Optional[int] = None,
        performance_drop: float = 0.03,
        adwin_delta: float = 0.002,
        degradation_confirm_samples: Optional[int] = None,
    ) -> None:
        mode = str(mode or "adwin").strip().lower()
        if mode not in {"adwin", "hybrid"}:
            raise ValueError("drift detector mode must be 'adwin' or 'hybrid'.")
        self.mode = mode
        self.window_size = max(20, int(window_size))
        self.warmup_samples = (
            0 if mode == "adwin" and warmup_samples is None
            else max(20, int(warmup_samples or max(100, self.window_size // 2)))
        )
        self.min_separation = (
            0 if mode == "adwin" and min_separation is None
            else max(1, int(min_separation or max(50, self.window_size // 2)))
        )
        self.performance_drop = max(0.0, float(performance_drop))
        self.adwin_delta = float(adwin_delta)
        self.degradation_confirm_samples = max(5, int(
            degradation_confirm_samples
            if degradation_confirm_samples is not None
            else max(20, min(250, self.window_size // 4))
        ))
        self.adwin = ADWIN(delta=self.adwin_delta) if ADWIN is not None else None
        self.page_hinkley = PageHinkley() if (mode == "hybrid" and PageHinkley is not None) else None
        self.fast_acc: Optional[float] = None
        self.slow_acc: Optional[float] = None
        self._last_drift = -10**18
        self._last_signal: Optional[DriftSignal] = None
        self._events: list[dict[str, Any]] = []
        self._warning_steps = 0
        self._warning_episodes = 0
        self._in_warning = False
        self._degradation_streak = 0
        self._updates = 0

    @staticmethod
    def _flag(detector: Any) -> bool:
        if detector is None:
            return False
        for attr in ("drift_detected", "change_detected"):
            try:
                if hasattr(detector, attr):
                    return bool(getattr(detector, attr))
            except Exception:
                return False
        return False

    @staticmethod
    def _warning_flag(detector: Any) -> bool:
        if detector is None:
            return False
        try:
            return bool(getattr(detector, "warning_detected", False))
        except Exception:
            return False

    def _update_accuracy_emas(self, accuracy: Optional[float]) -> Optional[float]:
        if accuracy is None:
            return None
        try:
            value = float(accuracy)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        alpha_fast = 2.0 / (max(10, self.window_size // 5) + 1.0)
        alpha_slow = 2.0 / (max(50, self.window_size) + 1.0)
        self.fast_acc = value if self.fast_acc is None else (alpha_fast * value + (1 - alpha_fast) * self.fast_acc)
        self.slow_acc = value if self.slow_acc is None else (alpha_slow * value + (1 - alpha_slow) * self.slow_acc)
        return max(0.0, float(self.slow_acc - self.fast_acc))

    def update(self, error: float, rolling_accuracy: Optional[float], sample_index: int) -> DriftSignal:
        self._updates += 1
        sample_index = int(sample_index)
        err = float(error)
        if self.adwin is not None:
            self.adwin.update(err)
        if self.page_hinkley is not None:
            self.page_hinkley.update(err)

        adwin = self._flag(self.adwin)
        page = self._flag(self.page_hinkley)
        gap = self._update_accuracy_emas(rolling_accuracy)
        degradation = bool(gap is not None and gap >= self.performance_drop)
        if degradation:
            self._degradation_streak += 1
        else:
            self._degradation_streak = 0
        sustained_degradation = bool(
            degradation and self._degradation_streak >= self.degradation_confirm_samples
        )

        if self.mode == "adwin":
            candidate = adwin
            warning = self._warning_flag(self.adwin)
        else:
            votes = int(adwin) + int(page)
            # v4 keeps statistical agreement as the strongest path, but also
            # allows a material performance drop to become a confirmed change
            # episode when it persists for a bounded horizon.  This avoids the
            # v3 behaviour where thousands of per-sample warnings could be
            # displayed while confirmed events remained permanently zero.
            candidate = bool(
                votes >= 2
                or ((adwin or page) and degradation)
                or sustained_degradation
            )
            warning = bool(
                self._warning_flag(self.adwin)
                or self._warning_flag(self.page_hinkley)
                or (degradation and not candidate)
            )

        separated = sample_index - self._last_drift >= self.min_separation
        warmed = sample_index >= self.warmup_samples
        detected = bool(candidate and separated and warmed)
        sources = []
        if adwin:
            sources.append("ADWIN")
        if page:
            sources.append("PageHinkley")
        if degradation:
            sources.append("performance_drop")
        if sustained_degradation:
            sources.append("sustained_performance_drop")
        if warning:
            self._warning_steps += 1
            if not self._in_warning:
                self._warning_episodes += 1
            self._in_warning = True
        else:
            self._in_warning = False

        score = None
        if gap is not None:
            score = float(gap)
        elif self.adwin is not None:
            try:
                score = float(getattr(self.adwin, "estimation"))
            except Exception:
                score = None

        signal = DriftSignal(
            sample_index=sample_index,
            detected=detected,
            warning=warning,
            adwin=adwin,
            page_hinkley=page,
            degradation=degradation,
            degradation_gap=gap,
            score=score,
            sources=sources,
        )
        self._last_signal = signal
        if detected:
            self._last_drift = sample_index
            self._events.append(asdict(signal))
            if self.mode == "hybrid":
                # Reset statistical state after a confirmed episode to avoid a
                # detector latch causing a burst of duplicate alerts.
                self.adwin = ADWIN(delta=self.adwin_delta) if ADWIN is not None else None
                self.page_hinkley = PageHinkley() if PageHinkley is not None else None
                self.fast_acc = rolling_accuracy if rolling_accuracy is not None else self.fast_acc
                self.slow_acc = rolling_accuracy if rolling_accuracy is not None else self.slow_acc
                self._degradation_streak = 0
                self._in_warning = False
        return signal

    @property
    def drift_detected(self) -> bool:
        return bool(self._last_signal.detected) if self._last_signal else False

    @property
    def estimation(self) -> float:
        if self._last_signal and self._last_signal.score is not None:
            return float(self._last_signal.score)
        raise AttributeError("No finite drift score is available.")

    @property
    def width(self) -> float:
        if self.adwin is not None and hasattr(self.adwin, "width"):
            return float(getattr(self.adwin, "width"))
        raise AttributeError("Detector width is unavailable.")

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "name": self.name if self.mode == "hybrid" else "ADWIN",
            "updates": int(self._updates),
            "confirmed_drifts": int(len(self._events)),
            "warnings": int(self._warning_episodes),
            "warning_steps": int(self._warning_steps),
            "warning_episodes": int(self._warning_episodes),
            "warmup_samples": int(self.warmup_samples),
            "min_separation": int(self.min_separation),
            "performance_drop_threshold": float(self.performance_drop),
            "degradation_confirm_samples": int(self.degradation_confirm_samples),
            "confirmation_rule": (
                "ADWIN+PageHinkley, statistical+performance-drop, or sustained performance-drop"
                if self.mode == "hybrid" else "ADWIN compatibility mode"
            ),
            "events": list(self._events),
        }


def adapt_framework(
    framework: Any,
    recent_training: Iterable[tuple[dict[str, float], Any]],
    sample_index: int,
    policy: str = "monitor_only",
    replay_size: int = 500,
    trigger_sources: Optional[list[str]] = None,
) -> Optional[dict[str, Any]]:
    """Apply a generic reset + bounded replay policy after confirmed drift.

    This is an AwareML wrapper action, not a claim that the upstream framework
    natively exposes a refit API.  The action is therefore recorded explicitly.
    """
    policy = str(policy or "monitor_only").strip().lower()
    if policy == "monitor_only":
        return None
    if policy != "adaptive_replay":
        raise ValueError("drift action policy must be 'monitor_only' or 'adaptive_replay'.")

    history = list(recent_training)
    if replay_size > 0:
        history = history[-int(replay_size):]
    started = __import__("time").perf_counter()
    try:
        framework.reset()
        replayed = 0
        for x, y in history:
            framework.learn_one(dict(x), y)
            replayed += 1
        return {
            "sample": int(sample_index),
            "policy": "adaptive_replay",
            "action": "awareml_reset_and_replay",
            "replayed_samples": int(replayed),
            "trigger_sources": list(trigger_sources or []),
            "duration_sec": float(__import__("time").perf_counter() - started),
            "status": "ok",
            "note": "AwareML wrapper reset/replay; not an upstream-native refit claim.",
        }
    except Exception as exc:
        return {
            "sample": int(sample_index),
            "policy": "adaptive_replay",
            "action": "awareml_reset_and_replay",
            "replayed_samples": 0,
            "trigger_sources": list(trigger_sources or []),
            "duration_sec": float(__import__("time").perf_counter() - started),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
