from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class ObjectiveWeights:
    accuracy: float = 0.55
    runtime: float = 0.15
    energy: float = 0.10
    co2: float = 0.10
    fairness: float = 0.05
    interpretability: float = 0.05

    def normalized(self) -> "ObjectiveWeights":
        vals = {k: max(0.0, float(v)) for k, v in asdict(self).items()}
        s = sum(vals.values())
        if s <= 0:
            raise ValueError("At least one objective weight must be positive.")
        return ObjectiveWeights(**{k: v / s for k, v in vals.items()})

    def as_dict(self) -> dict[str, float]:
        return asdict(self.normalized())


@dataclass
class RunConfig:
    target: str
    sensitive_attribute: Optional[str] = None
    window_size: int = 500
    max_samples: int = 5000
    seed: int = 42
    time_budget_sec: float = 60.0
    positive_label: Any = 1
    track_sustainability: bool = False

    # Fairness audit policy. ``audit_only`` keeps the protected attribute for
    # group metrics but removes it from model inputs; ``include`` reproduces
    # experiments where the model can directly use the protected attribute.
    sensitive_feature_policy: str = "audit_only"

    # Phase 2 instrumentation controls.
    missing_prediction_policy: str = "incorrect"
    fairness_min_group_n: int = 10
    drift_recovery_tolerance: float = 0.02
    drift_recovery_max_samples: Optional[int] = None
    drift_min_assessment_samples: Optional[int] = None
    capture_native_xai_snapshots: bool = True

    # Phase 3/3.1 XAI controls. "auto" = SHAP -> LIME -> repeated permutation.
    xai_method: str = "auto"
    xai_max_rows: int = 250
    xai_replay_warning_threshold: float = 0.05

    # Prediction-quality safeguard used when interpreting fairness gaps.
    prediction_near_constant_threshold: float = 0.95


@dataclass
class MetricPoint:
    sample: int
    accuracy: float
    f1_macro: float
    rolling_accuracy: Optional[float] = None
    rolling_f1_macro: Optional[float] = None
    mean_prediction_latency_ms: Optional[float] = None
    throughput_samples_sec: Optional[float] = None
    drift: bool = False
    dp_diff: Optional[float] = None
    eo_diff: Optional[float] = None
    equalized_odds_gap: Optional[float] = None
    predictive_parity_diff: Optional[float] = None
    error_rate_gap: Optional[float] = None
    worst_group_accuracy: Optional[float] = None
    worst_group_macro_f1: Optional[float] = None


@dataclass
class FrameworkResult:
    framework: str
    backend: str
    status: str
    accuracy: float
    f1_macro: float
    runtime_sec: float
    samples: int
    energy_kwh: Optional[float] = None
    co2_kg: Optional[float] = None
    throughput_samples_sec: Optional[float] = None
    mean_prediction_latency_ms: Optional[float] = None
    p95_prediction_latency_ms: Optional[float] = None
    instrumentation_overhead_sec: float = 0.0
    drift_events: list[int] = field(default_factory=list)
    drift_summary: dict[str, Any] = field(default_factory=dict)
    points: list[MetricPoint] = field(default_factory=list)
    fairness: dict[str, Any] = field(default_factory=dict)
    explainability: dict[str, Any] = field(default_factory=dict)
    prediction_diagnostics: dict[str, Any] = field(default_factory=dict)
    dataset_provenance: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    sustainability: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    experiment_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["points"] = [asdict(p) for p in self.points]
        return d


@dataclass
class Recommendation:
    framework: str
    utility: float
    rank: int
    rationale: list[str]
    near_pareto: bool = False
    uncertainty_note: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
