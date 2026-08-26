from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import socket
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "2.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_token(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("-")
    clean = "".join(out).strip("-")
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean or "unknown"


def make_experiment_id(
    dataset_id: str,
    framework: str,
    seed: int,
    protocol_version: str = "meta-v2",
    nonce: Optional[str] = None,
) -> str:
    """Create a collision-resistant, human-readable experiment identifier.

    The id is deterministic when ``nonce`` is supplied, which is useful for HPC
    manifests. Otherwise a UTC timestamp is included so repeated local runs do
    not overwrite one another.
    """
    token = nonce or utc_now_iso()
    raw = f"{dataset_id}|{framework}|{int(seed)}|{protocol_version}|{token}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    stamp = _clean_token(token).replace("_", "-")[:24]
    return (
        f"{_clean_token(dataset_id)}__{_clean_token(framework)}__s{int(seed)}__"
        f"{_clean_token(protocol_version)}__{stamp}__{digest}"
    )


def default_environment() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }


def _require_probability(name: str, value: Optional[float]) -> None:
    if value is None:
        return
    fv = float(value)
    if not math.isfinite(fv) or fv < 0.0 or fv > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}.")


def _require_nonnegative(name: str, value: Optional[float]) -> None:
    if value is None:
        return
    fv = float(value)
    if not math.isfinite(fv) or fv < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}.")


@dataclass
class DatasetDescriptor:
    dataset_id: str
    file_name: str
    family: str
    source_type: str = "unknown"
    is_synthetic: Optional[bool] = None
    generator: Optional[str] = None
    drift_type: Optional[str] = None
    target: Optional[str] = None
    sensitive_attributes: List[str] = field(default_factory=list)
    fingerprint_sha256: Optional[str] = None
    notes: Optional[str] = None
    family_inference: Optional[str] = None

    def validate(self) -> "DatasetDescriptor":
        if not str(self.dataset_id).strip():
            raise ValueError("dataset_id is required.")
        if not str(self.file_name).strip():
            raise ValueError("file_name is required.")
        if not str(self.family).strip():
            raise ValueError("family is required.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class RunSummaryRecord:
    experiment_id: str
    protocol_version: str
    dataset_id: str
    framework: str
    seed: int
    status: str
    backend: Optional[str] = None
    framework_version: Optional[str] = None
    git_commit: Optional[str] = None
    created_at_utc: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    target: Optional[str] = None
    sensitive_attribute: Optional[str] = None
    n_samples: int = 0
    n_features: int = 0
    n_classes: int = 0
    dataset_provenance: Dict[str, Any] = field(default_factory=dict)
    window_size: int = 0
    max_samples: int = 0
    time_budget_sec: float = 0.0

    prequential_accuracy: Optional[float] = None
    prequential_macro_f1: Optional[float] = None
    runtime_sec: Optional[float] = None
    throughput_samples_sec: Optional[float] = None
    mean_prediction_latency_ms: Optional[float] = None
    p95_prediction_latency_ms: Optional[float] = None
    instrumentation_overhead_sec: float = 0.0

    drift_count: int = 0
    drift_recovery_rate: Optional[float] = None
    mean_drift_recovery_samples: Optional[float] = None
    max_accuracy_drop_after_drift: Optional[float] = None
    drift_summary: Dict[str, Any] = field(default_factory=dict)
    energy_kwh: Optional[float] = None
    co2_kg: Optional[float] = None
    sustainability_status: str = "not_measured"

    fairness_summary: Dict[str, Any] = field(default_factory=dict)
    explainability_summary: Dict[str, Any] = field(default_factory=dict)
    prediction_diagnostics: Dict[str, Any] = field(default_factory=dict)
    sustainability: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=default_environment)
    hpc: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def validate(self) -> "RunSummaryRecord":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version={self.schema_version!r}.")
        if not self.experiment_id:
            raise ValueError("experiment_id is required.")
        if not self.dataset_id:
            raise ValueError("dataset_id is required.")
        if not self.framework:
            raise ValueError("framework is required.")
        if int(self.seed) < 0:
            raise ValueError("seed must be >= 0.")
        if self.status not in {"ok", "failed", "partial", "skipped"}:
            raise ValueError("status must be one of: ok, failed, partial, skipped.")
        if int(self.n_samples) < 0 or int(self.n_features) < 0 or int(self.n_classes) < 0:
            raise ValueError("dataset dimensions must be non-negative.")
        _require_probability("prequential_accuracy", self.prequential_accuracy)
        _require_probability("prequential_macro_f1", self.prequential_macro_f1)
        _require_probability("drift_recovery_rate", self.drift_recovery_rate)
        _require_probability("max_accuracy_drop_after_drift", self.max_accuracy_drop_after_drift)
        _require_nonnegative("mean_drift_recovery_samples", self.mean_drift_recovery_samples)
        for name in [
            "runtime_sec", "throughput_samples_sec", "mean_prediction_latency_ms",
            "p95_prediction_latency_ms", "instrumentation_overhead_sec", "energy_kwh", "co2_kg",
        ]:
            _require_nonnegative(name, getattr(self, name))
        if self.sustainability_status not in {"measured", "not_measured", "failed", "partial"}:
            raise ValueError("Invalid sustainability_status.")
        # Research integrity: a missing measurement is represented as None, never
        # converted to a decorative zero by the schema.
        if self.sustainability_status == "not_measured" and (self.energy_kwh is not None or self.co2_kg is not None):
            raise ValueError("not_measured sustainability records must keep energy/co2 as None.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class WindowMetricRecord:
    experiment_id: str
    window_id: int
    sample_index: int
    prequential_accuracy: Optional[float]
    prequential_macro_f1: Optional[float]
    rolling_accuracy: Optional[float] = None
    rolling_macro_f1: Optional[float] = None
    prediction_latency_ms: Optional[float] = None
    throughput_samples_sec: Optional[float] = None
    drift_detected: bool = False
    energy_kwh_cumulative: Optional[float] = None
    co2_kg_cumulative: Optional[float] = None
    timestamp_utc: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> "WindowMetricRecord":
        if int(self.window_id) < 0 or int(self.sample_index) < 0:
            raise ValueError("window_id and sample_index must be non-negative.")
        for name in ["prequential_accuracy", "prequential_macro_f1", "rolling_accuracy", "rolling_macro_f1"]:
            _require_probability(name, getattr(self, name))
        for name in ["prediction_latency_ms", "throughput_samples_sec", "energy_kwh_cumulative", "co2_kg_cumulative"]:
            _require_nonnegative(name, getattr(self, name))
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class DriftEventRecord:
    experiment_id: str
    sample_index: int
    detector: str
    event_type: str = "drift"
    score: Optional[float] = None
    performance_before: Optional[float] = None
    performance_after: Optional[float] = None
    recovery_samples: Optional[int] = None
    accuracy_drop: Optional[float] = None
    recovered_at_sample: Optional[int] = None
    assessment_end_sample: Optional[int] = None
    degradation_observed: Optional[bool] = None
    timestamp_utc: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> "DriftEventRecord":
        if int(self.sample_index) < 0:
            raise ValueError("sample_index must be non-negative.")
        _require_probability("performance_before", self.performance_before)
        _require_probability("performance_after", self.performance_after)
        if self.recovery_samples is not None and int(self.recovery_samples) < 0:
            raise ValueError("recovery_samples must be non-negative.")
        if self.recovered_at_sample is not None and int(self.recovered_at_sample) < int(self.sample_index):
            raise ValueError("recovered_at_sample cannot precede the drift sample.")
        if self.assessment_end_sample is not None and int(self.assessment_end_sample) < int(self.sample_index):
            raise ValueError("assessment_end_sample cannot precede the drift sample.")
        _require_probability("accuracy_drop", self.accuracy_drop)
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class FairnessSnapshotRecord:
    experiment_id: str
    window_id: int
    sample_index: int
    status: str
    sensitive_attribute: Optional[str] = None
    dp_diff: Optional[float] = None
    eo_diff: Optional[float] = None
    equalized_odds_gap: Optional[float] = None
    predictive_parity_diff: Optional[float] = None
    error_rate_gap: Optional[float] = None
    worst_group_accuracy: Optional[float] = None
    worst_group_macro_f1: Optional[float] = None
    group_support: Dict[str, int] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> "FairnessSnapshotRecord":
        if self.status not in {"ok", "insufficient_support", "not_requested", "failed"}:
            raise ValueError("Invalid fairness status.")
        metrics = [
            "dp_diff", "eo_diff", "equalized_odds_gap", "predictive_parity_diff",
            "error_rate_gap", "worst_group_accuracy", "worst_group_macro_f1",
        ]
        for name in metrics:
            _require_probability(name, getattr(self, name))
        if self.status == "insufficient_support" and any(getattr(self, n) is not None for n in metrics[:5]):
            raise ValueError("insufficient_support fairness snapshots must use None for disparity metrics.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class ExplainabilitySnapshotRecord:
    experiment_id: str
    window_id: int
    sample_index: int
    status: str
    method: Optional[str] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    stability: Optional[float] = None
    fidelity: Optional[float] = None
    sensitivity: Optional[float] = None
    consistency: Optional[float] = None
    sparsity: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> "ExplainabilitySnapshotRecord":
        if self.status not in {"ok", "unsupported", "insufficient_data", "failed"}:
            raise ValueError("Invalid explainability status.")
        for name in ["stability", "fidelity", "sensitivity", "consistency", "sparsity"]:
            _require_probability(name, getattr(self, name))
        if self.status == "ok" and not self.method:
            raise ValueError("method is required for successful explainability snapshots.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class SustainabilitySnapshotRecord:
    experiment_id: str
    status: str
    duration_sec: Optional[float] = None
    energy_kwh: Optional[float] = None
    co2_kg: Optional[float] = None
    country_iso: Optional[str] = None
    carbon_intensity_g_per_kwh: Optional[float] = None
    backend: Optional[str] = None
    hardware: Dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> "SustainabilitySnapshotRecord":
        if self.status not in {"measured", "not_measured", "partial", "failed"}:
            raise ValueError("Invalid sustainability status.")
        for name in ["duration_sec", "energy_kwh", "co2_kg", "carbon_intensity_g_per_kwh"]:
            _require_nonnegative(name, getattr(self, name))
        if self.status == "not_measured" and (self.energy_kwh is not None or self.co2_kg is not None):
            raise ValueError("not_measured sustainability snapshots must use None for energy/co2.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


def canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
