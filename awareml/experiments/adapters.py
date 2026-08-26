from __future__ import annotations

import os
import platform
import subprocess
from typing import Any, Dict, Optional

from awareml.types import FrameworkResult, RunConfig
from .records import RunSummaryRecord


def _git_commit() -> Optional[str]:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3, check=False
        )
        value = p.stdout.strip()
        return value if p.returncode == 0 and value else None
    except Exception:
        return None


def framework_result_to_run_summary(
    result: FrameworkResult,
    config: RunConfig,
    experiment_id: str,
    dataset_id: str,
    protocol_version: str = "meta-v2",
    n_features: int = 0,
    n_classes: int = 0,
    dataset_provenance: Optional[Dict[str, Any]] = None,
) -> RunSummaryRecord:
    sustainability = dict(result.sustainability or {})
    raw_status = str(sustainability.get("status") or "not_measured")
    status_map = {
        "ok": "measured",
        "measured": "measured",
        "not_measured": "not_measured",
        "disabled": "not_measured",
        "failed": "failed",
        "partial": "partial",
    }
    sustain_status = status_map.get(raw_status, "partial" if result.energy_kwh is not None else "not_measured")
    energy = result.energy_kwh if sustain_status != "not_measured" else None
    co2 = result.co2_kg if sustain_status != "not_measured" else None
    runtime = max(0.0, float(result.runtime_sec or 0.0))
    throughput = float(result.samples) / runtime if runtime > 0 else None
    return RunSummaryRecord(
        experiment_id=experiment_id,
        protocol_version=protocol_version,
        dataset_id=dataset_id,
        framework=result.framework,
        backend=result.backend,
        seed=int(config.seed),
        status=result.status if result.status in {"ok", "failed", "partial", "skipped"} else "partial",
        git_commit=_git_commit(),
        target=config.target,
        sensitive_attribute=config.sensitive_attribute,
        n_samples=int(result.samples),
        n_features=int(n_features),
        n_classes=int(n_classes),
        dataset_provenance=dict(dataset_provenance or result.dataset_provenance or {}),
        window_size=int(config.window_size),
        max_samples=int(config.max_samples),
        time_budget_sec=float(config.time_budget_sec),
        prequential_accuracy=float(result.accuracy) if result.accuracy is not None else None,
        prequential_macro_f1=float(result.f1_macro) if result.f1_macro is not None else None,
        runtime_sec=runtime,
        throughput_samples_sec=(
            float(result.throughput_samples_sec)
            if result.throughput_samples_sec is not None
            else throughput
        ),
        mean_prediction_latency_ms=result.mean_prediction_latency_ms,
        p95_prediction_latency_ms=result.p95_prediction_latency_ms,
        instrumentation_overhead_sec=float(result.instrumentation_overhead_sec or 0.0),
        drift_count=len(result.drift_events or []),
        drift_recovery_rate=(result.drift_summary or {}).get("recovery_rate"),
        mean_drift_recovery_samples=(result.drift_summary or {}).get("mean_recovery_samples"),
        max_accuracy_drop_after_drift=(result.drift_summary or {}).get("max_accuracy_drop"),
        drift_summary=dict(result.drift_summary or {}),
        energy_kwh=energy,
        co2_kg=co2,
        sustainability_status=sustain_status,
        fairness_summary=dict(result.fairness or {}),
        explainability_summary=dict(result.explainability or {}),
        prediction_diagnostics=dict(result.prediction_diagnostics or {}),
        sustainability=sustainability,
        parameters=dict(result.parameters or {}),
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        },
        hpc={
            "job_id": os.getenv("SLURM_JOB_ID"),
            "array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
            "node": os.getenv("SLURMD_NODENAME") or os.getenv("HOSTNAME"),
        },
        error=result.error,
    )
