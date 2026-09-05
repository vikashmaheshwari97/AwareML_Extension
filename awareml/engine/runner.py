from __future__ import annotations

from collections import deque
from dataclasses import asdict
import time
from typing import Any, Callable, Iterable, Optional
import numpy as np
import pandas as pd

try:
    from river.drift import ADWIN
except Exception:
    ADWIN = None

from awareml.analysis import SlidingFairness, SustainabilitySession, explain_framework
from awareml.config import settings
from awareml.data import StreamingEncoder
from awareml.types import FrameworkResult, MetricPoint, RunConfig
from .metrics import (
    DriftRecoveryTracker,
    LatencyTracker,
    OnlineClassificationMetrics,
    RollingClassificationMetrics,
    PredictionDiagnosticsTracker,
)
from .uncertainty import bootstrap_mean_ci


def _normalize_importance(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    clean: dict[str, float] = {}
    for k, v in raw.items():
        try:
            fv = abs(float(v))
        except Exception:
            continue
        if np.isfinite(fv):
            clean[str(k)] = fv
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {k: float(v / total) for k, v in clean.items()}


def _explanation_to_importance(explanation: dict[str, Any]) -> dict[str, float]:
    fi = explanation.get("feature_importance") if isinstance(explanation, dict) else None
    if isinstance(fi, dict):
        return _normalize_importance(fi)
    if isinstance(fi, list):
        raw = {}
        for item in fi:
            if not isinstance(item, dict):
                continue
            feature = item.get("feature")
            value = item.get("importance")
            if feature is not None:
                raw[str(feature)] = value
        return _normalize_importance(raw)
    return {}


def _map_fairness_status(status: str) -> str:
    return {
        "ok": "ok",
        "insufficient_group_support": "insufficient_support",
        "insufficient_support": "insufficient_support",
        "not_requested": "not_requested",
        "failed": "failed",
    }.get(str(status), "failed")


def _temporal_fairness_summary(points: list[MetricPoint]) -> dict[str, Any]:
    """Summarize fairness trajectories without collapsing missing windows to zero."""
    fields = [
        "dp_diff",
        "eo_diff",
        "equalized_odds_gap",
        "predictive_parity_diff",
        "error_rate_gap",
        "group_brier_score_gap",
        "group_ece_gap",
    ]
    out: dict[str, Any] = {"n_windows": len(points), "metrics": {}}
    for field in fields:
        vals = []
        samples = []
        for point in points:
            value = getattr(point, field, None)
            if value is None:
                continue
            try:
                fv = float(value)
            except Exception:
                continue
            if np.isfinite(fv):
                vals.append(fv)
                samples.append(float(point.sample))
        if not vals:
            out["metrics"][field] = {"n": 0, "mean": None, "max": None, "p95": None, "volatility": None, "time_weighted_mean": None}
            continue
        arr = np.asarray(vals, dtype=float)
        if len(vals) >= 2 and samples[-1] > samples[0]:
            tw = float(np.trapz(arr, np.asarray(samples, dtype=float)) / (samples[-1] - samples[0]))
        else:
            tw = float(arr[0])
        out["metrics"][field] = {
            "n": int(len(vals)),
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "p95": float(np.percentile(arr, 95)),
            "volatility": float(np.std(arr)),
            "time_weighted_mean": tw,
        }
    wg_acc = [p.worst_group_accuracy for p in points if p.worst_group_accuracy is not None]
    wg_f1 = [p.worst_group_macro_f1 for p in points if p.worst_group_macro_f1 is not None]
    out["worst_over_time_accuracy"] = float(min(wg_acc)) if wg_acc else None
    out["worst_over_time_macro_f1"] = float(min(wg_f1)) if wg_f1 else None
    return out


def _persist_run_result(
    store,
    result: FrameworkResult,
    cfg: RunConfig,
    experiment_id: str,
    dataset_id: str,
    protocol_version: str,
    n_features: int,
    n_classes: int,
    dataset_provenance: Optional[dict[str, Any]] = None,
) -> None:
    from awareml.experiments.adapters import framework_result_to_run_summary

    summary = framework_result_to_run_summary(
        result=result,
        config=cfg,
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        protocol_version=protocol_version,
        n_features=n_features,
        n_classes=n_classes,
        dataset_provenance=dataset_provenance,
    )
    store.write_run_summary(summary, overwrite=True)


def _run_one(
    framework,
    df: pd.DataFrame,
    cfg: RunConfig,
    progress: Optional[Callable] = None,
    *,
    experiment_store=None,
    experiment_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    protocol_version: str = "meta-v2",
    dataset_provenance: Optional[dict[str, Any]] = None,
) -> FrameworkResult:
    if cfg.sensitive_feature_policy not in {"include", "audit_only"}:
        raise ValueError("sensitive_feature_policy must be 'include' or 'audit_only'.")
    features = [c for c in df.columns if c != cfg.target]
    if (
        cfg.sensitive_attribute
        and cfg.sensitive_feature_policy == "audit_only"
        and cfg.sensitive_attribute in features
    ):
        features = [c for c in features if c != cfg.sensitive_attribute]
    encoder = StreamingEncoder()
    metrics = OnlineClassificationMetrics(cfg.missing_prediction_policy)
    rolling = RollingClassificationMetrics(cfg.window_size, cfg.missing_prediction_policy)
    latency = LatencyTracker()
    prediction_diagnostics = PredictionDiagnosticsTracker(
        positive_label=cfg.positive_label,
        near_constant_threshold=cfg.prediction_near_constant_threshold,
    )
    recovery = DriftRecoveryTracker(
        tolerance=cfg.drift_recovery_tolerance,
        max_recovery_samples=cfg.drift_recovery_max_samples,
        min_assessment_samples=(
            cfg.drift_min_assessment_samples
            if cfg.drift_min_assessment_samples is not None
            else max(10, min(100, int(cfg.window_size) // 5))
        ),
    )
    fairness = SlidingFairness(
        window_size=cfg.window_size,
        positive_label=cfg.positive_label,
        min_group_n=cfg.fairness_min_group_n,
        calibration_bins=cfg.fairness_calibration_bins,
        degenerate_prediction_threshold=cfg.prediction_near_constant_threshold,
    )
    detector = ADWIN() if ADWIN is not None else None
    points: list[MetricPoint] = []
    drift_events: list[int] = []
    drift_scores: dict[int, Optional[float]] = {}
    window_acc: list[float] = []
    window_f1: list[float] = []
    recent_X = deque(maxlen=min(300, cfg.window_size))
    recent_y = deque(maxlen=min(300, cfg.window_size))
    instrumentation_overhead = 0.0

    sustain = SustainabilitySession(
        enabled=cfg.track_sustainability,
        country_iso=settings.country_iso,
        project_name=f"AwareML-{framework.name}",
        region=cfg.sustainability_region,
        warmup_sec=cfg.sustainability_warmup_sec,
        warmup_samples=cfg.sustainability_warmup_samples,
        repetition_id=cfg.sustainability_repetition_id,
        repetitions_planned=cfg.sustainability_repetitions_planned,
    ).start()
    started = time.perf_counter()
    processed = 0
    deadline = started + max(1.0, float(cfg.time_budget_sec))
    last_drift = False
    window_id = 0
    last_snapshot_sample = 0

    def snapshot_window(sample_index: int, drift_in_window: bool) -> None:
        nonlocal window_id, last_snapshot_sample
        if sample_index <= 0 or sample_index == last_snapshot_sample:
            return
        window_id += 1
        last_snapshot_sample = sample_index
        fair = fairness.compute() if cfg.sensitive_attribute else {"status": "not_requested"}
        model_elapsed = max(1e-12, time.perf_counter() - started)
        throughput = sample_index / model_elapsed
        recent_latency = latency.recent_mean_ms(min(cfg.window_size, len(latency.values)))
        point = MetricPoint(
            sample=sample_index,
            accuracy=metrics.accuracy,
            f1_macro=metrics.f1_macro,
            rolling_accuracy=rolling.accuracy if rolling.n else None,
            rolling_f1_macro=rolling.f1_macro if rolling.n else None,
            mean_prediction_latency_ms=recent_latency,
            throughput_samples_sec=throughput,
            drift=bool(drift_in_window),
            dp_diff=fair.get("dp_diff"),
            eo_diff=fair.get("equal_opportunity_diff"),
            equalized_odds_gap=fair.get("equalized_odds_gap"),
            predictive_parity_diff=fair.get("predictive_parity_diff"),
            error_rate_gap=fair.get("error_rate_gap"),
            group_brier_score_gap=fair.get("group_brier_score_gap"),
            group_ece_gap=fair.get("group_ece_gap"),
            worst_group_accuracy=fair.get("worst_group_accuracy"),
            worst_group_macro_f1=fair.get("worst_group_macro_f1"),
        )
        points.append(point)
        window_acc.append(rolling.accuracy if rolling.n else metrics.accuracy)
        window_f1.append(rolling.f1_macro if rolling.n else metrics.f1_macro)

        if experiment_store is None or experiment_id is None:
            return

        from awareml.experiments import (
            ExplainabilitySnapshotRecord,
            FairnessSnapshotRecord,
            WindowMetricRecord,
        )

        experiment_store.append_window(WindowMetricRecord(
            experiment_id=experiment_id,
            window_id=window_id,
            sample_index=sample_index,
            prequential_accuracy=metrics.accuracy,
            prequential_macro_f1=metrics.f1_macro,
            rolling_accuracy=rolling.accuracy if rolling.n else None,
            rolling_macro_f1=rolling.f1_macro if rolling.n else None,
            prediction_latency_ms=recent_latency,
            throughput_samples_sec=throughput,
            drift_detected=bool(drift_in_window),
        ))

        experiment_store.append_fairness(FairnessSnapshotRecord(
            experiment_id=experiment_id,
            window_id=window_id,
            sample_index=sample_index,
            status=_map_fairness_status(fair.get("status", "not_requested")),
            sensitive_attribute=cfg.sensitive_attribute,
            dp_diff=fair.get("dp_diff"),
            eo_diff=fair.get("equal_opportunity_diff"),
            equalized_odds_gap=fair.get("equalized_odds_gap"),
            predictive_parity_diff=fair.get("predictive_parity_diff"),
            error_rate_gap=fair.get("error_rate_gap"),
            calibration_status={
                "insufficient_group_support": "insufficient_support"
            }.get(
                str(fair.get("calibration_status") or "unavailable"),
                str(fair.get("calibration_status") or "unavailable"),
            ),
            group_brier_score_gap=fair.get("group_brier_score_gap"),
            group_ece_gap=fair.get("group_ece_gap"),
            calibration_reason=fair.get("calibration_reason"),
            worst_group_accuracy=fair.get("worst_group_accuracy"),
            worst_group_macro_f1=fair.get("worst_group_macro_f1"),
            group_support={str(k): int(v) for k, v in (fair.get("groups") or {}).items()},
        ))

        if cfg.capture_native_xai_snapshots:
            try:
                native = _normalize_importance(framework.native_feature_importance())
            except Exception:
                native = {}
            if native:
                experiment_store.append_explainability(ExplainabilitySnapshotRecord(
                    experiment_id=experiment_id,
                    window_id=window_id,
                    sample_index=sample_index,
                    status="ok",
                    method="native",
                    feature_importance=native,
                ))

    try:
        limit = min(len(df), max(1, int(cfg.max_samples)))
        drift_since_snapshot = False
        for i in range(limit):
            if time.perf_counter() >= deadline:
                break
            row = df.iloc[i]
            y = row[cfg.target]
            x_raw = row[features]
            x = encoder.transform_row(x_raw)

            t_pred = time.perf_counter_ns()
            pred = framework.predict_one(x)
            latency.update((time.perf_counter_ns() - t_pred) / 1_000_000.0)
            prediction_diagnostics.update(pred)

            # Phase 14: probability evidence is requested only for a fairness audit.
            # It is measured as instrumentation overhead and never synthesized.
            y_proba = None
            if cfg.sensitive_attribute:
                t_probability = time.perf_counter()
                try:
                    y_proba = framework.predict_proba_one(x)
                except Exception:
                    y_proba = None
                instrumentation_overhead += max(
                    0.0, time.perf_counter() - t_probability
                )

            # Preserve the rolling baseline before observing the current label;
            # this is the reference used for post-drift degradation/recovery.
            rolling_before = rolling.accuracy if rolling.n else None

            # Prequential metrics are updated before training.
            metrics.update(y, pred)
            rolling.update(y, pred)

            drift = False
            if detector is not None and pred is not None:
                try:
                    error = 0.0 if pred == y else 1.0
                    detector.update(error)
                    drift = bool(
                        getattr(detector, "drift_detected", False)
                        or getattr(detector, "change_detected", False)
                    )
                except Exception:
                    drift = False
            if drift:
                sample_no = i + 1
                drift_events.append(sample_no)
                drift_since_snapshot = True
                score = None
                for attr in ("estimation", "width"):
                    try:
                        value = float(getattr(detector, attr))
                        if np.isfinite(value):
                            score = value
                            break
                    except Exception:
                        continue
                drift_scores[sample_no] = score
                recovery.on_drift(
                    sample_no,
                    baseline_accuracy=rolling_before,
                    immediate_accuracy=rolling.accuracy if rolling.n else None,
                )

            if cfg.sensitive_attribute and cfg.sensitive_attribute in row.index:
                fairness.update(
                    y,
                    pred,
                    row[cfg.sensitive_attribute],
                    y_proba=y_proba,
                )

            framework.learn_one(x, y)
            processed += 1
            recent_X.append(x)
            recent_y.append(y)
            recovery.update(processed, rolling.accuracy if rolling.n else None)
            last_drift = drift

            if processed % cfg.window_size == 0:
                snapshot_window(processed, drift_since_snapshot)
                drift_since_snapshot = False

            if progress is not None and (processed % max(25, cfg.window_size // 5) == 0):
                progress(framework.name, processed, limit)

        # Capture a final partial window when the run stops because of the time
        # budget or a sample cap that is not an exact multiple of window_size.
        if processed > last_snapshot_sample:
            snapshot_window(processed, drift_since_snapshot or last_drift)

    except Exception as e:
        model_runtime = max(0.0, time.perf_counter() - started)
        sustainability = sustain.stop().to_dict()
        try:
            parameters = framework.get_params()
        except Exception:
            parameters = {}
        result = FrameworkResult(
            framework=framework.name,
            backend=framework.backend,
            status="failed",
            accuracy=metrics.accuracy,
            f1_macro=metrics.f1_macro,
            runtime_sec=model_runtime,
            samples=processed,
            throughput_samples_sec=(processed / model_runtime) if model_runtime > 0 else None,
            mean_prediction_latency_ms=latency.mean_ms,
            p95_prediction_latency_ms=latency.p95_ms,
            instrumentation_overhead_sec=instrumentation_overhead,
            drift_events=drift_events,
            drift_summary=recovery.summary(),
            points=points,
            fairness=fairness.compute() if cfg.sensitive_attribute else {"status": "not_requested"},
            prediction_diagnostics=prediction_diagnostics.summary(),
            dataset_provenance=dict(dataset_provenance or {}),
            sustainability=sustainability,
            energy_kwh=sustainability.get("energy_kwh"),
            co2_kg=sustainability.get("co2_kg"),
            parameters=parameters,
            experiment_id=experiment_id,
            error=f"{type(e).__name__}: {e}",
        )
        if experiment_store is not None and experiment_id and dataset_id:
            _persist_run_result(
                experiment_store, result, cfg, experiment_id, dataset_id,
                protocol_version, len(features), int(df[cfg.target].nunique(dropna=True)),
                dataset_provenance=dataset_provenance,
            )
        try:
            framework.close()
        except Exception:
            pass
        return result

    # Stop framework sustainability measurement before post-hoc explainability.
    # This prevents XAI computation from contaminating framework energy/runtime
    # comparisons used by the meta-recommender.
    model_runtime = max(0.0, time.perf_counter() - started)
    sustainability = sustain.stop().to_dict()

    xai_started = time.perf_counter()
    X_exp = pd.DataFrame(list(recent_X))
    y_exp = pd.Series(list(recent_y))
    if len(X_exp) >= 30:
        try:
            explanation = explain_framework(
                framework,
                X_exp,
                y_exp,
                seed=cfg.seed,
                max_rows=cfg.xai_max_rows,
                method_preference=cfg.xai_method,
                reference_accuracy=(rolling.accuracy if rolling.n else None),
                categorical_features=list(encoder.category_maps.keys()),
                categorical_value_names=encoder.category_maps,
                replay_warning_threshold=cfg.xai_replay_warning_threshold,
            )
        except Exception as exc:
            explanation = {
                "status": "failed",
                "method": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        explanation = {"status": "insufficient_data", "method": None}
    instrumentation_overhead += max(0.0, time.perf_counter() - xai_started)

    fair_final = fairness.compute() if cfg.sensitive_attribute else {"status": "not_requested"}
    if cfg.sensitive_attribute:
        fair_final = dict(fair_final)
        fair_final["sensitive_feature_policy"] = cfg.sensitive_feature_policy
        fair_final["temporal_summary"] = _temporal_fairness_summary(points)
    acc_ci = bootstrap_mean_ci(window_acc, seed=cfg.seed)
    f1_ci = bootstrap_mean_ci(window_f1, seed=cfg.seed + 31)
    unc = {
        # Backward-compatible key used by the current Decision Lab.
        "window_accuracy": acc_ci,
        "rolling_accuracy": acc_ci,
        "rolling_macro_f1": f1_ci,
    }
    drift_summary = recovery.summary()

    try:
        parameters = framework.get_params()
    except Exception:
        parameters = {}

    result = FrameworkResult(
        framework=framework.name,
        backend=framework.backend,
        status="ok",
        accuracy=metrics.accuracy,
        f1_macro=metrics.f1_macro,
        runtime_sec=model_runtime,
        samples=processed,
        throughput_samples_sec=(processed / model_runtime) if model_runtime > 0 else None,
        mean_prediction_latency_ms=latency.mean_ms,
        p95_prediction_latency_ms=latency.p95_ms,
        instrumentation_overhead_sec=instrumentation_overhead,
        energy_kwh=sustainability.get("energy_kwh"),
        co2_kg=sustainability.get("co2_kg"),
        drift_events=drift_events,
        drift_summary=drift_summary,
        points=points,
        fairness=fair_final,
        explainability=explanation,
        prediction_diagnostics=prediction_diagnostics.summary(),
        dataset_provenance=dict(dataset_provenance or {}),
        uncertainty=unc,
        sustainability=sustainability,
        parameters=parameters,
        experiment_id=experiment_id,
    )

    if experiment_store is not None and experiment_id is not None:
        from awareml.experiments import (
            DriftEventRecord,
            ExplainabilitySnapshotRecord,
            SustainabilitySnapshotRecord,
        )

        for ep in drift_summary.get("episodes", []):
            experiment_store.append_drift_event(DriftEventRecord(
                experiment_id=experiment_id,
                sample_index=int(ep["sample_index"]),
                detector="ADWIN" if detector is not None else "none",
                score=drift_scores.get(int(ep["sample_index"])),
                performance_before=ep.get("baseline_accuracy"),
                performance_after=ep.get("min_accuracy_after"),
                recovery_samples=ep.get("recovery_samples"),
                accuracy_drop=ep.get("accuracy_drop"),
                recovered_at_sample=ep.get("recovered_at_sample"),
                assessment_end_sample=ep.get("assessment_end_sample"),
                degradation_observed=ep.get("degradation_observed"),
            ))

        experiment_store.append_explainability(ExplainabilitySnapshotRecord(
            experiment_id=experiment_id,
            window_id=max(1, window_id),
            sample_index=processed,
            status=str(explanation.get("status") or "failed"),
            method=explanation.get("method"),
            feature_importance=_explanation_to_importance(explanation),
            stability=explanation.get("stability"),
            fidelity=explanation.get("fidelity"),
            sensitivity=explanation.get("sensitivity"),
            consistency=explanation.get("consistency"),
            sparsity=explanation.get("sparsity"),
            details={
                "method_attempts": explanation.get("method_attempts", []),
                "method_metadata": explanation.get("method_metadata", {}),
                "base_accuracy_on_explanation_window": explanation.get("base_accuracy_on_explanation_window"),
                "prequential_reference_accuracy": explanation.get("prequential_reference_accuracy"),
                "replay_accuracy_gap": explanation.get("replay_accuracy_gap"),
                "replay_warning_threshold": explanation.get("replay_warning_threshold"),
                "replay_warning": explanation.get("replay_warning"),
                "deletion_accuracy": explanation.get("deletion_accuracy"),
            },
        ))

        status_map = {
            "measured": "measured",
            "not_measured": "not_measured",
            "measurement_incomplete": "partial",
            "measurement_failed": "failed",
            "failed": "failed",
        }
        sustain_status = status_map.get(str(sustainability.get("status")), "partial")
        experiment_store.append_sustainability(SustainabilitySnapshotRecord(
            experiment_id=experiment_id,
            status=sustain_status,
            duration_sec=sustainability.get("duration_sec"),
            energy_kwh=sustainability.get("energy_kwh") if sustain_status != "not_measured" else None,
            co2_kg=sustainability.get("co2_kg") if sustain_status != "not_measured" else None,
            country_iso=sustainability.get("country_iso"),
            carbon_intensity_g_per_kwh=sustainability.get(
                "carbon_intensity_g_per_kwh"
            ),
            backend=sustainability.get("measurement_backend"),
            hardware={
                "cpu": sustainability.get("cpu"),
                "physical_cpus": sustainability.get("physical_cpus"),
                "logical_cpus": sustainability.get("logical_cpus"),
                "ram_gb": sustainability.get("ram_gb"),
                "gpu": sustainability.get("gpu"),
                "python": sustainability.get("python"),
                "codecarbon_version": sustainability.get("codecarbon_version"),
                "region": sustainability.get("region"),
                "warmup_sec": sustainability.get("warmup_sec"),
                "warmup_samples": sustainability.get("warmup_samples"),
                "repetition_id": sustainability.get("repetition_id"),
                "repetitions_planned": sustainability.get("repetitions_planned"),
                "measurement_failure_reason": sustainability.get(
                    "measurement_failure_reason"
                ),
                "carbon_intensity_source": sustainability.get(
                    "carbon_intensity_source"
                ),
            },
        ))

        if dataset_id:
            _persist_run_result(
                experiment_store, result, cfg, experiment_id, dataset_id,
                protocol_version, len(features), int(df[cfg.target].nunique(dropna=True)),
                dataset_provenance=dataset_provenance,
            )

    try:
        framework.close()
    except Exception:
        pass
    return result


def run_benchmark(
    df: pd.DataFrame,
    config: RunConfig,
    frameworks: Optional[Iterable[str]] = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
    *,
    record_experiments: bool = False,
    experiment_root: str = "artifacts/meta_experiments",
    dataset_id: Optional[str] = None,
    protocol_version: str = "meta-v2",
    experiment_nonce: Optional[str] = None,
    dataset_provenance: Optional[dict[str, Any]] = None,
) -> list[FrameworkResult]:
    """Run the five-framework benchmark under one prequential protocol.

    When ``record_experiments`` is enabled, each framework receives its own
    immutable experiment directory containing window, drift, fairness,
    explainability and sustainability records suitable for later Slurm/HPC
    consolidation. Existing UI callers remain backward compatible because all
    research-recording arguments are keyword-only and disabled by default.
    """
    if config.target not in df.columns:
        raise ValueError(f"Target column '{config.target}' is missing.")
    if config.sensitive_attribute and config.sensitive_attribute not in df.columns:
        raise ValueError(f"Sensitive attribute '{config.sensitive_attribute}' is missing.")
    if config.missing_prediction_policy not in {"incorrect", "skip"}:
        raise ValueError("missing_prediction_policy must be 'incorrect' or 'skip'.")
    if str(config.xai_method).lower() not in {"auto", "shap", "lime", "permutation"}:
        raise ValueError("xai_method must be one of: auto, shap, lime, permutation.")
    if int(config.xai_max_rows) < 30:
        raise ValueError("xai_max_rows must be at least 30.")
    if int(config.fairness_calibration_bins) < 2:
        raise ValueError("fairness_calibration_bins must be at least 2.")
    if float(config.sustainability_warmup_sec) < 0:
        raise ValueError("sustainability_warmup_sec must be >= 0.")
    if int(config.sustainability_repetition_id) < 1:
        raise ValueError("sustainability_repetition_id must be >= 1.")
    if int(config.sustainability_repetitions_planned) < int(config.sustainability_repetition_id):
        raise ValueError(
            "sustainability_repetitions_planned cannot be smaller than repetition_id."
        )
    if float(config.xai_replay_warning_threshold) < 0:
        raise ValueError("xai_replay_warning_threshold must be >= 0.")
    if not 0.5 <= float(config.prediction_near_constant_threshold) <= 1.0:
        raise ValueError("prediction_near_constant_threshold must be in [0.5, 1.0].")
    if record_experiments and not dataset_id:
        raise ValueError("dataset_id is required when record_experiments=True.")

    if dataset_provenance is None:
        from awareml.experiments.provenance import build_dataset_provenance
        dataset_provenance = build_dataset_provenance(
            df,
            target=config.target,
            sensitive_attribute=config.sensitive_attribute,
        )

    store = None
    make_id = None
    if record_experiments:
        from awareml.experiments import ExperimentStore, make_experiment_id
        store = ExperimentStore(experiment_root)
        make_id = make_experiment_id

    from awareml.frameworks import create_frameworks

    result = []
    n_features = max(0, len(df.columns) - 1)
    n_classes = int(df[config.target].nunique(dropna=True))

    for fw in create_frameworks(frameworks, seed=config.seed):
        exp_id = None
        if store is not None and make_id is not None:
            exp_id = make_id(
                dataset_id=str(dataset_id),
                framework=fw.name,
                seed=config.seed,
                protocol_version=protocol_version,
                nonce=experiment_nonce,
            )
            store.write_execution_manifest(
                exp_id,
                {
                    "experiment_id": exp_id,
                    "dataset_id": dataset_id,
                    "framework": fw.name,
                    "backend": fw.backend,
                    "protocol_version": protocol_version,
                    "run_config": asdict(config),
                    "dataset_shape": {"rows": int(len(df)), "features": n_features, "classes": n_classes},
                    "dataset_provenance": dict(dataset_provenance or {}),
                    "processed_sample_range": {
                        "start_index": 0,
                        "requested_max_samples": int(config.max_samples),
                        "available_rows": int(len(df)),
                    },
                    "research_note": (
                        "Test-then-train prequential protocol. Missing predictions are handled according "
                        "to run_config.missing_prediction_policy. Framework energy measurement stops "
                        "before post-hoc explainability to avoid contaminating sustainability comparison."
                    ),
                },
                overwrite=True,
            )

        result.append(_run_one(
            fw,
            df,
            config,
            progress=progress,
            experiment_store=store,
            experiment_id=exp_id,
            dataset_id=dataset_id,
            protocol_version=protocol_version,
            dataset_provenance=dataset_provenance,
        ))
    return result
