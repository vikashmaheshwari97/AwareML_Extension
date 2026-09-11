from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
PAYLOAD = ROOT / "phase14_payload"
BACKUP = ROOT / ".phase14_hardening_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256(path: Path):
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes():
    paths = [
        ROOT / "data/journal/objective_selection_benchmark_v1/frozen/manifest.json",
        ROOT / "data/journal/recommender_multiobjective_validation_v1/frozen/manifest.json",
        ROOT / "awareml/llm/objective_selection_v3.py",
        ROOT / "awareml/llm/objective_selection_v31.py",
        ROOT / "awareml/recommender/v2_service.py",
        ROOT / "awareml/recommender/v2_ranking.py",
        ROOT / "data/meta/models/recommender_v2/manifest.json",
        ROOT / "data/meta/snapshots/meta_logs_v2.json",
        ROOT / "data/meta/snapshots/recommender_train_v2.parquet",
    ]
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in paths
        if path.exists()
    }


def backup(path: Path):
    if not path.exists():
        return
    out = BACKUP / path.relative_to(ROOT)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, out)


def copy_payload():
    for src in PAYLOAD.rglob("*"):
        if not src.is_file():
            continue
        dest = ROOT / src.relative_to(PAYLOAD)
        backup(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def replace_once(path: Path, old: str, new: str, label: str):
    text = path.read_text(encoding="utf-8")
    if new in text:
        print("Already applied:", label)
        return
    if old not in text:
        raise RuntimeError(
            "Could not apply {}: expected anchor was not found in {}".format(label, path)
        )
    backup(path)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("Patched:", label)


def patch_types():
    path = ROOT / "awareml" / "types.py"

    replace_once(
        path,
        '    prediction_near_constant_threshold: float = 0.95\n',
        '''    prediction_near_constant_threshold: float = 0.95

    # Phase 14 fairness-calibration and sustainability protocol controls.
    fairness_calibration_bins: int = 10
    sustainability_region: Optional[str] = None
    sustainability_warmup_sec: float = 0.0
    sustainability_warmup_samples: int = 0
    sustainability_repetition_id: int = 1
    sustainability_repetitions_planned: int = 1
''',
        "RunConfig Phase-14 controls",
    )

    replace_once(
        path,
        '    error_rate_gap: Optional[float] = None\n    worst_group_accuracy: Optional[float] = None\n',
        '''    error_rate_gap: Optional[float] = None
    group_brier_score_gap: Optional[float] = None
    group_ece_gap: Optional[float] = None
    worst_group_accuracy: Optional[float] = None
''',
        "MetricPoint calibration fairness fields",
    )


def patch_records():
    path = ROOT / "awareml" / "experiments" / "records.py"

    replace_once(
        path,
        '''    error_rate_gap: Optional[float] = None
    worst_group_accuracy: Optional[float] = None
''',
        '''    error_rate_gap: Optional[float] = None
    calibration_status: str = "unavailable"
    group_brier_score_gap: Optional[float] = None
    group_ece_gap: Optional[float] = None
    calibration_reason: Optional[str] = None
    worst_group_accuracy: Optional[float] = None
''',
        "FairnessSnapshotRecord calibration fields",
    )

    replace_once(
        path,
        '''            "dp_diff", "eo_diff", "equalized_odds_gap", "predictive_parity_diff",
            "error_rate_gap", "worst_group_accuracy", "worst_group_macro_f1",
        ]
''',
        '''            "dp_diff", "eo_diff", "equalized_odds_gap", "predictive_parity_diff",
            "error_rate_gap", "group_brier_score_gap", "group_ece_gap",
            "worst_group_accuracy", "worst_group_macro_f1",
        ]
''',
        "FairnessSnapshotRecord probability metric validation",
    )

    replace_once(
        path,
        '''        if self.status == "insufficient_support" and any(getattr(self, n) is not None for n in metrics[:5]):
            raise ValueError("insufficient_support fairness snapshots must use None for disparity metrics.")
        return self
''',
        '''        if self.status == "insufficient_support" and any(getattr(self, n) is not None for n in metrics[:5]):
            raise ValueError("insufficient_support fairness snapshots must use None for hard-label disparity metrics.")
        if self.calibration_status not in {
            "ok", "unavailable", "insufficient_support", "not_requested", "failed"
        }:
            raise ValueError("Invalid calibration fairness status.")
        if self.calibration_status != "ok" and (
            self.group_brier_score_gap is not None or self.group_ece_gap is not None
        ):
            raise ValueError(
                "Unavailable/insufficient calibration fairness must keep Brier/ECE gaps as None."
            )
        return self
''',
        "FairnessSnapshotRecord calibration status validation",
    )


def patch_runner():
    path = ROOT / "awareml" / "engine" / "runner.py"

    replace_once(
        path,
        '''        "predictive_parity_diff",
        "error_rate_gap",
    ]
''',
        '''        "predictive_parity_diff",
        "error_rate_gap",
        "group_brier_score_gap",
        "group_ece_gap",
    ]
''',
        "Temporal calibration fairness summary",
    )

    replace_once(
        path,
        '''    fairness = SlidingFairness(
        window_size=cfg.window_size,
        positive_label=cfg.positive_label,
        min_group_n=cfg.fairness_min_group_n,
    )
''',
        '''    fairness = SlidingFairness(
        window_size=cfg.window_size,
        positive_label=cfg.positive_label,
        min_group_n=cfg.fairness_min_group_n,
        calibration_bins=cfg.fairness_calibration_bins,
    )
''',
        "SlidingFairness calibration bins",
    )

    replace_once(
        path,
        '''    sustain = SustainabilitySession(
        enabled=cfg.track_sustainability,
        country_iso=settings.country_iso,
        project_name=f"AwareML-{framework.name}",
    ).start()
''',
        '''    sustain = SustainabilitySession(
        enabled=cfg.track_sustainability,
        country_iso=settings.country_iso,
        project_name=f"AwareML-{framework.name}",
        region=cfg.sustainability_region,
        warmup_sec=cfg.sustainability_warmup_sec,
        warmup_samples=cfg.sustainability_warmup_samples,
        repetition_id=cfg.sustainability_repetition_id,
        repetitions_planned=cfg.sustainability_repetitions_planned,
    ).start()
''',
        "Sustainability protocol metadata",
    )

    replace_once(
        path,
        '''            predictive_parity_diff=fair.get("predictive_parity_diff"),
            error_rate_gap=fair.get("error_rate_gap"),
            worst_group_accuracy=fair.get("worst_group_accuracy"),
''',
        '''            predictive_parity_diff=fair.get("predictive_parity_diff"),
            error_rate_gap=fair.get("error_rate_gap"),
            group_brier_score_gap=fair.get("group_brier_score_gap"),
            group_ece_gap=fair.get("group_ece_gap"),
            worst_group_accuracy=fair.get("worst_group_accuracy"),
''',
        "MetricPoint calibration fairness values",
    )

    replace_once(
        path,
        '''            predictive_parity_diff=fair.get("predictive_parity_diff"),
            error_rate_gap=fair.get("error_rate_gap"),
            worst_group_accuracy=fair.get("worst_group_accuracy"),
            worst_group_macro_f1=fair.get("worst_group_macro_f1"),
            group_support={str(k): int(v) for k, v in (fair.get("groups") or {}).items()},
''',
        '''            predictive_parity_diff=fair.get("predictive_parity_diff"),
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
''',
        "FairnessSnapshotRecord calibration values",
    )

    replace_once(
        path,
        '''            pred = framework.predict_one(x)
            latency.update((time.perf_counter_ns() - t_pred) / 1_000_000.0)
            prediction_diagnostics.update(pred)
''',
        '''            pred = framework.predict_one(x)
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
''',
        "Prequential probability capture",
    )

    replace_once(
        path,
        '''            if cfg.sensitive_attribute and cfg.sensitive_attribute in row.index:
                fairness.update(y, pred, row[cfg.sensitive_attribute])
''',
        '''            if cfg.sensitive_attribute and cfg.sensitive_attribute in row.index:
                fairness.update(
                    y,
                    pred,
                    row[cfg.sensitive_attribute],
                    y_proba=y_proba,
                )
''',
        "Calibration fairness update",
    )

    replace_once(
        path,
        '''            "measurement_incomplete": "partial",
            "failed": "failed",
        }
''',
        '''            "measurement_incomplete": "partial",
            "measurement_failed": "failed",
            "failed": "failed",
        }
''',
        "Sustainability failure status mapping",
    )

    replace_once(
        path,
        '''            country_iso=sustainability.get("country_iso"),
            backend=sustainability.get("measurement_backend"),
            hardware={
                "cpu": sustainability.get("cpu"),
                "logical_cpus": sustainability.get("logical_cpus"),
                "ram_gb": sustainability.get("ram_gb"),
                "gpu": sustainability.get("gpu"),
                "python": sustainability.get("python"),
                "codecarbon_version": sustainability.get("codecarbon_version"),
            },
''',
        '''            country_iso=sustainability.get("country_iso"),
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
            },
''',
        "Sustainability snapshot protocol fields",
    )


def patch_runner_validation():
    path = ROOT / "awareml" / "engine" / "runner.py"
    old = '''    if int(config.xai_max_rows) < 30:
        raise ValueError("xai_max_rows must be at least 30.")
'''
    new = '''    if int(config.xai_max_rows) < 30:
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
'''
    replace_once(path, old, new, "Phase-14 RunConfig validation")


def patch_analysis_init():
    path = ROOT / "awareml" / "analysis" / "__init__.py"
    text = path.read_text(encoding="utf-8")
    if "summarize_repeatability" in text:
        return
    backup(path)
    text = text.replace(
        "from .sustainability import SustainabilitySession\n",
        "from .sustainability import SustainabilitySession\n"
        "from .repeatability import summarize_repeatability, hardware_table, phase14_gate\n",
    )
    text = text.replace(
        '__all__ = ["SlidingFairness", "explain_framework", "SustainabilitySession"]',
        '__all__ = ["SlidingFairness", "explain_framework", "SustainabilitySession", '
        '"summarize_repeatability", "hardware_table", "phase14_gate"]',
    )
    path.write_text(text, encoding="utf-8")
    print("Patched: analysis exports")


def patch_specialist_ui():
    path = ROOT / "awareml" / "ui_v2" / "pages_specialist.py"

    replace_once(
        path,
        '''    "Predictive parity": "predictive_parity_diff",
    "Error-rate parity": "error_rate_gap",
}
''',
        '''    "Predictive parity": "predictive_parity_diff",
    "Error-rate parity": "error_rate_gap",
    "Group Brier-score gap": "group_brier_score_gap",
    "Group ECE gap": "group_ece_gap",
}
''',
        "Temporal calibration fairness selectors",
    )

    replace_once(
        path,
        '''        row = {"Framework": r.get("framework"), "Status": f.get("status"), "Window N": f.get("window_n")}
        for label, key in metric_map.items():
            row[label] = f.get(key)
        rows.append(row)
''',
        '''        row = {
            "Framework": r.get("framework"),
            "Status": f.get("status"),
            "Window N": f.get("window_n"),
            "Calibration status": f.get("calibration_status") or "unavailable",
            "Probability coverage": f.get("probability_coverage"),
            "Group Brier-score gap": f.get("group_brier_score_gap"),
            "Group ECE gap": f.get("group_ece_gap"),
            "Calibration reason": f.get("calibration_reason"),
        }
        for label, key in metric_map.items():
            row[label] = f.get(key)
        rows.append(row)
''',
        "Fairness Lab calibration table source",
    )

    replace_once(
        path,
        '''    st.dataframe(fairness_display, use_container_width=True, hide_index=True)
    if not common_labels:
''',
        '''    st.dataframe(fairness_display, use_container_width=True, hide_index=True)

    section(
        "Calibration fairness · Phase 14",
        "Probability-based audit evidence. Lower Brier/ECE gaps are better. "
        "Unavailable probabilities remain N/A and are never converted to zero.",
    )
    calibration_display = fair[[
        "Framework",
        "Calibration status",
        "Probability coverage",
        "Group Brier-score gap",
        "Group ECE gap",
        "Calibration reason",
    ]].copy()
    for column in [
        "Probability coverage",
        "Group Brier-score gap",
        "Group ECE gap",
    ]:
        calibration_display[column] = calibration_display[column].map(
            lambda value: "N/A"
            if pd.isna(value)
            else "{:.4f}".format(float(value))
        )
    st.dataframe(
        calibration_display,
        use_container_width=True,
        hide_index=True,
    )
    if not (fair["Calibration status"] == "ok").any():
        st.info(
            "No framework currently has sufficient valid probability evidence "
            "for group calibration fairness. This is an availability result, not zero disparity."
        )
    else:
        st.caption(
            "Group Brier-score gap and Group ECE gap are max-minus-min disparities "
            "across eligible sensitive groups in the current sliding window."
        )

    if not common_labels:
''',
        "Fairness Lab Phase-14 calibration section",
    )

    replace_once(
        path,
        '''            "RAM GB": s.get("ram_gb"),
        })
    sdf = pd.DataFrame(rows)
''',
        '''            "RAM GB": s.get("ram_gb"),
            "Physical CPUs": s.get("physical_cpus"),
            "Country": s.get("country_iso"),
            "Region": s.get("region"),
            "Carbon intensity gCO2/kWh": s.get(
                "carbon_intensity_g_per_kwh"
            ),
            "Warm-up s": s.get("warmup_sec"),
            "Repetition": s.get("repetition_id"),
            "Repetitions planned": s.get("repetitions_planned"),
            "Failure reason": s.get("measurement_failure_reason"),
        })
    sdf = pd.DataFrame(rows)

    st.info(
        "Phase 14 records CPU/GPU/RAM, country/region, CodeCarbon version, "
        "carbon intensity, measurement duration, warm-up, repetition metadata "
        "and failure reasons. Missing measurements remain N/A rather than zero."
    )
''',
        "Sustainability Lab protocol columns",
    )


def patch_advanced_ui():
    path = ROOT / "awareml" / "ui_v2" / "pages_advanced.py"
    text = path.read_text(encoding="utf-8")
    import_line = "from .phase14_hardening import phase14_hardening_page\n"
    if import_line not in text:
        backup(path)
        anchor = "from .page_utils import phase_pills\n"
        if anchor not in text:
            raise RuntimeError("Could not locate pages_advanced import anchor.")
        text = text.replace(anchor, anchor + import_line, 1)
        path.write_text(text, encoding="utf-8")

    replace_once(
        path,
        '''    labs = {
        "Decision Lab · observed post-run ranking": decision_lab_v2_page,
''',
        '''    labs = {
        "Phase 14 · Fairness + Sustainability Hardening": phase14_hardening_page,
        "Decision Lab · observed post-run ranking": decision_lab_v2_page,
''',
        "Phase-14 Advanced Research Lab",
    )


def main():
    if not (ROOT / "app.py").exists() or not (ROOT / "awareml").exists():
        raise RuntimeError("Run this installer from the AwareML_Extension project root.")
    if not PAYLOAD.exists():
        raise RuntimeError("phase14_payload is missing. Extract the complete ZIP first.")

    before = protected_hashes()
    BACKUP.mkdir(parents=True, exist_ok=True)

    copy_payload()
    patch_types()
    patch_records()
    patch_runner()
    patch_runner_validation()
    patch_analysis_init()
    patch_specialist_ui()
    patch_advanced_ui()

    after = protected_hashes()
    changed = [key for key in before if before.get(key) != after.get(key)]
    if changed:
        raise RuntimeError(
            "Protected frozen/model artifacts changed unexpectedly: {}".format(changed)
        )

    print("=" * 96)
    print("AwareML Phase 14 — Fairness + Sustainability Hardening: APPLIED")
    print("=" * 96)
    print("Backup:", BACKUP)
    print("Protected Phase-12/13/V3/V3.1/V2 evidence unchanged: True")
    print()
    print("Installed:")
    print("  • existing fairness criteria retained")
    print("  • Group Brier-score gap + Group ECE gap")
    print("  • explicit probability availability/coverage/reason")
    print("  • CPU/GPU/RAM/country/region/CodeCarbon/carbon-intensity metadata")
    print("  • warm-up + repetition + measurement-failure metadata")
    print("  • Runtime/Energy/CO2 mean ± sample SD utilities")
    print("  • Phase-14 Research Gate workspace")
    print("  • repeatability runner + methodology docs")
    print()
    print("Run next:")
    print(r"  pytest -q .\tests\test_phase14_hardening.py")
    print(r"  python -m scripts.validate_phase14_hardening")
    print(r"  streamlit run app.py")
    print("=" * 96)


if __name__ == "__main__":
    main()
