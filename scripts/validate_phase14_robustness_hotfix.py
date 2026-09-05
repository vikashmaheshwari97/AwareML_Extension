from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative_path):
    path = ROOT / relative_path
    return (
        path.read_text(encoding="utf-8")
        if path.exists()
        else ""
    )


def main():
    fairness = read("awareml/analysis/fairness.py")
    sustainability = read(
        "awareml/analysis/sustainability.py"
    )
    autostream = read(
        "awareml/frameworks/autostreamml.py"
    )
    specialist = read(
        "awareml/ui_v2/pages_specialist.py"
    )
    integrated = read(
        "awareml/ui_v2/phase14_integrated_sections.py"
    )
    repeat_runner = read(
        "scripts/run_phase14_repeatability.py"
    )
    legacy_ui = read("awareml/ui/pages.py")
    engine = read("awareml/engine/runner.py")

    checks = {
        "degenerate_prediction_diagnostic": (
            "prediction_behavior_status"
            in fairness
        ),
        "constant_probability_diagnostic": (
            "probability_behavior_status"
            in fairness
        ),
        "zero_gap_preserved_with_warning": (
            "structurally trivial"
            in fairness
        ),
        "autostream_numeric_warning_guard": (
            "divide by zero encountered in log"
            in autostream
            and "invalid value encountered in divide"
            in autostream
        ),
        "autostream_probability_validation": (
            "_validated_probability_dict"
            in autostream
        ),
        "exclusive_codecarbon": (
            "allow_multiple_runs=False"
            in sustainability
        ),
        "carbon_intensity_unit_fix": (
            "derived_from_measured_co2_and_energy"
            in sustainability
            and 'getattr(final, "emissions_rate"'
            not in sustainability
        ),
        "single_integrated_calibration_section": (
            specialist.count(
                'Calibration fairness · Phase 14'
            )
            == 0
            and "render_phase14_fairness_details(results)"
            in specialist
        ),
        "artifact_repeatability_loader": (
            "find_latest_matching_run" in integrated
            and "list_dataset_runs" in integrated
            and "dataset_content_sha256" in integrated
        ),
        "repeatability_streamlit_preflight": (
            "_active_streamlit_processes"
            in repeat_runner
        ),
        "repeatability_profile_defaults": (
            "dutch_census_test_profile.json"
            in repeat_runner
        ),
        "runstudio_positive_label_profile": (
            "_awareml_positive_profile_key"
            in legacy_ui
        ),
        "fairness_threshold_wired": (
            "degenerate_prediction_threshold="
            in engine
        ),
        "carbon_source_persisted": (
            "carbon_intensity_source"
            in engine
        ),
    }

    print("=" * 92)
    print(
        "AwareML Phase-14 robustness hotfix validation"
    )
    print("=" * 92)

    failed = []

    for name, ok in checks.items():
        print(
            "{:<58} {}".format(
                name,
                "PASS" if ok else "FAIL",
            )
        )
        if not ok:
            failed.append(name)

    print("=" * 92)

    if failed:
        raise SystemExit(
            "FAILED: " + ", ".join(failed)
        )

    print(
        "Phase-14 robustness hotfix validation: PASS"
    )


if __name__ == "__main__":
    main()
