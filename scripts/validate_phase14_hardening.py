from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(path):
    p = ROOT / path
    return p.read_text(encoding="utf-8") if p.exists() else ""


def main():
    advanced = text("awareml/ui_v2/pages_advanced.py")
    specialist = text("awareml/ui_v2/pages_specialist.py")
    integrated = text("awareml/ui_v2/phase14_integrated_sections.py")
    fairness = text("awareml/analysis/fairness.py")
    sustain = text("awareml/analysis/sustainability.py")
    runner = text("scripts/run_phase14_repeatability.py")

    checks = {
        "separate_phase14_workspace_removed": (
            "Phase 14 · Fairness + Sustainability Hardening" not in advanced
        ),
        "fairness_lab_retained": '"Fairness Lab": fairness_v2_page' in advanced,
        "sustainability_lab_retained": (
            '"Sustainability Lab": sustainability_v2_page' in advanced
        ),
        "fairness_phase14_integrated": (
            "render_phase14_fairness_details(results)" in specialist
        ),
        "sustainability_phase14_integrated": (
            "render_phase14_sustainability_details(results)" in specialist
        ),
        "per_group_calibration_ui": (
            "Per-group calibration and hard-label details" in integrated
        ),
        "repeatability_ui": "Dataset-specific repeatability · Phase 14" in integrated,
        "brier_gap_implemented": "group_brier_score_gap" in fairness,
        "ece_gap_implemented": "group_ece_gap" in fairness,
        "sustainability_protocol_metadata": all(
            token in sustain
            for token in [
                "carbon_intensity_g_per_kwh",
                "warmup_sec",
                "repetition_id",
                "measurement_failure_reason",
            ]
        ),
        "downloads_dataset_resolution": (
            'Path.home() / "Downloads"' in runner
        ),
        "dutch_schema_defaults": all(
            token in runner
            for token in [
                'DUTCH_TARGET = "occupation_binary"',
                'DUTCH_SENSITIVE = "sex"',
            ]
        ),
    }

    print("=" * 92)
    print("AwareML Phase-14 integrated-labs validation")
    print("=" * 92)
    failed = []
    for name, ok in checks.items():
        print("{:<58} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)
    print("=" * 92)
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))
    print("Phase-14 integrated-labs validation: PASS")


if __name__ == "__main__":
    main()
