from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    path = ROOT / relative
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main():
    registry = read("awareml/analysis/repeatability_registry.py")
    repeatability = read("awareml/analysis/repeatability.py")
    runner = read("scripts/run_phase14_repeatability.py")
    integrated = read("awareml/ui_v2/phase14_integrated_sections.py")
    specialist = read("awareml/ui_v2/pages_specialist.py")
    pages = read("awareml/ui/pages.py")
    protocol = read("data/journal/fairness_sustainability_hardening_v1/design/protocol.json")

    checks = {
        "paper_ready_gate_five": (
            "PAPER_READY_MIN_REPETITIONS = 5" in registry
        ),
        "dataset_content_hash": (
            "canonical_dataframe_sha256" in registry
        ),
        "identity_uses_hash_target_sensitive_positive": all(
            token in registry
            for token in [
                "dataset_content_sha256",
                '"target": str(target)',
                '"sensitive_attribute": sensitive',
                '"positive_label": positive',
            ]
        ),
        "timestamped_run_directories": (
            "run_dir = identity_root / run_id" in runner
            and "while run_dir.exists()" in runner
        ),
        "registry_preserves_runs": (
            "register_run(" in runner
            and "list_dataset_runs" in integrated
            and "find_latest_matching_run" in integrated
        ),
        "generic_dataset_runner": (
            "--csv" in runner
            and "--target" in runner
            and "--sensitive" in runner
            and "--positive-label" in runner
        ),
        "five_default_seeds": (
            'default="42,43,44,45,46"' in runner
        ),
        "per_run_hash_provenance": all(
            token in runner
            for token in [
                "dataset_file_sha256",
                "dataset_content_sha256",
                "dataset_identity_key",
                '"source_sha256": dataset_file_hash',
                '"content_sha256": dataset_content_hash',
            ]
        ),
        "streamlit_stores_active_hash": (
            '["dataset_content_sha256"] = canonical_dataframe_sha256(df)'
            in pages
        ),
        "ui_exact_registry_match": (
            "find_latest_matching_run" in integrated
            and "dataset_content_sha256=context" in integrated
        ),
        "beginner_no_evidence_message": (
            "No saved repeatability evidence matches this exact dataset"
            in integrated
        ),
        "all_saved_datasets_registry_ui": (
            "Repeatability registry · all saved datasets" in integrated
        ),
        "fairness_winner_degenerate_exclusion": (
            "Lowest interpretable mean disparity" in specialist
            and "EXCLUDED · degenerate predictions" in specialist
        ),
        "raw_fairness_not_removed": (
            "Comparable mean gap" in specialist
            and "Worst available gap" in specialist
        ),
        "repeatability_summary_uses_formal_default": (
            "PAPER_READY_MIN_REPETITIONS" in repeatability
        ),
        "protocol_frozen_to_five_repetitions": (
            '"paper_ready_minimum_repetitions": 5' in protocol
            and '"overwrite_policy": "never overwrite previous run evidence"' in protocol
        ),
    }

    print("=" * 96)
    print("AwareML Phase-14 final-freeze validation")
    print("=" * 96)

    failed = []
    for name, ok in checks.items():
        print("{:<64} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)

    print("=" * 96)

    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))

    print("Phase-14 final-freeze implementation validation: PASS")
    print(
        "Paper-ready empirical status still depends on an exact matching "
        "dataset/configuration having >=5 measured repetitions."
    )


if __name__ == "__main__":
    main()
