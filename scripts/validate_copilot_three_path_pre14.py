from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    pages = (ROOT / "awareml/ui_v2/pages_copilot.py").read_text(encoding="utf-8")
    ui = (ROOT / "awareml/ui_v2/copilot_three_path.py").read_text(encoding="utf-8")
    protocol = json.loads(
        (ROOT / "data/journal/recommender_final_test_protocol_v1/design/predeclared_protocol.json").read_text(encoding="utf-8")
    )

    checks = {
        "three_top_level_tabs": all(x in pages for x in [
            "1 · Goal Copilot",
            "2 · Historical Preference Prior · No dataset",
            "3 · Dataset-aware ML Recommender V2",
        ]),
        "historical_prior_not_ml": "Historical aggregation · not machine learning" in ui,
        "dataset_aware_true_ml": "actual learned meta-recommender" in ui,
        "winner_sensitivity_visible": "Why can the same framework keep winning?" in ui,
        "meta_log_audit_present": (ROOT / "awareml/recommender/meta_logs_v2_audit.py").exists(),
        "heldout_23_predeclared": protocol.get("held_out_dataset_count") == 23,
        "expected_345_runs": protocol.get("expected_framework_runs") == 345,
        "same_three_seeds": protocol.get("seeds") == [42, 43, 44],
        "final_protocol_matches_development": (
            protocol.get("run_protocol", {}).get("max_samples") == 30000
            and protocol.get("run_protocol", {}).get("window_size") == 1000
            and protocol.get("run_protocol", {}).get("time_budget_sec") == 60
        ),
    }

    print("=" * 88)
    print("AwareML three-path Copilot validation")
    print("=" * 88)
    failed = []
    for name, ok in checks.items():
        print("{:<48} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)
    print("=" * 88)
    if failed:
        print("FAILED:", ", ".join(failed))
        raise SystemExit(1)
    print("Three-path Copilot validation: PASS")

if __name__ == "__main__":
    main()
