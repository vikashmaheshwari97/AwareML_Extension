
from __future__ import annotations

import json
from pathlib import Path

from awareml.studies.information_seeking_morten_v2 import (
    coding_scheme,
    protocol,
)


ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "awareml" / "ui_v2" / "phase17_morten_v2.py"


def main() -> int:
    study_protocol = protocol()
    coding = coding_scheme()
    ui = UI.read_text(encoding="utf-8")

    measurements = set(study_protocol.get("measurements") or [])
    checks = {
        "target_is_12":
            (study_protocol.get("participant_target") or {}).get(
                "final_target"
            ) == 12,
        "ml_automl_background_explicit":
            "ML / AutoML"
            in str(
                (study_protocol.get("eligibility") or {}).get(
                    "required_background"
                )
            ),
        "explicit_track1_scenario_context":
            "Track-1" in json.dumps(
                study_protocol.get("study_context") or {}
            ),
        "repeat_evidence_access_measured":
            "repeated_evidence_access_count" in measurements,
        "total_decision_time_measured":
            "total_time_to_final_decision" in measurements,
        "complete_timestamped_sequence_measured":
            "complete_timestamped_interaction_sequence" in measurements,
        "manual_coding_authoritative":
            "Manual qualitative coding is authoritative"
            in str(coding.get("principle")),
        "evidence_revisiting_code_present":
            any(
                item.get("id") == "evidence_revisiting"
                for item in coding.get("codes") or []
            ),
        "ui_logs_area_visit_number":
            "area_visit_number" in ui,
        "ui_records_total_decision_time":
            "total_time_to_final_decision_sec" in ui,
        "ui_uses_verbalization_prompts":
            "Verbalization prompts for facilitated sessions" in ui,
        "no_think_aloud_label":
            "Think-aloud prompts" not in ui,
        "no_statistical_saturation_claim":
            "statistical saturation" not in json.dumps(
                study_protocol
            ).lower(),
        "exploratory_clustering_present":
            "exploratory_clusters" in ui,
    }

    print(json.dumps(checks, indent=2, sort_keys=True))
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print(
            "\nInformation-Seeking v2 validation failed: {}".format(
                ", ".join(failed)
            )
        )
        return 2

    print("\nInformation-Seeking v2 validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
