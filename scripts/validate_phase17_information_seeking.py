from __future__ import annotations

import json
from pathlib import Path

from awareml.studies.information_seeking import (
    CATEGORY_LABELS,
    classify_follow_up,
    derive_representative_pattern,
)
from awareml.studies.information_seeking_grounding import (
    detect_requested_topics,
    grounded_information_answer,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data" / "journal" / "information_seeking_v1" / "design" / "protocol.json"
CODING = ROOT / "data" / "journal" / "information_seeking_v1" / "design" / "coding_scheme.json"
UI = ROOT / "awareml" / "ui_v2" / "phase17_information_seeking.py"


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    coding = json.loads(CODING.read_text(encoding="utf-8"))
    ui = UI.read_text(encoding="utf-8")

    checks = {
        "central_research_question_present":
            protocol.get("central_rq") == "When do users ask for more information instead of accepting the initial explanation?",
        "suggested_sample_10_to_15":
            (protocol.get("participant_target") or {}).get("suggested_min") == 10
            and (protocol.get("participant_target") or {}).get("suggested_max") == 15,
        "all_required_behavior_fields_present":
            all(
                key in (protocol.get("measurements") or [])
                for key in [
                    "whether_follow_up_occurred",
                    "number_of_follow_ups",
                    "first_follow_up_category",
                    "time_before_first_follow_up",
                    "evidence_viewed",
                    "recommendation_accepted_overridden_rejected",
                ]
            ),
        "classifier_is_assist_only":
            (protocol.get("classifier") or {}).get("role") == "analysis_assist_only"
            and not bool((protocol.get("classifier") or {}).get("qualitative_coding_replacement")),
        "qualitative_coding_scheme_present":
            len(coding.get("codes") or []) >= 5,
        "five_core_categories_present":
            all(
                key in CATEGORY_LABELS
                for key in [
                    "evidence_request",
                    "explanation_probe",
                    "challenge",
                    "counterfactual_or_comparison",
                    "clarification",
                ]
            ),
        "classifier_smoke_test":
            classify_follow_up("Show me the evidence") == "evidence_request"
            and classify_follow_up("Why is this recommended?") == "explanation_probe"
            and classify_follow_up("Are you sure this is right?") == "challenge"
            and classify_follow_up("Compare it with another framework") == "counterfactual_or_comparison"
            and classify_follow_up("What does this metric mean?") == "clarification",
        "multi_topic_detection":
            detect_requested_topics("Explain OAML accuracy, fairness, and sustainability") == [
                "performance", "fairness", "sustainability"
            ],
        "representative_pattern_smoke_test":
            derive_representative_pattern(0, None, [], "accept") == "Accept without follow-up",
        "participant_logs_all_required_fields":
            all(
                marker in ui
                for marker in [
                    '"follow_up_occurred"',
                    '"follow_up_count"',
                    '"first_follow_up_category"',
                    '"time_before_first_follow_up_sec"',
                    '"evidence_viewed"',
                    '"recommendation_decision"',
                ]
            ),
        "researcher_manual_coding_present":
            "Manual qualitative themes" in ui
            and "qualitative_code" in ui
            and "classifier_not_authoritative" in ui,
        "participant_classifier_label_hidden":
            "behavior category:" not in ui
            and "classifier_role" in ui,
        "standardized_context_supported":
            "save_study_context" in ui
            and "resolve_study_context" in ui
            and "Participants do not need to run the dashboard first" in ui,
        "final_mode_safely_gated":
            "AWAREML_INFORMATION_SEEKING_FINAL_ARMED" in ui
            and "FINAL_DESIGN_MANIFEST.exists()" in ui,
    }

    print(json.dumps(checks, indent=2))
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("Information-Seeking validation failed: {}".format(", ".join(failed)))
    print("\nInformation-Seeking validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
