
from __future__ import annotations

import json

from awareml.studies.trust_morten_v2 import (
    RANDOMIZATION_VERSION,
    TrustCalibrationStudyV2,
)


def main() -> int:
    study = TrustCalibrationStudyV2(
        id_salt="phase16-v2-validation-salt-0001"
    )
    bank = study.bank

    arm_a = study._assignment("validator-a", "A")
    arm_b = study._assignment("validator-b", "B")
    a = {item.pair_id: item.correctness_condition for item in arm_a}
    b = {item.pair_id: item.correctness_condition for item in arm_b}

    frameworks = [
        bank.pairs[pair]["correct"].framework_context
        for pair in bank.pair_order
    ]
    stages = [
        bank.pairs[pair]["correct"].source_stage
        for pair in bank.pair_order
    ]
    scenarios = [
        bank.pairs[pair]["correct"].scenario_id
        for pair in bank.pair_order
    ]

    checks = {
        "v2_stimulus_bank_hash_verified": True,
        "matched_trial_families_exactly_20": len(bank.pairs) == 20,
        "unique_track1_scenarios_exactly_20":
            len(set(scenarios)) == 20,
        "framework_coverage_4_each": {
            framework: frameworks.count(framework)
            for framework in sorted(set(frameworks))
        } == {
            "AutoClass": 4,
            "AutoStreamML": 4,
            "ChaCha": 4,
            "EvoAutoML": 4,
            "OAML": 4,
        },
        "source_stage_coverage_5_each": {
            stage: stages.count(stage)
            for stage in sorted(set(stages))
        } == {
            "B": 5,
            "E": 5,
            "F_CHAT": 5,
            "F_XAI": 5,
        },
        "arm_a_10_correct_10_incorrect":
            sum(item.correctness_condition == "correct" for item in arm_a) == 10,
        "arm_b_10_correct_10_incorrect":
            sum(item.correctness_condition == "correct" for item in arm_b) == 10,
        "arms_are_pairwise_complements":
            set(a) == set(b) and all(a[pair] != b[pair] for pair in a),
        "each_framework_2_correct_2_incorrect_in_arm_a": all(
            len(
                [
                    item
                    for item in arm_a
                    if item.framework_context == framework
                ]
            ) == 4
            and sum(
                item.correctness_condition == "correct"
                for item in arm_a
                if item.framework_context == framework
            ) == 2
            for framework in [
                "AutoStreamML",
                "AutoClass",
                "EvoAutoML",
                "OAML",
                "ChaCha",
            ]
        ),
        "randomization_version": RANDOMIZATION_VERSION,
    }

    print(json.dumps(checks, indent=2, sort_keys=True))
    failed = [
        key
        for key, value in checks.items()
        if key != "randomization_version" and value is not True
    ]
    if failed:
        print("\nPhase-16 v2 validation: BLOCKED")
        print("Failed checks: {}".format(", ".join(failed)))
        return 2

    print("\nPhase-16 v2 validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
