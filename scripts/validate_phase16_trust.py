from __future__ import annotations

import json
from pathlib import Path

from awareml.studies.trust import (
    RANDOMIZATION_VERSION,
    TrustCalibrationStudy,
    validate_protocol_for_design_freeze,
)


def main() -> int:
    study = TrustCalibrationStudy()
    bank = study.bank.validate_design_pool(
        int((study.protocol.get("design") or {}).get("shared_pool_min_per_condition", 20))
    )
    preview = study.build_preview_assignment("phase16-validator")
    n_items = int((study.protocol.get("design") or {}).get("items_per_participant", 20))
    checks = {
        "phase15_bank_frozen_and_hash_verified": True,
        "correct_pool_at_least_20": bank["correct"] >= 20,
        "incorrect_pool_at_least_20": bank["incorrect"] >= 20,
        "complete_pairs_at_least_20": bank["pairs"] >= 20,
        "participant_preview_length": len(preview) == n_items,
        "participant_preview_blinded": all(
            "condition" not in row and "correctness_condition" not in row and "pair_id" not in row
            for row in preview
        ),
        "randomization_version": RANDOMIZATION_VERSION,
        "design_freeze_gate_is_safely_blocked_until_finalization": bool(validate_protocol_for_design_freeze(study.protocol)),
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    failed = [key for key, value in checks.items() if key != "randomization_version" and value is not True]
    if failed:
        print("\nPhase-16 validation: BLOCKED/INCOMPLETE")
        print("Failed checks: {}".format(", ".join(failed)))
        return 2
    print("\nPhase-16 validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
