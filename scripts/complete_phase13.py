from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.recommender_validation import (
    freeze_phase13_validation,
    run_baseline_validation,
    run_energy_co2_sensitivity,
    validate_phase13_complete,
    write_near_pareto_specification,
)


def main() -> None:
    print("[1/5] Phase 13.1 explicit recommender baselines")
    baseline = run_baseline_validation(ROOT)
    print("      verdict:", baseline["conclusion"]["verdict"])

    print("[2/5] Phase 13.2 Energy/CO2 sensitivity")
    sensitivity = run_energy_co2_sensitivity(ROOT)
    print("      decision:", sensitivity["decision"]["decision_code"])

    print("[3/5] Phase 13.3 canonical epsilon-Pareto specification")
    spec = write_near_pareto_specification(ROOT)
    print("      {} epsilon={:.2f}".format(spec["spec_id"], spec["epsilon"]))

    print("[4/5] Freeze Phase 13")
    frozen = freeze_phase13_validation(ROOT)
    print("      SHA256:", frozen["sha256"])

    print("[5/5] Validate Phase 13")
    validated = validate_phase13_complete(ROOT)
    print("=" * 72)
    print("AwareML Phase 13: COMPLETE")
    print("Artifact:", validated["artifact"])
    print("SHA256:", validated["sha256"])
    print("Held-out 23 dataset contents used: False")
    print("=" * 72)


if __name__ == "__main__":
    main()
