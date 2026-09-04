from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.recommender_validation import validate_phase13_complete


def main() -> None:
    result = validate_phase13_complete(ROOT)
    print("=" * 72)
    print("AwareML Phase 13 COMPLETE validation: PASS")
    print("=" * 72)
    print("Artifact:", result["artifact"])
    print("SHA256:", result["sha256"])
    print("Development datasets:", result["development_dataset_count"])
    print("Candidate meta-models:", result["candidate_meta_model_count"])
    print("Baseline verdict:", result["baseline_conclusion"]["verdict"])
    print("Energy/CO2 decision:", result["energy_co2_decision"]["decision_code"])
    print("Near-Pareto spec:", result["near_pareto"]["spec_id"])
    print("Near-Pareto epsilon:", result["near_pareto"]["epsilon"])
    print("Held-out 23 dataset contents used: False")
    print("Release status:", result["release_status"])
    print("=" * 72)


if __name__ == "__main__":
    main()
