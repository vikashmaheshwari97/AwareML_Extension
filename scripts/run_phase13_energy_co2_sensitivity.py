from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.recommender_validation import run_energy_co2_sensitivity


def main() -> None:
    result = run_energy_co2_sensitivity(ROOT)
    print("=" * 72)
    print("AwareML Phase 13.2 Energy vs CO2 sensitivity: COMPLETE")
    print("=" * 72)
    print(result["summary"].to_string(index=False))
    print()
    print(result["pairwise"].to_string(index=False))
    print()
    decision = result["decision"]
    print("Development Spearman rho(Energy, CO2):", decision["development_spearman_energy_co2"])
    print("Decision:", decision["decision_code"])
    print(decision["journal_decision"])
    print("Held-out 23 dataset contents used: False")
    print("=" * 72)


if __name__ == "__main__":
    main()
