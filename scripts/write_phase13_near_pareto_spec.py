from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.recommender_validation import write_near_pareto_specification


def main() -> None:
    spec = write_near_pareto_specification(ROOT)
    print("=" * 72)
    print("AwareML Phase 13.3 canonical near-Pareto specification: COMPLETE")
    print("=" * 72)
    print("Specification:", spec["spec_id"])
    print("epsilon:", spec["epsilon"])
    print("Normalization:", spec["normalization_id"])
    print("Space:", spec["space"])
    print("=" * 72)


if __name__ == "__main__":
    main()
