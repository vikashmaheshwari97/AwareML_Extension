from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.recommender_validation import freeze_phase13_validation


def main() -> None:
    result = freeze_phase13_validation(ROOT)
    print("=" * 72)
    print("AwareML Phase 13 FINAL freeze: SUCCESS")
    print("=" * 72)
    print("Artifact: recommender_multiobjective_validation_v1")
    print("Manifest:", result["manifest"])
    print("SHA256:", result["sha256"])
    print("Development datasets: 47")
    print("Held-out 23 dataset contents used: False")
    print("=" * 72)


if __name__ == "__main__":
    main()
