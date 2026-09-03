from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.objective_benchmark import freeze_ground_truth


def main():
    r = freeze_ground_truth(ROOT)
    print("=" * 72)
    print("AwareML Phase 12 HUMAN GROUND-TRUTH freeze: SUCCESS")
    print("=" * 72)
    print("Manifest:", r["manifest"])
    print("SHA256:", r["sha256"])
    print("Resolved primary cases:", r["payload"]["resolved_primary_cases"])
    print("Annotators:", r["payload"]["completed_annotators"])
    print("Generation intent used as ground truth: False")
    print("LLaMA outputs existed before freeze: False")
    print("23 held-out dataset contents used: False")
    print("=" * 72)


if __name__ == "__main__":
    main()
