from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.objective_benchmark import freeze_design


def main():
    result = freeze_design(ROOT)
    print("=" * 72)
    print("AwareML Phase 12 benchmark DESIGN freeze: SUCCESS")
    print("=" * 72)
    print("Manifest:", result["manifest"])
    print("SHA256:", result["sha256"])
    print("Candidate scenarios:", result["summary"]["candidate_count"])
    print("k' counts:", result["summary"]["k_prime_design_counts"])
    print("Ground truth created yet: False")
    print("LLaMA benchmark executed yet: False")
    print("23 held-out dataset contents used: False")
    print("=" * 72)


if __name__ == "__main__":
    main()
