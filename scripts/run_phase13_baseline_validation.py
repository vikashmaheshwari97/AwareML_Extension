from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.recommender_validation import run_baseline_validation


def main() -> None:
    result = run_baseline_validation(ROOT)
    table = result["table"]
    conclusion = result["conclusion"]
    print("=" * 72)
    print("AwareML Phase 13.1 recommender baseline validation: COMPLETE")
    print("=" * 72)
    print(table.to_string(index=False))
    print()
    print("Question:", conclusion["question"])
    print("Verdict:", conclusion["verdict"])
    print("Best simple baseline:", conclusion["best_simple_baseline"])
    print("Interpretation:", conclusion["interpretation"])
    print("Held-out 23 dataset contents used: False")
    print("=" * 72)


if __name__ == "__main__":
    main()
