from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.objective_benchmark import run_paraphrase_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    m = run_paraphrase_benchmark(ROOT, resume=args.resume)
    print("=" * 72)
    print("AwareML Phase 12 paraphrase robustness: COMPLETE")
    print("=" * 72)
    print("Families:", m["families"])
    print("Paraphrase evaluations:", m["paraphrase_evaluations"])
    print("Base exact match:", "{:.4f}".format(m["base_exact_match_rate"]))
    print("Paraphrase exact match:", "{:.4f}".format(m["paraphrase_exact_match_rate"]))
    print(
        "Prediction consistency with base:",
        "{:.4f}".format(m["paraphrase_prediction_consistency_rate"]),
    )
    print("Set-change rate:", "{:.4f}".format(m["set_change_rate"]))
    print("=" * 72)


if __name__ == "__main__":
    main()
