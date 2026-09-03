from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.objective_benchmark import run_adversarial_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    r = run_adversarial_benchmark(ROOT, resume=args.resume)
    print("=" * 72)
    print("AwareML Phase 12 adversarial evaluation: COMPLETE")
    print("=" * 72)
    print("Cases:", r["n"])
    print("Failure taxonomy:", r["taxonomy_counts"])
    print("=" * 72)


if __name__ == "__main__":
    main()
