from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.objective_benchmark import prepare_annotation_packets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite blank annotation packets. Do not use after annotators start.",
    )
    args = parser.parse_args()

    result = prepare_annotation_packets(ROOT, force=args.force)
    print("=" * 72)
    print("AwareML Phase 12 annotation packets: READY")
    print("=" * 72)
    print("Generated scenarios retained:", result["generated_retained"])
    print("Human-written scenarios:", result["human_written"])
    print("Final annotation pool:", result["final_annotation_pool"])
    print("Packets:")
    for path in result["annotation_files"]:
        print(" ", path)
    print("=" * 72)


if __name__ == "__main__":
    main()
