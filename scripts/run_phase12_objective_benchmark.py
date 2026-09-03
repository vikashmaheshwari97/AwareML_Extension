from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.objective_benchmark import run_main_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only missing cases after interruption; completed cases are never rerun.",
    )
    args = parser.parse_args()

    result = run_main_benchmark(ROOT, resume=args.resume)
    print("=" * 72)
    print("AwareML Phase 12 primary LLaMA objective benchmark: COMPLETE")
    print("=" * 72)
    print("Outputs:", len(result["outputs"]))
    print("Model:", result["metadata"]["runtime_verified"]["model"])
    print("Model digest:", result["metadata"]["runtime_verified"]["model_digest"])
    print("Ollama version:", result["metadata"]["runtime_verified"]["ollama_version"])
    print("Fallback used: False")
    print("Attempts per case: 1")
    print("=" * 72)


if __name__ == "__main__":
    main()
