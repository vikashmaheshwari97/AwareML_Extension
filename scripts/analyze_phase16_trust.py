from __future__ import annotations

import argparse
import json
from pathlib import Path

from awareml.studies.trust import DEFAULT_DB_PATH, DEFAULT_PROTOCOL_PATH, Phase16Store
from awareml.studies.trust_analysis import analyze_phase16_store, write_analysis_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase-16 trust-calibration responses.")
    parser.add_argument("--mode", choices=["pilot", "final"], default="final")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/phase16/analysis"),
    )
    args = parser.parse_args()

    store = Phase16Store(args.db)
    result = analyze_phase16_store(store, args.mode, args.protocol)
    run_dir = args.output_dir / args.mode
    paths = write_analysis_outputs(result, run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("\nWrote:")
    for name, path in paths.items():
        print("  {}: {}".format(name, path))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
