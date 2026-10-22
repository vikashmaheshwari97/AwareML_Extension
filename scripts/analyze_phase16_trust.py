
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from awareml.studies.trust_morten_v2 import (
    DEFAULT_DB_PATH,
    Phase16V2Store,
    analyze_v2,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze revised Phase-16 Trust Calibration v2 responses."
    )
    parser.add_argument("--mode", choices=["pilot", "final"], default="final")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/phase16/analysis_v2"),
    )
    args = parser.parse_args()

    store = Phase16V2Store(args.db)
    result = analyze_v2(store, args.mode)

    run_dir = args.output_dir / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = run_dir / "analysis_results.json"
    analysis_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    pair_path = run_dir / "trial_family_exposure.csv"
    pd.DataFrame(result.get("pair_coverage") or []).to_csv(
        pair_path, index=False
    )

    framework_path = run_dir / "framework_exposure.csv"
    pd.DataFrame(result.get("framework_coverage") or []).to_csv(
        framework_path, index=False
    )

    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("\nWrote:")
    print("  analysis: {}".format(analysis_path))
    print("  trial exposure: {}".format(pair_path))
    print("  framework exposure: {}".format(framework_path))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
