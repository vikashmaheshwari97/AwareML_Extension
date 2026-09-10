from __future__ import annotations

import argparse
from pathlib import Path

from awareml.studies.trust import DEFAULT_DB_PATH, DEFAULT_PROTOCOL_PATH, ProtocolGateError
from awareml.studies.trust_freeze import FINAL_FREEZE_ROOT, freeze_phase16_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze completed Phase-16 trust-calibration evidence.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output-root", type=Path, default=FINAL_FREEZE_ROOT)
    parser.add_argument(
        "--analysis-script",
        type=Path,
        default=Path("scripts/analyze_phase16_trust.py"),
    )
    args = parser.parse_args()
    try:
        manifest = freeze_phase16_results(
            args.db,
            args.protocol,
            args.output_root,
            args.analysis_script,
        )
    except ProtocolGateError as exc:
        print("PHASE-16 FINAL FREEZE: BLOCKED")
        print(str(exc))
        return 2
    print("PHASE-16 FINAL FREEZE: PASS")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
