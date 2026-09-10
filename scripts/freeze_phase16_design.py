from __future__ import annotations

import argparse
from pathlib import Path

from awareml.studies.trust import DEFAULT_PROTOCOL_PATH, ProtocolGateError
from awareml.studies.trust_freeze import DESIGN_FREEZE_ROOT, freeze_phase16_design


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the finalized Phase-16 human-study design.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output-root", type=Path, default=DESIGN_FREEZE_ROOT)
    parser.add_argument("--stimulus-root", type=Path, default=None)
    args = parser.parse_args()
    try:
        manifest = freeze_phase16_design(args.protocol, args.output_root, args.stimulus_root)
    except ProtocolGateError as exc:
        print("PHASE-16 DESIGN FREEZE: BLOCKED")
        print(str(exc))
        return 2
    print("PHASE-16 DESIGN FREEZE: PASS")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
