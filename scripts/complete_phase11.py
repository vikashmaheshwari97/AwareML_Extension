from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    print("\n>>> " + " ".join(args[1:]))
    proc = subprocess.run(args, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    run([
        sys.executable,
        str(ROOT / "scripts" / "validate_phase11_copilot_v2.py"),
        "--live-ollama",
    ])

    run([
        sys.executable,
        str(ROOT / "scripts" / "freeze_phase11_copilot_v2.py"),
    ])

    run([
        sys.executable,
        str(ROOT / "scripts" / "validate_phase11_complete.py"),
    ])

    print("\n" + "=" * 72)
    print("ALL PHASE 11 STEPS PASSED")
    print("Copilot Objective-Selection V2 is frozen.")
    print("Phase-12 benchmark data has not been created yet.")
    print("=" * 72)


if __name__ == "__main__":
    main()
