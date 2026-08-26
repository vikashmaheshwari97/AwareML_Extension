from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label, command):
    print("\n>>> {}".format(label), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def main():
    run("Phase 9 backend/UI validation", [
        sys.executable, str(ROOT / "scripts" / "validate_phase9_ui.py")
    ])
    run("Phase 9 freeze", [
        sys.executable, str(ROOT / "scripts" / "freeze_phase9_ui.py")
    ])
    run("Phase 9 final validation", [
        sys.executable, str(ROOT / "scripts" / "validate_phase9_complete.py")
    ])
    print("\nALL PHASE 9 STEPS PASSED")


if __name__ == "__main__":
    main()
