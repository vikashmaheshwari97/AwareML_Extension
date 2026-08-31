from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(
        description="Run all Phase-10 completion gates."
    )
    parser.add_argument("--ollama-base-url", default=None)
    args = parser.parse_args()

    run([
        sys.executable,
        str(ROOT / "scripts" / "validate_phase10_protocol.py"),
        "--preflight",
    ])

    freeze = [
        sys.executable,
        str(ROOT / "scripts" / "freeze_phase10_protocol.py"),
    ]
    if args.ollama_base_url:
        freeze.extend(["--ollama-base-url", args.ollama_base_url])
    run(freeze)

    final_validation = [
        sys.executable,
        str(ROOT / "scripts" / "validate_phase10_protocol.py"),
    ]
    if args.ollama_base_url:
        final_validation.extend(["--ollama-base-url", args.ollama_base_url])
    run(final_validation)

    print("\n" + "=" * 72)
    print("ALL PHASE 10 STEPS PASSED")
    print("Journal Experimental Protocol v1 is frozen.")
    print("The 23 held-out dataset contents were not used.")
    print("=" * 72)


if __name__ == "__main__":
    main()
