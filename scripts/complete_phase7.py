from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label, command):
    print(
        "\n>>> {}".format(label),
        flush=True,
    )
    subprocess.run(
        command,
        cwd=str(ROOT),
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-ollama",
        action="store_true",
        help=(
            "Require one live local-Ollama "
            "validation before freezing Phase 7."
        ),
    )
    args = parser.parse_args()

    validate = [
        sys.executable,
        str(
            ROOT
            / "scripts"
            / "validate_phase7_copilot.py"
        ),
    ]
    if args.with_ollama:
        validate.extend(
            [
                "--ollama",
                "--require-ollama",
            ]
        )

    run(
        "Phase 7 backend validation",
        validate,
    )
    run(
        "Phase 7 freeze",
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "freeze_phase7_copilot.py"
            ),
        ],
    )
    run(
        "Phase 7 final validation",
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "validate_phase7_complete.py"
            ),
        ],
    )

    print(
        "\nALL PHASE 7 STEPS PASSED"
    )


if __name__ == "__main__":
    main()
