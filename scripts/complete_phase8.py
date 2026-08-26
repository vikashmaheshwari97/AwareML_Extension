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
        "--offline",
        action="store_true",
        help=(
            "Development-only mode: skip the live Ollama "
            "subset and allow an offline freeze."
        ),
    )
    parser.add_argument(
        "--ollama-datasets",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--model",
        default=None,
    )
    args = parser.parse_args()

    evaluation = [
        sys.executable,
        str(
            ROOT
            / "scripts"
            / "run_phase8_faithfulness.py"
        ),
    ]

    validation = [
        sys.executable,
        str(
            ROOT
            / "scripts"
            / "validate_phase8_faithfulness.py"
        ),
    ]

    freeze = [
        sys.executable,
        str(
            ROOT
            / "scripts"
            / "freeze_phase8_faithfulness.py"
        ),
    ]

    if args.offline:
        freeze.append(
            "--allow-offline"
        )
    else:
        evaluation.extend(
            [
                "--ollama",
                "--require-ollama",
                "--ollama-datasets",
                str(
                    args.ollama_datasets
                ),
            ]
        )
        if args.model:
            evaluation.extend(
                [
                    "--model",
                    args.model,
                ]
            )
        validation.append(
            "--require-ollama"
        )

    run(
        "Phase 8 evidence-faithfulness evaluation",
        evaluation,
    )
    run(
        "Phase 8 validation",
        validation,
    )
    run(
        "Phase 8 freeze",
        freeze,
    )
    run(
        "Phase 8 final validation",
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "validate_phase8_complete.py"
            ),
        ],
    )

    print(
        "\nALL PHASE 8 STEPS PASSED"
    )


if __name__ == "__main__":
    main()
