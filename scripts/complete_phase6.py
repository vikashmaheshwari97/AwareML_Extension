from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


STEPS = [
    (
        "6.3 train final models",
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "train_recommender_v2_final.py"
            ),
        ],
    ),
    (
        "6.3 validate",
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "validate_recommender_v2_final.py"
            ),
        ],
    ),
    (
        "6.4 preference evaluation",
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "evaluate_preference_ranking_v2.py"
            ),
        ],
    ),
    (
        "6.4 validate",
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "validate_preference_ranking_v2.py"
            ),
        ],
    ),
    (
        "6.5 freeze",
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "freeze_recommender_v2.py"
            ),
        ],
    ),
    (
        "6.5 complete validation",
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "validate_phase6_complete.py"
            ),
        ],
    ),
]


def main() -> None:
    print(
        "AwareML Phase 6.3 -> 6.5 orchestrator"
    )
    for label, command in STEPS:
        print(
            "\n>>> {}".format(label),
            flush=True,
        )
        subprocess.run(
            command,
            cwd=str(ROOT),
            check=True,
        )
    print(
        "\nALL PHASE 6.3-6.5 STEPS PASSED"
    )


if __name__ == "__main__":
    main()
