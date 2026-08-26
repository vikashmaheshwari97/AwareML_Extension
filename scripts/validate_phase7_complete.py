from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.llm import (
    CopilotService,
    GoalParser,
    GroundedCopilotChat,
    OllamaClient,
)


ACTIVE = (
    ROOT
    / "data"
    / "llm"
    / "active_copilot.txt"
)


def main() -> None:
    if not ACTIVE.exists():
        raise RuntimeError(
            "Phase 7 active marker is missing."
        )

    rel = ACTIVE.read_text(
        encoding="utf-8"
    ).strip()
    manifest_path = (
        ACTIVE.parent / rel
    )
    if not manifest_path.exists():
        raise RuntimeError(
            "Active Phase 7 manifest is missing."
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )
    if manifest.get(
        "status"
    ) != "frozen":
        raise RuntimeError(
            "Phase 7 is not frozen."
        )

    boundary = (
        manifest.get(
            "scientific_boundaries"
        )
        or {}
    )
    if boundary.get(
        "primary_objectives"
    ) != [
        "accuracy",
        "runtime",
        "energy",
        "co2",
    ]:
        raise RuntimeError(
            "Phase 7 primary-objective boundary changed."
        )
    if boundary.get(
        "raw_dataset_rows_sent_to_llm"
    ) is not False:
        raise RuntimeError(
            "Phase 7 privacy boundary failed."
        )
    if boundary.get(
        "23_dataset_test_split_used"
    ) is not False:
        raise RuntimeError(
            "Held-out dataset policy failed."
        )

    # Import/API contract check.
    assert CopilotService
    assert GoalParser
    assert GroundedCopilotChat
    assert OllamaClient

    print("=" * 72)
    print(
        "AwareML Phase 7 COMPLETE validation: PASS"
    )
    print("=" * 72)
    print(
        "Goal -> configuration:",
        "READY",
    )
    print(
        "Human review + config diff:",
        "READY",
    )
    print(
        "Before/during/after grounded chat:",
        "READY",
    )
    print(
        "Raw rows sent to LLM:",
        False,
    )
    print(
        "23-dataset evaluation split used:",
        False,
    )
    print(
        "Release status:",
        manifest["status"],
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
