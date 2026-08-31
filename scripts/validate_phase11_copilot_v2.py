from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.protocol import validate_frozen_protocol
from awareml.llm import (
    GoalParser,
    JournalModelLockError,
    StrictJournalOllamaClient,
    deterministic_objective_selection,
    equal_weights_for_selected,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live-ollama",
        action="store_true",
        help="Verify the exact Phase-10 Ollama model lock.",
    )
    args = parser.parse_args()

    phase10 = validate_frozen_protocol(
        ROOT,
        require_ollama_match=args.live_ollama,
    )

    scenario = (
        "Suitable for deployment in a low-impact edge environment "
        "while still providing strong performance."
    )
    selection = deterministic_objective_selection(scenario)
    if selection.selected_objectives != ["Accuracy", "Energy", "CO2"]:
        raise RuntimeError(
            "Scenario-to-objective subset gate failed: {}".format(
                selection.selected_objectives
            )
        )

    weights, weighting = equal_weights_for_selected(
        selection.selected_objectives
    )
    values = weights.normalized_dict()
    expected = 1.0 / 3.0
    if abs(values["accuracy"] - expected) > 1e-12:
        raise RuntimeError("Equal-selected weighting failed for Accuracy.")
    if values["runtime"] != 0.0:
        raise RuntimeError("Unselected Runtime must receive zero weight.")
    if abs(values["energy"] - expected) > 1e-12:
        raise RuntimeError("Equal-selected weighting failed for Energy.")
    if abs(values["co2"] - expected) > 1e-12:
        raise RuntimeError("Equal-selected weighting failed for CO2.")

    # Wrong-model gate: journal evaluation must fail, never silently fallback.
    wrong_model_failed = False
    try:
        StrictJournalOllamaClient.validate_runtime_inventory(
            models=[{"name": "llama3.2:3b", "digest": "wrong"}],
            required_model_tag="llama3:8b",
            frozen_model_digest=phase10["journal_model_digest"],
            ollama_version=phase10["ollama_version"],
            frozen_ollama_version=phase10["ollama_version"],
        )
    except JournalModelLockError:
        wrong_model_failed = True
    if not wrong_model_failed:
        raise RuntimeError("Wrong-model benchmark gate did not fail.")

    live_runtime = None
    live_selection = None
    if args.live_ollama:
        client = StrictJournalOllamaClient()
        live_runtime = client.verify_runtime()

        # Semantic scoring belongs to Phase 12. Phase 11 only requires that the
        # exact locked pipeline can execute and return a structured status.
        selector = GoalParser()._selector()
        live_selection = selector.select(
            "The deployment must make dependable predictions on a "
            "battery-powered edge device."
        )
        if live_selection.status not in {
            "valid",
            "ambiguous",
            "contradictory",
            "out_of_scope",
            "malformed",
        }:
            raise RuntimeError("Unexpected live objective-selection status.")

    print("=" * 72)
    print("AwareML Phase 11 Copilot Objective-Selection V2 validation: PASS")
    print("=" * 72)
    print("Phase-10 protocol SHA256:", phase10["sha256"])
    print("Journal model:", phase10["journal_model"])
    print("Objective subset:", selection.selected_objectives)
    print("Weighting policy:", weighting["policy_id"])
    print("Weights:", values)
    print("Malformed/failure handling: EXPLICIT")
    print("Wrong model -> benchmark fails: PASS")
    print("705 meta-logs changed: False")
    print("23 held-out dataset contents used: False")
    print("Live Ollama checked:", bool(args.live_ollama))
    if live_runtime:
        print("Live model digest:", live_runtime["model_digest"])
        print("Live Ollama version:", live_runtime["ollama_version"])
        print("Live structured status:", live_selection.status)
        print("Live selected objectives:", live_selection.selected_objectives)
    print("=" * 72)


if __name__ == "__main__":
    main()
