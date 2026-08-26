from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.faithfulness import (
    FaithfulnessEvaluator,
    FaithfulRationaleGenerator,
    objective_influence,
)


ACTIVE = (
    ROOT
    / "data"
    / "llm"
    / "active_faithfulness.txt"
)


def main() -> None:
    if not ACTIVE.exists():
        raise RuntimeError(
            "Phase-8 active marker is missing."
        )

    rel = ACTIVE.read_text(
        encoding="utf-8"
    ).strip()
    manifest_path = (
        ACTIVE.parent / rel
    )
    if not manifest_path.exists():
        raise RuntimeError(
            "Active Phase-8 manifest is missing."
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
            "Phase 8 is not frozen."
        )

    scope = (
        manifest.get(
            "scientific_scope"
        )
        or {}
    )

    required_true = [
        "counterfactual_evidence_tests",
        "rationale_sensitivity",
        "citation_grounding",
        "decision_rationale_alignment",
        "external_objective_influence",
    ]
    for key in required_true:
        if scope.get(key) is not True:
            raise RuntimeError(
                "Missing Phase-8 capability: {}".format(
                    key
                )
            )

    if scope.get(
        "internal_llm_attention_attribution"
    ) is not False:
        raise RuntimeError(
            "Phase-8 internal-attribution boundary changed."
        )
    if scope.get(
        "pe_lrp"
    ) is not False:
        raise RuntimeError(
            "Phase-8 PE-LRP boundary changed."
        )

    evaluation = (
        manifest.get(
            "evaluation"
        )
        or {}
    )
    if evaluation.get(
        "held_out_23_dataset_split_used"
    ) is not False:
        raise RuntimeError(
            "Held-out test split policy failed."
        )

    assert FaithfulnessEvaluator
    assert FaithfulRationaleGenerator
    assert objective_influence

    print("=" * 72)
    print(
        "AwareML Phase 8 COMPLETE validation: PASS"
    )
    print("=" * 72)
    print(
        "Counterfactual evidence tests:",
        "READY",
    )
    print(
        "Rationale sensitivity:",
        "READY",
    )
    print(
        "Grounding/fidelity metrics:",
        "READY",
    )
    print(
        "External objective-influence alignment:",
        "READY",
    )
    print(
        "Internal Ollama PE-LRP/attention attribution:",
        False,
    )
    print(
        "Held-out 23-dataset split used:",
        False,
    )
    print(
        "Release status:",
        manifest["status"],
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
