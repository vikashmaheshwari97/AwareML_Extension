from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from awareml.llm.objective_selection_v33 import EvidenceGroundedObjectiveSelectorV33


DEFAULT_SCENARIOS = [
    (
        "multi_objective",
        "A battery-powered wildlife monitor must make dependable detections while keeping power draw low and maintaining a small environmental footprint.",
    ),
    (
        "runtime_accuracy",
        "A collision-warning service must respond immediately and avoid incorrect warnings.",
    ),
    (
        "energy_only",
        "The remote sensor should last for months between charges.",
    ),
    (
        "out_of_scope",
        "Make the dashboard prettier and improve the interface color scheme.",
    ),
    (
        "contradictory",
        "Response time does not matter, but the service must respond immediately.",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run interactive V3.3 smoke scenarios.")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Custom scenario. May be passed multiple times.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    selector = EvidenceGroundedObjectiveSelectorV33(root=repo)
    cases = (
        [("custom_{:02d}".format(i + 1), s) for i, s in enumerate(args.scenario)]
        if args.scenario
        else DEFAULT_SCENARIOS
    )

    for case_id, scenario in cases:
        result = selector.select(scenario)
        audit = dict(selector.last_audit or {})
        print(
            json.dumps(
                {
                    "case_id": case_id,
                    "scenario": scenario,
                    "status": result.status,
                    "selected_objectives": list(result.selected_objectives),
                    "fallback_used": bool(result.fallback_used),
                    "semantic_recovered_objectives": audit.get(
                        "semantic_recovered_objectives", []
                    ),
                    "stage": audit.get("stage"),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
