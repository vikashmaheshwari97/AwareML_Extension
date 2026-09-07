from __future__ import annotations

import argparse
import json
from pathlib import Path

from awareml.explanation_integrity.benchmark import (
    freeze_all_controlled,
    run_controlled_phase15,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase-15 controlled correctness/faithfulness benchmark "
            "and optionally freeze the controlled journal artifacts."
        )
    )
    parser.add_argument(
        "--output",
        default="artifacts/phase15/controlled_validation",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help=(
            "Freeze controlled evaluator-validation artifacts under "
            "data/journal. This does not claim actual LLM performance."
        ),
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help=(
            "Explicitly allow replacement of an existing controlled frozen "
            "artifact. Avoid this after Phase 15 is formally frozen."
        ),
    )
    args = parser.parse_args()

    results = run_controlled_phase15()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase15_controlled_benchmark.json").write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print("Phase 15 · controlled explanation-integrity benchmark")
    print("=" * 96)
    print(
        "Correctness cases:",
        results["correctness"]["case_count"],
    )
    print(
        "Correctness explanations:",
        results["correctness"]["explanation_count"],
    )
    print("Stimulus bank:", results["stimuli"]["summary"])
    print(
        "Known-faithful mean score:",
        results["faithfulness"]["summary"]
        ["known_faithful"]["mean_faithfulness_score"],
    )
    print(
        "Known-unfaithful sticky mean score:",
        results["faithfulness"]["summary"]
        ["known_unfaithful_sticky"]["mean_faithfulness_score"],
    )
    print("Saved:", out.resolve())

    if args.freeze:
        frozen = freeze_all_controlled(
            Path("data/journal"),
            overwrite=bool(args.allow_overwrite),
        )
        print()
        print("Frozen controlled artifacts:")
        for name, payload in frozen.items():
            print("  {} -> {}".format(name, payload["directory"]))

    print("=" * 96)


if __name__ == "__main__":
    main()
