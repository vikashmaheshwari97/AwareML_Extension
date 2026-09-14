from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.objective_benchmark_v2 import (
    Phase12V2Error,
    aggregate_annotations,
    audit_benchmark,
    finalize_benchmark,
    freeze_adversarial_set,
    freeze_design,
    freeze_paraphrase_set,
    freeze_primary_benchmark,
    prepare_annotation_packets,
    prepare_paraphrase_bases,
    prepare_paraphrase_reviews,
    run_adversarial,
    run_paraphrases,
    run_primary,
    score_adversarial,
    score_paraphrases,
    score_primary,
    validate_all_evaluation_inputs_frozen,
    validate_design_inputs,
)


def emit(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AwareML Phase-12-v2 fresh confirmatory objective-selection benchmark"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-design")
    sub.add_parser("freeze-design")
    p = sub.add_parser("prepare-annotations")
    p.add_argument("--force", action="store_true")
    sub.add_parser("aggregate-annotations")
    sub.add_parser("freeze-primary")
    p = sub.add_parser("prepare-paraphrases")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("prepare-paraphrase-reviews")
    p.add_argument("--force", action="store_true")
    sub.add_parser("freeze-paraphrases")
    sub.add_parser("freeze-adversarial")
    sub.add_parser("validate-evaluation-freeze")

    for name in ("run-primary", "run-paraphrases", "run-adversarial"):
        p = sub.add_parser(name)
        p.add_argument(
            "--selector",
            choices=("baseline", "v32", "both"),
            default="both",
            help="Run frozen Phase-11 V2, V3.2, or both. 'both' is required for final paired comparison.",
        )

    sub.add_parser("score-primary")
    sub.add_parser("score-paraphrases")
    sub.add_parser("score-adversarial")
    sub.add_parser("finalize")
    sub.add_parser("audit")
    args = parser.parse_args()

    selectors = None
    if hasattr(args, "selector"):
        selectors = ("baseline", "v32") if args.selector == "both" else (args.selector,)

    try:
        if args.command == "validate-design":
            emit(validate_design_inputs(ROOT))
        elif args.command == "freeze-design":
            emit(freeze_design(ROOT))
        elif args.command == "prepare-annotations":
            emit(prepare_annotation_packets(ROOT, force=args.force))
        elif args.command == "aggregate-annotations":
            emit(aggregate_annotations(ROOT))
        elif args.command == "freeze-primary":
            emit(freeze_primary_benchmark(ROOT))
        elif args.command == "prepare-paraphrases":
            emit(prepare_paraphrase_bases(ROOT, force=args.force))
        elif args.command == "prepare-paraphrase-reviews":
            emit(prepare_paraphrase_reviews(ROOT, force=args.force))
        elif args.command == "freeze-paraphrases":
            emit(freeze_paraphrase_set(ROOT))
        elif args.command == "freeze-adversarial":
            emit(freeze_adversarial_set(ROOT))
        elif args.command == "validate-evaluation-freeze":
            emit(validate_all_evaluation_inputs_frozen(ROOT))
        elif args.command == "run-primary":
            emit(run_primary(ROOT, selectors=selectors))
        elif args.command == "run-paraphrases":
            emit(run_paraphrases(ROOT, selectors=selectors))
        elif args.command == "run-adversarial":
            emit(run_adversarial(ROOT, selectors=selectors))
        elif args.command == "score-primary":
            emit(score_primary(ROOT))
        elif args.command == "score-paraphrases":
            emit(score_paraphrases(ROOT))
        elif args.command == "score-adversarial":
            emit(score_adversarial(ROOT))
        elif args.command == "finalize":
            emit(finalize_benchmark(ROOT))
        elif args.command == "audit":
            emit(audit_benchmark(ROOT))
        else:
            parser.error("unknown command")
    except Phase12V2Error as exc:
        print("PHASE12-V2 ERROR:", exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
