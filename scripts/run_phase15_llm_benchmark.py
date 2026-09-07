from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from awareml.explanation_integrity.benchmark import run_llm_phase15


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the model-facing Phase-15 correctness + faithfulness benchmark "
            "using the configured local Ollama model. Results are timestamped, "
            "partial failures are preserved, and outputs are NOT auto-frozen."
        )
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional development limit. Omit for all controlled cases.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=300.0,
        help=(
            "Read timeout for each Ollama generation. Default: 300 seconds. "
            "Increase this for slower CPU-only model execution."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help=(
            "Retries after a failed/timeout generation. Default: 1 "
            "(up to two attempts per generation)."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop immediately on the first generation failure. "
            "Default behavior records the failure and continues."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phase15/llm_runs",
    )
    args = parser.parse_args()

    print("=" * 96)
    print("Phase 15 · empirical LLM explanation-integrity run")
    print("=" * 96)
    print("Per-generation timeout: {} s".format(args.timeout_sec))
    print("Retries: {}".format(args.retries))
    print(
        "The benchmark needs two explanation generations per completed case "
        "(original + counterfactual)."
    )
    print()

    def _progress(event):
        kind = event.get("event")
        index = event.get("index")
        total = event.get("total")
        case_id = event.get("case_id")

        if kind == "case_start":
            print(
                "[{}/{}] {} · generating original explanation ..."
                .format(index, total, case_id),
                flush=True,
            )
        elif kind == "original_complete":
            print(
                "[{}/{}] {} · original complete; generating counterfactual ..."
                .format(index, total, case_id),
                flush=True,
            )
        elif kind == "case_complete":
            score = event.get("faithfulness_score")
            print(
                "[{}/{}] {} · complete · faithfulness={}"
                .format(index, total, case_id, score),
                flush=True,
            )

    result = run_llm_phase15(
        max_cases=args.max_cases,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        fail_fast=args.fail_fast,
        progress_callback=_progress,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model = str(result.get("model") or "unknown-model").replace(":", "-")
    run_dir = Path(args.output_root) / "{}__{}".format(stamp, model)
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "phase15_llm_run.json").write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            result.get("summary") or {},
            indent=2,
            ensure_ascii=False,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    (run_dir / "failures.json").write_text(
        json.dumps(
            result.get("failures") or [],
            indent=2,
            ensure_ascii=False,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )

    print("Model:", result.get("model"))
    print(
        "Explanation prompt:",
        result.get("explanation_prompt_version"),
    )
    print(
        "Resolved model:",
        (result.get("model_status") or {}).get("resolved_model"),
    )
    print("Requested cases:", result.get("case_count"))
    print(
        "Correctness completed:",
        result.get("completed_correctness_count"),
    )
    print(
        "Faithfulness completed:",
        result.get("completed_faithfulness_count"),
    )
    print("Failures:", result.get("failure_count"))
    print()
    print(json.dumps(result.get("summary"), indent=2))
    print()
    print("Saved:", run_dir.resolve())

    if not result.get("complete"):
        print()
        print(
            "INCOMPLETE RUN: partial evidence was saved rather than discarded. "
            "Inspect failures.json. For timeout failures, rerun with a larger "
            "--timeout-sec (for example 600)."
        )
        raise SystemExit(2)

    print()
    print(
        "IMPORTANT: This empirical model run is intentionally not auto-frozen. "
        "Review it before using it for a journal-facing LLM performance claim."
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
