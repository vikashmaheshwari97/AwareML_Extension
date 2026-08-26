from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
from awareml.benchmark_suite import run_repeated_suite


def _parse_item(text: str):
    # name=path.csv:target
    if "=" not in text or ":" not in text:
        raise argparse.ArgumentTypeError("Use NAME=PATH.csv:TARGET")
    name, rest = text.split("=", 1)
    path, target = rest.rsplit(":", 1)
    return name, path, target


def main():
    p = argparse.ArgumentParser(description="Run repeated multi-dataset AwareML benchmark suite")
    p.add_argument("--dataset", action="append", required=True, help="NAME=PATH.csv:TARGET (repeatable)")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--time-budget", type=float, default=60.0)
    p.add_argument("--output", default="artifacts/suite_results.csv")
    args = p.parse_args()

    datasets, targets = {}, {}
    for item in args.dataset:
        name, path, target = _parse_item(item)
        datasets[name] = pd.read_csv(path)
        targets[name] = target
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out = run_repeated_suite(
        datasets, targets, seeds=seeds, window_size=args.window,
        max_samples=args.max_samples, time_budget_sec=args.time_budget,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.groupby("framework")[["accuracy", "runtime_sec"]].agg(["mean", "std"]))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
