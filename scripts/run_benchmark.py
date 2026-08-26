from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import pandas as pd
from awareml.engine.runner import run_benchmark
from awareml.types import RunConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--target", required=True)
    p.add_argument("--sensitive")
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--time-budget", type=float, default=60)
    p.add_argument("--output", default="artifacts/benchmark.json")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    cfg = RunConfig(
        target=args.target, sensitive_attribute=args.sensitive, window_size=args.window,
        max_samples=args.max_samples, time_budget_sec=args.time_budget,
    )
    results = run_benchmark(df, cfg)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps([r.to_dict() for r in results], indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
