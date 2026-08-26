from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
from awareml.recommender.evaluation import grouped_benchmark


def main():
    p = argparse.ArgumentParser()
    p.add_argument("meta_csv", help="CSV with one row per framework/dataset/profile evaluation")
    p.add_argument("--output", default="artifacts/meta_baseline_results.csv")
    args = p.parse_args()
    df = pd.read_csv(args.meta_csv)
    out = grouped_benchmark(df)
    print(out.to_string(index=False))
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
