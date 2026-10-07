from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from awareml.ui_v2.stream_shift_audit import window_distribution_shift


def main() -> None:
    parser = argparse.ArgumentParser(description="Descriptive Dutch-stream audit used by AwareML v4 diagnostics.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--target", default="occupation_binary")
    parser.add_argument("--sensitive", default="sex")
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    n = min(len(df), args.samples) if args.samples else len(df)
    audit = window_distribution_shift(df, args.target, args.window_size, n)
    print("Rows in CSV:", len(df))
    print("Rows audited:", n)
    print(audit.to_string(index=False))

    if args.sensitive in df.columns and args.target in df.columns:
        print("\nObserved target-rate gap by sensitive group (descriptive label audit; not model fairness):")
        rows = []
        for start in range(0, n, args.window_size):
            window = df.iloc[start:min(start + args.window_size, n)]
            if len(window) < max(50, args.window_size // 4):
                continue
            rates = window.groupby(args.sensitive)[args.target].mean()
            rows.append({
                "Sample": start + len(window),
                "Target-rate gap": float(rates.max() - rates.min()) if len(rates) >= 2 else None,
            })
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
