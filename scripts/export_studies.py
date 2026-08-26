from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
from awareml.studies.store import StudyStore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=["trust", "information_seeking"], required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    df = StudyStore().export(args.study)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Exported {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
