from __future__ import annotations

import argparse
import json
from pathlib import Path

from awareml.studies.information_seeking_analysis import (
    analyze_information_seeking,
    manual_coding,
    session_summaries,
)
from awareml.studies.store import StudyStore


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "journal" / "information_seeking_v1" / "analysis"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "final"], default="pilot")
    args = parser.parse_args()

    study_name = "information_seeking_{}".format(args.mode)
    events = StudyStore().export(study_name)
    summary = analyze_information_seeking(events)
    sessions = session_summaries(events)
    codes = manual_coding(events)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "{}_analysis.json".format(args.mode)).write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    sessions.to_csv(OUT / "{}_sessions.csv".format(args.mode), index=False)
    codes.to_csv(OUT / "{}_manual_codes.csv".format(args.mode), index=False)

    print(json.dumps(summary, indent=2))
    print("Analysis written to:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
