
from __future__ import annotations

import argparse
import json
from pathlib import Path

from awareml.studies.information_seeking_morten_v2 import (
    analyze,
    interaction_sequence,
    session_summaries,
)
from awareml.studies.store import StudyStore


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "journal" / "information_seeking_v2" / "analysis"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "final"], default="pilot")
    args = parser.parse_args()

    study_name = "information_seeking_v2_{}".format(args.mode)
    events = StudyStore().export(study_name)
    summary = analyze(events)
    sessions = session_summaries(events)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "{}_analysis.json".format(args.mode)).write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    sessions.to_csv(
        OUT / "{}_sessions.csv".format(args.mode),
        index=False,
    )
    events.to_csv(
        OUT / "{}_events.csv".format(args.mode),
        index=False,
    )

    if not sessions.empty:
        sequence_dir = OUT / "{}_sequences".format(args.mode)
        sequence_dir.mkdir(parents=True, exist_ok=True)
        for session_hash in sessions["session_hash"].astype(str):
            interaction_sequence(events, session_hash).to_csv(
                sequence_dir / "{}.csv".format(session_hash),
                index=False,
            )

    print(json.dumps(summary, indent=2, default=str))
    print("Analysis written to:", OUT)
    return 0 if summary.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
