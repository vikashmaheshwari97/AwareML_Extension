from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export the frozen Phase-15 trust stimulus bank for Track-2 study use."
        )
    )
    parser.add_argument(
        "--researcher",
        action="store_true",
        help=(
            "Export researcher-only labels/error types. "
            "Default is the blinded participant-facing bank."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    args = parser.parse_args()

    frozen = (
        ROOT
        / "data"
        / "journal"
        / "trust_stimulus_bank_v1"
        / "frozen"
    )

    source = (
        frozen / "stimuli_researcher.json"
        if args.researcher
        else frozen / "stimuli_participant.json"
    )
    if not source.exists():
        raise FileNotFoundError(source)

    rows = json.loads(source.read_text(encoding="utf-8"))
    frame = pd.json_normalize(rows)

    default_name = (
        "trust_stimulus_bank_v1_researcher.csv"
        if args.researcher
        else "trust_stimulus_bank_v1_participant.csv"
    )
    out = Path(args.output or ("artifacts/phase15/" + default_name))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    print("Exported:", out.resolve())
    print("Rows:", len(frame))
    if args.researcher:
        print(
            "WARNING: researcher export contains ground-truth condition labels."
        )
    else:
        print(
            "Participant export is blinded: no correctness label/error type/verifier metrics."
        )


if __name__ == "__main__":
    main()
