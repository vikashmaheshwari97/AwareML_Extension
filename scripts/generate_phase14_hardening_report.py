from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from awareml.analysis.repeatability import (
    hardware_table,
    phase14_gate,
    summarize_repeatability,
)


ROOT = Path(__file__).resolve().parents[1]


def main():
    repeated = (
        ROOT
        / "artifacts"
        / "phase14"
        / "repeatability"
        / "phase14_repeated_results.json"
    )
    out = ROOT / "artifacts" / "phase14"
    out.mkdir(parents=True, exist_ok=True)

    if not repeated.exists():
        print(
            "No repeated-run evidence found at:",
            repeated,
        )
        print(
            "Run scripts.run_phase14_repeatability first. "
            "The Phase-14 implementation can still be validated without fabricating repeatability evidence."
        )
        return

    rows = json.loads(repeated.read_text(encoding="utf-8"))
    repeatability = summarize_repeatability(rows, min_repetitions=3)
    hardware = hardware_table(rows)
    gate = phase14_gate(rows, min_repetitions=3)

    repeatability.to_csv(out / "repeatability_table.csv", index=False)
    hardware.to_csv(out / "hardware_table.csv", index=False)
    (out / "phase14_gate.json").write_text(
        json.dumps(gate, indent=2, default=str),
        encoding="utf-8",
    )

    print("Repeatability table:")
    print(repeatability.to_string(index=False))
    print()
    print("Hardware table:")
    print(hardware.to_string(index=False))
    print()
    print("Gate:", json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
