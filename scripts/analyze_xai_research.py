from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Analyze recorded XAI or supplementary trust evidence.")
    parser.add_argument("kind", choices=["attribution", "concept", "behavior"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "behavior":
        import pandas as pd
        from awareml.studies.behavioral_calibration import analyze_behavior, REQUIRED
        report = analyze_behavior(pd.read_csv(args.input, dtype={k: str for k in REQUIRED[:-1]}))
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
        if args.kind == "attribution":
            from awareml.explanation_integrity.attribution_alignment import analyze_records
            report = analyze_records(payload.get("records", []) if isinstance(payload, dict) else payload)
        else:
            from awareml.explanation_integrity.concept_audit import analyze_probe
            report = analyze_probe(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
