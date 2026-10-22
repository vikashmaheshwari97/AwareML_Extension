
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from awareml.studies.trust_morten_v2 import (
    DEFAULT_DB_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PROTOCOL_PATH,
    Phase16V2Store,
    StimulusBankV2,
    analyze_v2,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze revised Phase-16 Trust Calibration v2 results."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/journal/trust_calibration_phase16_v2/frozen_results"
        ),
    )
    args = parser.parse_args()

    store = Phase16V2Store(args.db)
    result = analyze_v2(store, "final")
    completed = int(result.get("completed") or 0)

    if completed < 34:
        print("PHASE-16 v2 FINAL FREEZE: BLOCKED")
        print(
            "At least 34 valid completed Main Study participants are "
            "required; found {}.".format(completed)
        )
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    responses = pd.DataFrame(store.response_rows("final"))
    participants = pd.DataFrame(store.participant_rows("final"))
    events = pd.DataFrame(store.event_rows("final"))

    files = {
        "analysis_results.json": json.dumps(
            result, indent=2, sort_keys=True, default=str
        ).encode("utf-8"),
        "participants.csv": participants.to_csv(index=False).encode("utf-8"),
        "responses.csv": responses.to_csv(index=False).encode("utf-8"),
        "trial_events.csv": events.to_csv(index=False).encode("utf-8"),
        "trial_family_exposure.csv": pd.DataFrame(
            result.get("pair_coverage") or []
        ).to_csv(index=False).encode("utf-8"),
    }

    manifest_files = {}
    for name, content in files.items():
        path = args.output_root / name
        path.write_bytes(content)
        manifest_files[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }

    bank = StimulusBankV2()
    manifest = {
        "artifact": "phase16_trust_calibration_v2_frozen_results",
        "status": "frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed_participants": completed,
        "protocol_path": str(DEFAULT_PROTOCOL_PATH),
        "protocol_sha256": _sha(DEFAULT_PROTOCOL_PATH),
        "design_manifest_path": str(DEFAULT_MANIFEST_PATH),
        "design_manifest_sha256": _sha(DEFAULT_MANIFEST_PATH),
        "stimulus_manifest_path": str(bank.manifest_path),
        "stimulus_manifest_sha256": _sha(bank.manifest_path),
        "files": manifest_files,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("PHASE-16 v2 FINAL FREEZE: PASS")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
