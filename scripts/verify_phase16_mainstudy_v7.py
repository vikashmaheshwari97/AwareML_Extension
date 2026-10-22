from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/journal/trust_calibration_phase16_v1/design/protocol.json"
MANIFEST = ROOT / "data/journal/trust_calibration_phase16_v1/frozen_design/manifest.json"
PRIVATE_RUNTIME = ROOT / "artifacts/phase16/private_mainstudy_runtime.json"
DB = ROOT / "artifacts/phase16/trust_calibration.sqlite"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    runtime = json.loads(PRIVATE_RUNTIME.read_text(encoding="utf-8"))

    print("=" * 72)
    print("AwareML Main Study runtime verification")
    print("=" * 72)
    print("Phase-16 collection mode:", runtime.get("AWAREML_PHASE16_COLLECTION_MODE"))
    print("Phase-16 armed:", runtime.get("AWAREML_PHASE16_FINAL_ARMED"))
    print("Researcher key configured:", bool(runtime.get("AWAREML_STUDY_RESEARCHER_KEY")))
    print("Participant salt configured:", bool(runtime.get("AWAREML_PHASE16_ID_SALT")))
    print("Phase-17 collection mode:", runtime.get("AWAREML_INFORMATION_SEEKING_COLLECTION_MODE"))
    print("Phase-17 armed:", runtime.get("AWAREML_INFORMATION_SEEKING_FINAL_ARMED"))

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen_sha = str((((manifest.get("files") or {}).get("protocol.json") or {}).get("sha256") or ""))
    live_sha = sha256(PROTOCOL)
    print("Protocol matches frozen design:", live_sha == frozen_sha)

    if DB.exists():
        con = sqlite3.connect(str(DB))
        try:
            final_participants = con.execute(
                "SELECT COUNT(*) FROM participants WHERE collection_mode='final'"
            ).fetchone()[0]
            final_responses = con.execute(
                "SELECT COUNT(*) FROM responses WHERE collection_mode='final'"
            ).fetchone()[0]
        finally:
            con.close()
    else:
        final_participants = 0
        final_responses = 0

    print("Current Main Study registrations:", final_participants)
    print("Current Main Study responses:", final_responses)

    ready = (
        runtime.get("AWAREML_PHASE16_COLLECTION_MODE") == "final"
        and runtime.get("AWAREML_PHASE16_FINAL_ARMED") == "YES"
        and bool(runtime.get("AWAREML_STUDY_RESEARCHER_KEY"))
        and bool(runtime.get("AWAREML_PHASE16_ID_SALT"))
        and live_sha == frozen_sha
    )

    print("=" * 72)
    print("MAIN STUDY RUNTIME:", "READY" if ready else "CHECK REQUIRED")
    print("=" * 72)
    raise SystemExit(0 if ready else 2)


if __name__ == "__main__":
    main()
