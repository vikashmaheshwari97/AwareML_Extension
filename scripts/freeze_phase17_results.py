from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from awareml.studies.information_seeking_analysis import analyze_information_seeking
from awareml.studies.store import StudyStore


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "journal" / "information_seeking_v1"
FROZEN = BASE / "frozen"
RESULTS = FROZEN / "results"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    design_manifest = FROZEN / "design_manifest.json"
    if not design_manifest.exists():
        print("Freeze the Information-Seeking design before freezing final results.")
        return 2

    protocol = json.loads((FROZEN / "protocol.json").read_text(encoding="utf-8"))
    target = int((protocol.get("participant_target") or {}).get("final_target") or 0)
    events = StudyStore().export("information_seeking_final")
    analysis = analyze_information_seeking(events)
    completed = int(analysis.get("completed_sessions") or 0)

    if completed < target:
        print("Final results are not ready: completed={} target={}".format(completed, target))
        return 2

    RESULTS.mkdir(parents=True, exist_ok=True)
    events_path = RESULTS / "behavior_events.csv"
    analysis_path = RESULTS / "analysis.json"
    events.to_csv(events_path, index=False)
    analysis_path.write_text(
        json.dumps(analysis, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "artifact": "information_seeking_results_v1",
        "status": "frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed_sessions": completed,
        "target_completed_sessions": target,
        "behavior_events_sha256": sha(events_path),
        "analysis_sha256": sha(analysis_path),
    }
    (RESULTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    print("Information-Seeking final results freeze: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
