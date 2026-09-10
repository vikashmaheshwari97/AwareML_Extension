from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "data" / "journal" / "information_seeking_v1" / "design"
FROZEN = ROOT / "data" / "journal" / "information_seeking_v1" / "frozen"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol_path = DESIGN / "protocol.json"
    coding_path = DESIGN / "coding_scheme.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    coding = json.loads(coding_path.read_text(encoding="utf-8"))

    errors = []
    gate = protocol.get("finalization_gate") or {}
    if protocol.get("status") != gate.get("protocol_status_required", "final_ready"):
        errors.append("Set protocol.status to final_ready after research-team approval.")
    target = protocol.get("participant_target") or {}
    if not isinstance(target.get("final_target"), int) or target.get("final_target") < 1:
        errors.append("Set participant_target.final_target to the approved completed-participant target.")
    ethics = protocol.get("ethics") or {}
    if ethics.get("status") not in gate.get("ethics_status_allowed", []):
        errors.append("Finalize ethics.status as approved, exempt, or not_required.")
    materials = protocol.get("participant_materials") or {}
    if materials.get("status") != gate.get("participant_materials_status_required", "final"):
        errors.append("Finalize participant materials/consent wording.")
    if coding.get("status") != gate.get("coding_scheme_status_required", "final"):
        errors.append("Review the qualitative coding scheme and set its status to final.")

    if errors:
        print("Information-Seeking design is NOT ready to freeze:")
        for error in errors:
            print(" -", error)
        return 2

    FROZEN.mkdir(parents=True, exist_ok=True)
    frozen_protocol = FROZEN / "protocol.json"
    frozen_coding = FROZEN / "coding_scheme.json"
    shutil.copy2(protocol_path, frozen_protocol)
    shutil.copy2(coding_path, frozen_coding)
    manifest = {
        "artifact": "information_seeking_design_v1",
        "status": "frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(frozen_protocol),
        "coding_scheme_sha256": sha(frozen_coding),
    }
    (FROZEN / "design_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    print("Information-Seeking design freeze: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
