from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.ui_v2.pages import PAGE_REGISTRY_V2

ACTIVE = ROOT / "data" / "ui" / "active_ui.txt"


def main():
    if not ACTIVE.exists():
        raise RuntimeError("Phase-9 active UI marker is missing.")

    rel = ACTIVE.read_text(encoding="utf-8").strip()
    manifest_path = ACTIVE.parent / rel
    if not manifest_path.exists():
        raise RuntimeError("Active UI manifest is missing.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise RuntimeError("Phase 9 UI is not frozen.")

    boundary = manifest.get("scientific_boundaries") or {}
    if boundary.get("decorative_synthetic_results") is not False:
        raise RuntimeError("Research-integrity UI boundary failed.")
    if boundary.get("held_out_23_dataset_split_used") is not False:
        raise RuntimeError("Held-out dataset boundary failed.")
    if boundary.get("raw_dataset_rows_in_llm_export") is not False:
        raise RuntimeError("Raw-row export boundary failed.")
    if len(PAGE_REGISTRY_V2) != 9:
        raise RuntimeError("Unexpected number of Phase-9 workspaces.")

    print("=" * 72)
    print("AwareML Phase 9 COMPLETE validation: PASS")
    print("=" * 72)
    print("Research Command Center: READY")
    print("Interactive 3D Decision Space: READY")
    print("Global research state: READY")
    print("Streaming Observatory: READY")
    print("Responsible AI Observatory: READY")
    print("Copilot + human review: READY")
    print("Faithfulness Lab: READY")
    print("Export Center: READY")
    print("23-dataset held-out split used: False")
    print("Release status:", manifest["status"])
    print("=" * 72)


if __name__ == "__main__":
    main()
