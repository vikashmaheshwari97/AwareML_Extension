from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UI_DIR = ROOT / "data" / "ui" / "research_v2"
ACTIVE = ROOT / "data" / "ui" / "active_ui.txt"

PHASE_MANIFESTS = {
    "phase6": ROOT / "data" / "meta" / "models" / "recommender_v2" / "release_manifest.json",
    "phase7": ROOT / "data" / "llm" / "copilot_v1" / "manifest.json",
    "phase8": ROOT / "data" / "llm" / "faithfulness_v1" / "manifest.json",
}


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    for phase, path in PHASE_MANIFESTS.items():
        if not path.exists():
            raise RuntimeError("{} manifest is missing: {}".format(phase, path))

    UI_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "phase": "9",
        "release_id": "awareml-research-ui-v2",
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "capabilities": [
            "global_research_state",
            "research_command_center",
            "interactive_plotly_3d_decision_space",
            "preference_what_if_ranking",
            "run_studio_bridge",
            "streaming_observatory",
            "responsible_ai_observatory",
            "copilot_human_review_workspace",
            "faithfulness_lab",
            "research_export_center",
            "legacy_specialist_lab_bridge",
        ],
        "scientific_boundaries": {
            "decorative_synthetic_results": False,
            "raw_dataset_rows_in_llm_export": False,
            "held_out_23_dataset_split_used": False,
            "pre_run_predictions_labeled_as_predictions": True,
            "phase8_aef_labeled_as_development_evidence": True,
        },
        "backend_releases": {
            phase: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
            for phase, path in PHASE_MANIFESTS.items()
        },
    }

    manifest_path = UI_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path(str(manifest_path) + ".sha256").write_text(
        "{}  {}\n".format(sha256_file(manifest_path), manifest_path.name),
        encoding="utf-8",
    )

    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE.write_text("research_v2/manifest.json\n", encoding="utf-8")

    print("=" * 72)
    print("AwareML Phase 9 freeze: SUCCESS")
    print("=" * 72)
    print("Active UI marker:", ACTIVE)
    print("Manifest:", manifest_path)
    print("SHA256:", sha256_file(manifest_path))
    print("23-dataset held-out split used: False")
    print("=" * 72)


if __name__ == "__main__":
    main()
