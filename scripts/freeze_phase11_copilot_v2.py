from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.protocol import validate_frozen_protocol
from awareml.llm import StrictJournalOllamaClient


OUT_DIR = ROOT / "data" / "journal" / "objective_selection_v2"
MANIFEST = OUT_DIR / "manifest.json"
SHA_FILE = Path(str(MANIFEST) + ".sha256")
ACTIVE = ROOT / "data" / "journal" / "active_objective_selector.txt"


SOURCE_FILES = [
    "awareml/llm/schemas.py",
    "awareml/llm/weighting.py",
    "awareml/llm/journal_client.py",
    "awareml/llm/objective_selection.py",
    "awareml/llm/goal_parser.py",
    "awareml/llm/copilot.py",
    "awareml/llm/__init__.py",
    "awareml/ui_v2/pages_copilot.py",
    "tests/test_phase11_objective_selection.py",
    "scripts/validate_phase11_copilot_v2.py",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    phase10 = validate_frozen_protocol(
        ROOT,
        require_ollama_match=True,
    )
    runtime = StrictJournalOllamaClient().verify_runtime()

    source_hashes = {}
    for rel in SOURCE_FILES:
        path = ROOT / rel
        if not path.exists():
            raise RuntimeError("Missing Phase-11 source file: {}".format(rel))
        source_hashes[rel] = sha256_file(path)

    payload = {
        "schema_version": "1.0",
        "phase": 11,
        "release_id": "copilot_objective_selection_v2",
        "release_status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Roadmap-aligned scenario-to-objective-set inference with "
            "separate documented equal-selected weighting."
        ),
        "phase10_protocol": {
            "path": phase10["protocol_path"],
            "sha256": phase10["sha256"],
        },
        "objective_selection": {
            "task": "multi_label_objective_selection",
            "labels": ["Accuracy", "Runtime", "Energy", "CO2"],
            "statuses": [
                "valid",
                "ambiguous",
                "malformed",
                "out_of_scope",
                "contradictory",
            ],
            "primary_output_is_selected_objective_set": True,
            "weights_are_primary_benchmark_output": False,
        },
        "weighting_policy": {
            "policy_id": "equal_selected_v1",
            "rule": (
                "Equal weight among selected objectives; all unselected "
                "objectives receive zero."
            ),
        },
        "journal_llm": {
            "model": phase10["journal_model"],
            "model_digest": runtime["model_digest"],
            "ollama_version": runtime["ollama_version"],
            "silent_model_fallback": False,
        },
        "failure_handling": {
            "malformed_json": "explicit malformed status; interactive path may use labelled deterministic fallback",
            "ambiguous": "explicit ambiguous status; human review required",
            "out_of_scope": "explicit error/no proposal",
            "contradictory": "explicit error/no proposal",
            "wrong_model_or_digest": "hard failure",
        },
        "hcai_boundary": {
            "fairness_is_primary_objective_label": False,
            "explainability_is_primary_objective_label": False,
            "drift_is_primary_objective_label": False,
            "hcai_requirements_preserved_separately": True,
        },
        "data_guards": {
            "705_meta_logs_modified": False,
            "705_meta_logs_are_copilot_training_data": False,
            "23_heldout_dataset_contents_used": False,
            "phase12_benchmark_created": False,
        },
        "source_sha256": source_hashes,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(MANIFEST)
    SHA_FILE.write_text(
        "{}  {}\n".format(digest, MANIFEST.name),
        encoding="utf-8",
    )
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE.write_text(
        "objective_selection_v2/manifest.json\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("AwareML Phase 11 freeze: SUCCESS")
    print("=" * 72)
    print("Manifest:", MANIFEST)
    print("SHA256:", digest)
    print("Active selector marker:", ACTIVE)
    print("Journal model:", phase10["journal_model"])
    print("Weighting policy: equal_selected_v1")
    print("23 held-out dataset contents used: False")
    print("=" * 72)


if __name__ == "__main__":
    main()
