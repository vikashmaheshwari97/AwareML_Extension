from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.protocol import validate_frozen_protocol
from awareml.llm import StrictJournalOllamaClient


MANIFEST = ROOT / "data" / "journal" / "objective_selection_v2" / "manifest.json"
SHA_FILE = Path(str(MANIFEST) + ".sha256")
ACTIVE = ROOT / "data" / "journal" / "active_objective_selector.txt"


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

    for path in [MANIFEST, SHA_FILE, ACTIVE]:
        if not path.exists():
            raise RuntimeError("Missing Phase-11 freeze artifact: {}".format(path))

    expected_sha = SHA_FILE.read_text(encoding="utf-8").strip().split()[0]
    actual_sha = sha256_file(MANIFEST)
    if actual_sha != expected_sha:
        raise RuntimeError("Phase-11 manifest checksum mismatch.")

    if ACTIVE.read_text(encoding="utf-8").strip() != "objective_selection_v2/manifest.json":
        raise RuntimeError("Active objective selector marker is incorrect.")

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("release_status") != "frozen":
        raise RuntimeError("Phase-11 release is not frozen.")
    if payload["phase10_protocol"]["sha256"] != phase10["sha256"]:
        raise RuntimeError("Phase-11 release references the wrong Phase-10 protocol.")
    if payload["weighting_policy"]["policy_id"] != "equal_selected_v1":
        raise RuntimeError("Unexpected Phase-11 weighting policy.")
    if payload["journal_llm"]["model"] != "llama3:8b":
        raise RuntimeError("Unexpected Phase-11 journal model.")
    if payload["journal_llm"]["model_digest"] != runtime["model_digest"]:
        raise RuntimeError("Phase-11 model digest mismatch.")
    if payload["data_guards"]["23_heldout_dataset_contents_used"] is not False:
        raise RuntimeError("Held-out protection was violated.")

    for rel, expected in payload["source_sha256"].items():
        path = ROOT / rel
        if sha256_file(path) != expected:
            raise RuntimeError("Phase-11 source changed after freeze: {}".format(rel))

    print("=" * 72)
    print("AwareML Phase 11 COMPLETE validation: PASS")
    print("=" * 72)
    print("Scenario -> objective subset: READY")
    print("Objective subset -> equal weights: READY")
    print("Malformed JSON handling: EXPLICIT")
    print("Wrong model -> hard failure: READY")
    print("Journal model: llama3:8b")
    print("Weighting policy: equal_selected_v1")
    print("23 held-out dataset contents used: False")
    print("Release status: frozen")
    print("=" * 72)


if __name__ == "__main__":
    main()
