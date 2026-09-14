from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.llm.confirmatory_runtime_v32 import ConfirmatoryOllamaClientV32


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-runtime", action="store_true")
    args = parser.parse_args()
    manifest = ROOT / "data" / "journal" / "objective_selection_v32" / "manifest.json"
    if not manifest.exists():
        raise RuntimeError("V3.2 selector is not frozen yet.")
    sha_path = Path(str(manifest) + ".sha256")
    expected = sha_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(manifest)
    if actual != expected:
        raise RuntimeError("V3.2 manifest checksum mismatch.")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("release_status") != "frozen":
        raise RuntimeError("V3.2 selector manifest is not frozen.")
    for rel, digest in payload.get("source_sha256", {}).items():
        path = ROOT / rel
        if not path.exists() or sha256_file(path) != digest:
            raise RuntimeError("Frozen V3.2 source mismatch: {}".format(rel))
    active = ROOT / "data" / "journal" / "active_objective_selector.txt"
    if active.read_text(encoding="utf-8").strip() != "objective_selection_v2/manifest.json":
        raise RuntimeError("Legacy active selector was changed; confirmatory baseline provenance is compromised.")
    if not args.skip_runtime:
        runtime = ConfirmatoryOllamaClientV32(root=ROOT).verify_runtime()
        expected_llm = payload["journal_llm"]
        if runtime.get("model_digest") != expected_llm.get("model_digest"):
            raise RuntimeError("Current model digest differs from frozen V3.2 runtime.")
        if str(runtime.get("ollama_version")) != str(expected_llm.get("ollama_version")):
            raise RuntimeError("Current Ollama version differs from frozen V3.2 runtime.")
        if str(runtime.get("runtime_lock_id")) != str(expected_llm.get("runtime_lock_id")):
            raise RuntimeError("Current confirmatory runtime-lock ID differs from frozen V3.2 runtime.")
    print("V3.2 selector freeze validation: PASS")
    print("manifest_sha256:", actual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
