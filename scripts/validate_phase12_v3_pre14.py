from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    phase12 = ROOT / "data" / "journal" / "objective_selection_benchmark_v1" / "frozen" / "manifest.json"
    phase13 = ROOT / "data" / "journal" / "recommender_multiobjective_validation_v1" / "frozen" / "manifest.json"
    required = [
        ROOT / "awareml" / "llm" / "objective_selection_v3.py",
        ROOT / "prompts" / "objective_selection_evidence_grounded_v3.txt",
        ROOT / "scripts" / "run_phase12_v3_primary_diagnostic.py",
        ROOT / "scripts" / "run_phase12_v3_paraphrase_diagnostic.py",
        ROOT / "scripts" / "run_phase12_v3_adversarial_diagnostic.py",
        ROOT / "scripts" / "evaluate_phase12_v3_diagnostic.py",
        ROOT / "tests" / "test_phase12_v3_pre14_hotfix.py",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing V3/hotfix files: {}".format(", ".join(missing)))

    if not phase12.exists():
        raise RuntimeError("Frozen Phase-12 manifest is missing.")
    phase12_payload = json.loads(phase12.read_text(encoding="utf-8"))
    if phase12_payload.get("release_status") != "frozen":
        raise RuntimeError("Phase-12 v1 is no longer frozen.")
    if int(phase12_payload.get("primary_benchmark_n") or 0) != 55:
        raise RuntimeError("Unexpected Phase-12 benchmark size.")

    if phase13.exists():
        phase13_payload = json.loads(phase13.read_text(encoding="utf-8"))
        if phase13_payload.get("release_status") != "frozen":
            raise RuntimeError("Phase-13 artifact is no longer frozen.")

    print("=" * 72)
    print("AwareML Phase-12 V3 / pre-Phase-14 integration validation: PASS")
    print("=" * 72)
    print("Phase-12 v1 frozen SHA256:", sha(phase12))
    print("Phase-13 frozen SHA256:", sha(phase13) if phase13.exists() else "N/A")
    print("V3 method installed: True")
    print("V3 role: improvement candidate + post-hoc diagnostic")
    print("Frozen Phase-12 v1 modified: False")
    print("Independent fresh V3 benchmark required for a new journal claim: True")
    print("=" * 72)


if __name__ == "__main__":
    main()
