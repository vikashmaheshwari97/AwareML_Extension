from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.llm.journal_client import JournalModelLockError
from awareml.llm.confirmatory_runtime_v32 import ConfirmatoryOllamaClientV32
from awareml.llm.objective_selection_v32 import SELECTOR_ID, SELECTOR_VERSION


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_existing(manifest: Path) -> dict:
    sha_path = Path(str(manifest) + ".sha256")
    if not sha_path.exists():
        raise RuntimeError("V3.2 manifest exists without checksum.")
    expected = sha_path.read_text(encoding="utf-8").strip().split()[0]
    if sha256_file(manifest) != expected:
        raise RuntimeError("V3.2 manifest checksum mismatch.")
    payload = read_json(manifest)
    for rel, expected_sha in payload.get("source_sha256", {}).items():
        path = ROOT / rel
        if not path.exists() or sha256_file(path) != expected_sha:
            raise RuntimeError("Frozen V3.2 source changed: {}".format(rel))
    return payload


def main() -> int:
    out = ROOT / "data" / "journal" / "objective_selection_v32" / "manifest.json"
    if out.exists():
        payload = validate_existing(out)
        print("V3.2 selector is already frozen and valid.")
        print("selector_id:", payload.get("selector_id"))
        print("manifest_sha256:", sha256_file(out))
        return 0

    client = ConfirmatoryOllamaClientV32(root=ROOT)
    runtime = client.verify_runtime()
    if client.model != "llama3:8b":
        raise JournalModelLockError(
            "V3.2 confirmatory selector must use the exact llama3:8b Phase-11R runtime lock."
        )

    source_paths = [
        Path("awareml/llm/objective_selection_v32.py"),
        Path("prompts/objective_selection_evidence_grounded_v32.txt"),
        Path("configs/journal/objective_selection_schema_v32.json"),
        Path("configs/journal/phase12_v2_protocol.json"),
        Path("awareml/llm/journal_client.py"),
        Path("awareml/llm/confirmatory_runtime_v32.py"),
        Path("configs/journal/objective_selection_v32_runtime_lock.json"),
        Path("awareml/llm/schemas.py"),
        Path("awareml/llm/weighting.py"),
        Path("awareml/engine/pareto_spec.py"),
    ]
    missing = [str(p) for p in source_paths if not (ROOT / p).exists()]
    if missing:
        raise RuntimeError("Cannot freeze V3.2; missing assets: {}".format(", ".join(missing)))

    active_selector = ROOT / "data" / "journal" / "active_objective_selector.txt"
    legacy_active_value = active_selector.read_text(encoding="utf-8").strip() if active_selector.exists() else None
    if legacy_active_value != "objective_selection_v2/manifest.json":
        raise RuntimeError(
            "Expected legacy active selector to remain objective_selection_v2/manifest.json before V3.2 freeze; got {!r}.".format(
                legacy_active_value
            )
        )

    v31 = ROOT / "awareml" / "llm" / "objective_selection_v31.py"
    v1_benchmark = ROOT / "data" / "journal" / "objective_selection_benchmark_v1" / "frozen" / "manifest.json"
    v2_selector = ROOT / "data" / "journal" / "objective_selection_v2" / "manifest.json"

    payload = {
        "schema_version": "3.2",
        "release_id": "objective_selection_v32_confirmatory_freeze",
        "release_status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selector_id": SELECTOR_ID,
        "selector_version": SELECTOR_VERSION,
        "purpose": "Precision-oriented evidence-grounded objective selection frozen before fresh Phase-12-v2 ground truth/outcomes.",
        "objective_vocabulary": ["Accuracy", "Runtime", "Energy", "CO2"],
        "development_lineage": {
            "v31_source_sha256": sha256_file(v31) if v31.exists() else None,
            "legacy_phase12_v1_manifest_sha256": sha256_file(v1_benchmark) if v1_benchmark.exists() else None,
            "legacy_phase11_v2_selector_sha256": sha256_file(v2_selector) if v2_selector.exists() else None,
            "old_phase12_cases_are_development_only": True,
            "old_phase12_cases_may_not_be_final_v32_test": True
        },
        "journal_llm": {
            "model": client.model,
            "model_digest": runtime.get("model_digest"),
            "ollama_version": runtime.get("ollama_version"),
            "runtime_lock_id": runtime.get("runtime_lock_id"),
            "generation": dict(client.generation),
            "silent_model_fallback": False,
            "same_model_digest_as_phase10": True,
            "legacy_phase10_ollama_version": runtime.get("legacy_phase10_ollama_version"),
            "baseline_reporting_name": "Phase-11 V2 method replay under Phase-11R confirmatory runtime",
            "candidate_reporting_name": "Objective Selection V3.2"
        },
        "benchmark_behavior": {
            "benchmark_mode": True,
            "semantic_recovery": False,
            "malformed_response": "explicit_malformed_no_recovery",
            "wrong_model_or_digest": "hard_failure",
            "guard_role": "reject unsupported LLM additions only; never add omitted objectives",
            "evidence_requirement": "selected labels require an exact scenario-local quote that passes a compact objective-specific concept guard"
        },
        "interactive_behavior": {
            "semantic_recovery_available": True,
            "must_be_explicitly_enabled": True,
            "recovery_is_recorded_in_fallback_used_and_audit": True
        },
        "weighting_policy": {
            "policy_id": "equal_selected_v1",
            "rule": "Equal weight among selected objectives; all unselected objectives receive zero."
        },
        "near_pareto": {
            "spec_id": "epsilon_pareto_v1",
            "epsilon": 0.05,
            "source": "awareml/engine/pareto_spec.py"
        },
        "legacy_active_selector_marker_unchanged": {
            "path": "data/journal/active_objective_selector.txt",
            "value": legacy_active_value
        },
        "source_sha256": {
            str(p).replace("\\", "/"): sha256_file(ROOT / p) for p in source_paths
        }
    }
    write_json(out, payload)
    digest = sha256_file(out)
    Path(str(out) + ".sha256").write_text("{}  manifest.json\n".format(digest), encoding="utf-8")
    marker = ROOT / "data" / "journal" / "active_objective_selector_v32.txt"
    marker.write_text("objective_selection_v32/manifest.json\n", encoding="utf-8")

    print("V3.2 confirmatory selector freeze: COMPLETE")
    print("selector_id:", SELECTOR_ID)
    print("model:", client.model)
    print("model_digest:", runtime.get("model_digest"))
    print("ollama_version:", runtime.get("ollama_version"))
    print("manifest_sha256:", digest)
    print("legacy active_objective_selector.txt was NOT changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
