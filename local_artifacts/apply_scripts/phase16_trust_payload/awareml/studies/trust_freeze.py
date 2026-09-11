from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .trust import (
    DEFAULT_DB_PATH,
    DEFAULT_PROTOCOL_PATH,
    FINAL_DESIGN_MANIFEST,
    PHASE16_SCHEMA_VERSION,
    RANDOMIZATION_VERSION,
    Phase15StimulusBank,
    Phase16Store,
    ProtocolGateError,
    load_phase16_protocol,
    sha256_file,
    validate_protocol_for_design_freeze,
)
from .trust_analysis import analyze_phase16_store, write_analysis_outputs


DESIGN_FREEZE_ROOT = Path("data/journal/trust_calibration_phase16_v1/frozen_design")
FINAL_FREEZE_ROOT = Path("data/journal/trust_calibration_phase16_v1/frozen")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        handle.write("\n")


def _file_meta(path: Path) -> Dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}


def freeze_phase16_design(
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    output_root: Path = DESIGN_FREEZE_ROOT,
    stimulus_root: Optional[Path] = None,
) -> Path:
    protocol = load_phase16_protocol(protocol_path)
    errors = validate_protocol_for_design_freeze(protocol)
    if errors:
        raise ProtocolGateError("Cannot freeze Phase-16 design: " + "; ".join(errors))
    if output_root.exists() and any(output_root.iterdir()):
        raise ProtocolGateError("Frozen Phase-16 design already exists at {}".format(output_root))
    bank = Phase15StimulusBank(stimulus_root, verify_hashes=True)
    bank_summary = bank.validate_design_pool(
        int((protocol.get("design") or {}).get("shared_pool_min_per_condition", 20))
    )

    output_root.mkdir(parents=True, exist_ok=True)
    frozen_protocol = output_root / "protocol.json"
    shutil.copy2(str(protocol_path), str(frozen_protocol))
    stimulus_manifest_copy = output_root / "phase15_stimulus_manifest.json"
    shutil.copy2(str(bank.manifest_path), str(stimulus_manifest_copy))

    manifest = {
        "artifact": "phase16_trust_calibration_design_v1",
        "schema_version": PHASE16_SCHEMA_VERSION,
        "status": "frozen",
        "frozen_utc": _utc_now(),
        "central_rq": protocol.get("central_rq"),
        "source_stimulus_bank": {
            "root": str(bank.root),
            "manifest_sha256": sha256_file(bank.manifest_path),
            "summary": bank_summary,
        },
        "files": {
            "protocol.json": _file_meta(frozen_protocol),
            "phase15_stimulus_manifest.json": _file_meta(stimulus_manifest_copy),
        },
    }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def freeze_phase16_results(
    db_path: Path = DEFAULT_DB_PATH,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    output_root: Path = FINAL_FREEZE_ROOT,
    analysis_script_path: Path = Path("scripts/analyze_phase16_trust.py"),
) -> Path:
    if not FINAL_DESIGN_MANIFEST.exists():
        raise ProtocolGateError("Final freeze requires the frozen Phase-16 design manifest.")
    if output_root.exists() and any(output_root.iterdir()):
        raise ProtocolGateError("Frozen Phase-16 results already exist at {}".format(output_root))
    protocol = load_phase16_protocol(protocol_path)
    errors = validate_protocol_for_design_freeze(protocol)
    if errors:
        raise ProtocolGateError("Final protocol gate failed: " + "; ".join(errors))
    with FINAL_DESIGN_MANIFEST.open("r", encoding="utf-8") as handle:
        frozen_design = json.load(handle)
    frozen_protocol_sha = str(
        (((frozen_design.get("files") or {}).get("protocol.json") or {}).get("sha256") or "")
    )
    frozen_stimulus_sha = str(
        ((frozen_design.get("source_stimulus_bank") or {}).get("manifest_sha256") or "")
    )
    if frozen_protocol_sha and sha256_file(protocol_path) != frozen_protocol_sha:
        raise ProtocolGateError("Current protocol differs from the frozen Phase-16 design.")

    store = Phase16Store(db_path)
    participant_rows = store.participant_rows("final")
    assignment_rows = store.all_assignment_rows("final")
    for participant in participant_rows:
        if frozen_protocol_sha and str(participant.get("protocol_sha256")) != frozen_protocol_sha:
            raise ProtocolGateError("A final participant was collected under a different protocol hash.")
        if frozen_stimulus_sha and str(participant.get("stimulus_manifest_sha256")) != frozen_stimulus_sha:
            raise ProtocolGateError("A final participant was collected under a different stimulus-bank hash.")
        if str(participant.get("randomization_version")) != RANDOMIZATION_VERSION:
            raise ProtocolGateError("A final participant was collected under a different randomization version.")
    for assignment in assignment_rows:
        if str(assignment.get("randomization_version")) != RANDOMIZATION_VERSION:
            raise ProtocolGateError("A final assignment has an unexpected randomization version.")
        if str(assignment.get("correctness_condition")) not in {"correct", "incorrect"}:
            raise ProtocolGateError("A final assignment contains an invalid correctness condition.")

    analysis = analyze_phase16_store(store, "final", protocol_path)
    if analysis.get("status") != "ok":
        raise ProtocolGateError("Final analysis is not complete: {}".format(analysis.get("status")))
    gate = analysis.get("completion_gate") or {}
    if not gate.get("target_met"):
        raise ProtocolGateError(
            "Power/completion target not met: {} valid completed; required {}.".format(
                gate.get("valid_completed_participants"), gate.get("required_completed_participants")
            )
        )
    if gate.get("completeness_issues"):
        raise ProtocolGateError("Final data contain incomplete participant records.")

    output_root.mkdir(parents=True, exist_ok=True)
    frozen_protocol = output_root / "study_protocol.json"
    shutil.copy2(str(DESIGN_FREEZE_ROOT / "protocol.json"), str(frozen_protocol))
    frozen_design_manifest = output_root / "design_manifest.json"
    shutil.copy2(str(FINAL_DESIGN_MANIFEST), str(frozen_design_manifest))

    assignments = pd.DataFrame(store.all_assignment_rows("final"))
    participants = pd.DataFrame(store.participant_rows("final"))
    responses = pd.DataFrame(store.response_rows("final"))
    assignments_path = output_root / "stimulus_randomization.csv"
    participants_path = output_root / "participant_summary.csv"
    responses_path = output_root / "participant_responses.csv"
    assignments.to_csv(assignments_path, index=False)
    participants.to_csv(participants_path, index=False)
    responses.to_csv(responses_path, index=False)

    analysis_paths = write_analysis_outputs(analysis, output_root)
    frozen_script = output_root / "analysis_script.py"
    shutil.copy2(str(analysis_script_path), str(frozen_script))

    files = {
        "study_protocol.json": _file_meta(frozen_protocol),
        "design_manifest.json": _file_meta(frozen_design_manifest),
        "stimulus_randomization.csv": _file_meta(assignments_path),
        "participant_summary.csv": _file_meta(participants_path),
        "participant_responses.csv": _file_meta(responses_path),
        "analysis_script.py": _file_meta(frozen_script),
        "analysis_results.json": _file_meta(analysis_paths["analysis"]),
        "calibration_results.json": _file_meta(analysis_paths["calibration"]),
        "overtrust_results.json": _file_meta(analysis_paths["overtrust"]),
    }
    manifest = {
        "artifact": "phase16_trust_calibration_results_v1",
        "schema_version": PHASE16_SCHEMA_VERSION,
        "status": "frozen",
        "frozen_utc": _utc_now(),
        "central_rq": protocol.get("central_rq"),
        "collection_mode": "final",
        "counts": {
            "participants_completed_valid": analysis.get("participants_completed_valid"),
            "trials_analyzed": analysis.get("trials_analyzed"),
            "correct_trials": ((analysis.get("condition_summary") or {}).get("correct") or {}).get("n_trials"),
            "incorrect_trials": ((analysis.get("condition_summary") or {}).get("incorrect") or {}).get("n_trials"),
        },
        "completion_gate": gate,
        "files": files,
    }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path
