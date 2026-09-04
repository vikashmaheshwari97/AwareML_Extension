from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.engine.pareto import common_composite_fairness_keys
from awareml.ui_v2.research_evidence import load_phase12_objective_selection_reliability


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    specialist_path = ROOT / "awareml" / "ui_v2" / "pages_specialist.py"
    copilot_path = ROOT / "awareml" / "ui_v2" / "pages_copilot.py"
    core_path = ROOT / "awareml" / "ui_v2" / "pages_core.py"
    study_labs_path = ROOT / "awareml" / "ui_v2" / "study_labs_v3.py"

    specialist = specialist_path.read_text(encoding="utf-8")
    copilot = copilot_path.read_text(encoding="utf-8")
    core = core_path.read_text(encoding="utf-8")
    study_labs = study_labs_path.read_text(encoding="utf-8")

    checks = {
        "temporal_equal_opportunity_key": '"Equal opportunity": "equal_opportunity_diff"' in specialist,
        "fairness_complete_table": "All fairness metrics" in specialist,
        "comparable_composite_label": "Composite fairness score ↑" in specialist,
        "study_lab_delegation": (
            "return trust_calibration_research_page()" in specialist
            and "return information_seeking_research_page()" in specialist
        ),
        "information_seeking_routing": "category=category" in study_labs,
        "participant_mode_blinding": (
            "Participant mode" in study_labs and "Researcher-only manipulation check" in study_labs
        ),
        "phase12_reliability_panel": "Objective-selection reliability · frozen Phase 12" in copilot,
        "phase12_stale_wording_removed": "will be evaluated systematically in Phase 12" not in copilot,
        "epsilon_pareto_wording": "ε-Pareto candidate" in core,
    }

    phase12 = load_phase12_objective_selection_reliability(ROOT)
    checks["phase12_frozen_evidence_readable"] = bool(
        phase12
        and phase12.get("release_status") == "frozen"
        and int(phase12.get("benchmark_cases") or 0) == 55
    )

    phase12_manifest = (
        ROOT
        / "data"
        / "journal"
        / "objective_selection_benchmark_v1"
        / "frozen"
        / "manifest.json"
    )
    phase13_manifest = (
        ROOT
        / "data"
        / "journal"
        / "recommender_multiobjective_validation_v1"
        / "frozen"
        / "manifest.json"
    )
    checks["phase12_manifest_present"] = phase12_manifest.exists()
    checks["phase13_manifest_present"] = phase13_manifest.exists()

    failures = [name for name, ok in checks.items() if not ok]

    print("=" * 72)
    print("AwareML pre-Phase-14 hotfix validation")
    print("=" * 72)
    for name, ok in checks.items():
        print("{:<42} {}".format(name, "PASS" if ok else "FAIL"))
    print()
    if phase12_manifest.exists():
        print("Phase-12 frozen manifest SHA256:", sha256(phase12_manifest))
    if phase13_manifest.exists():
        print("Phase-13 frozen manifest SHA256:", sha256(phase13_manifest))
    print()
    if failures:
        print("FAILED checks:", ", ".join(failures))
        raise SystemExit(1)
    print("Pre-Phase-14 hotfix validation: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
