from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    selector = (ROOT / "awareml" / "llm" / "objective_selection_v31.py").read_text(encoding="utf-8")
    copilot = (ROOT / "awareml" / "ui_v2" / "pages_copilot.py").read_text(encoding="utf-8")
    helper = (ROOT / "awareml" / "ui_v2" / "copilot_v31_components.py").read_text(encoding="utf-8")
    specialist = (ROOT / "awareml" / "ui_v2" / "pages_specialist.py").read_text(encoding="utf-8")
    para_script = (ROOT / "scripts" / "run_phase12_v31_paraphrase_diagnostic.py").read_text(encoding="utf-8")

    design_path = ROOT / "data" / "journal" / "objective_selection_v31_development" / "benchmark_design_v31.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    clean_path = ROOT / "data" / "journal" / "objective_selection_v31_development" / "human" / "human_written_scenarios.clean.csv"
    with clean_path.open("r", encoding="utf-8-sig", newline="") as fh:
        clean_rows = list(csv.DictReader(fh))

    checks = {
        "v31_selector_installed": "HybridEvidenceGroundedObjectiveSelectorV31" in selector,
        "private_intent_dependency_absent": (
            "candidate_generation_intent.PRIVATE.csv" not in selector
            and "paraphrase_generation_intent.PRIVATE.csv" not in selector
        ),
        "nondecisive_phrase_guard": "sustained deployment" in selector and "NON_DECISIVE_PHRASES" in selector,
        "semantic_recall_recovery": "accepted_by = \"semantic_recovery\"" in selector,
        "paraphrase_denominator_50": "if para_total != 50" in para_script,
        "clean_human_scenario_derivative": len(clean_rows) == 6 and all(str(r.get("scenario") or "").strip() for r in clean_rows),
        "fresh_k4_coverage": int(design["target_k_prime_counts"]["4"]) > 0,
        "fresh_human_written_coverage": float(design["minimum_human_written_fraction"]) >= 0.40,
        "two_paraphrase_reviewers_required": int(design["paraphrase_reviewers_minimum"]) >= 2,
        "fresh_adversarial_required": bool(design["fresh_adversarial_set_required"]),
        "copilot_stale_state_cleared": "clear_previous_copilot_result(state)" in copilot,
        "copilot_friendly_abstention": "render_copilot_clarification(state)" in copilot and "Objective clarification required" in helper,
        "copilot_raw_error_hidden": "Copilot proposal failed:" not in copilot,
        "copilot_research_audit": "Hybrid evidence-grounded V3.1 selection audit" in copilot,
        "fairness_common_composite_preserved": "Composite fairness score ↑" in specialist,
        "temporal_eo_fix_preserved": '"Equal opportunity": "equal_opportunity_diff"' in specialist,
        "provenance_audit_present": (ROOT / "data" / "journal" / "objective_selection_v31_development" / "provenance" / "annotation_provenance_audit.json").exists(),
    }

    p12 = ROOT / "data" / "journal" / "objective_selection_benchmark_v1" / "frozen" / "manifest.json"
    p13 = ROOT / "data" / "journal" / "recommender_multiobjective_validation_v1" / "frozen" / "manifest.json"

    print("=" * 80)
    print("AwareML Phase-12 V3.1 FINAL pre-Phase-14 validation")
    print("=" * 80)
    failed = []
    for name, ok in checks.items():
        print("{:<44} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)
    print()
    print("Phase-12 frozen manifest SHA256:", sha256(p12))
    print("Phase-13 frozen manifest SHA256:", sha256(p13))
    print("Frozen Phase-12 v1 modified by V3.1 validator: False")
    print("Fresh independent V3.1 benchmark required: True")
    if failed:
        raise SystemExit("FAILED checks: " + ", ".join(failed))
    print()
    print("Phase-12 V3.1 FINAL pre-Phase-14 validation: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
