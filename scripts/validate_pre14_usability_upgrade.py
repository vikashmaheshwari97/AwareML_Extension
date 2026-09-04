from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def has(path, token):
    p = ROOT / path
    return p.exists() and token in p.read_text(encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with (ROOT / path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

checks = {
    "copilot_simple_research_view": has("awareml/ui_v2/pages_copilot.py", "render_accessible_objective_interpretation"),
    "copilot_plan_summary": has("awareml/ui_v2/pages_copilot.py", "render_copilot_plan_summary"),
    "utility_not_confidence": has("awareml/ui_v2/pre14_usability.py", "not a probability, correctness score or model confidence"),
    "objective_review_persistent": has("awareml/ui_v2/pre14_usability.py", "objective_reviews.jsonl"),
    "human_objective_rerank": has("awareml/ui_v2/pre14_usability.py", "apply_human_objective_override"),
    "target_cardinality_guard": has("awareml/ui_v2/pre14_usability.py", "High-cardinality numeric target detected"),
    "fairness_support_guard": has("awareml/ui_v2/pre14_usability.py", "Fairness validity & support"),
    "3d_copilot_priority_toggle": has("awareml/ui_v2/pages_core.py", "preference_context"),
    "3d_one_click_inspect": has("awareml/ui_v2/pages_core.py", "quick_framework_selector"),
    "decision_winner_explained": has("awareml/ui_v2/pages_specialist.py", "render_decision_lab_explanation"),
    "fairness_common_label": has("awareml/ui_v2/pages_specialist.py", "Composite (common available gaps)"),
    "pre_post_calibration": has("awareml/ui_v2/pre14_usability.py", "Pre-run vs post-run recommendation comparison"),
    "drift_display_modes": has("awareml/ui_v2/pages_observatory.py", "prepare_drift_display"),
    "drift_alert_wording": has("awareml/ui_v2/pages_observatory.py", "Framework-level drift alerts"),
    "adaptation_visual_semantics": has("awareml/ui_v2/plots.py", "Explicit adaptation / refit"),
    "v31_selector_untouched_marker": (ROOT / "awareml/llm/objective_selection_v31.py").exists(),
}

print("=" * 88)
print("AwareML FINAL pre-Phase-14 usability/safety upgrade validation")
print("=" * 88)
failed = []
for name, ok in checks.items():
    print("{:<48} {}".format(name, "PASS" if ok else "FAIL"))
    if not ok:
        failed.append(name)

p12 = "data/journal/objective_selection_benchmark_v1/frozen/manifest.json"
p13 = "data/journal/recommender_multiobjective_validation_v1/frozen/manifest.json"
print("\nPhase-12 frozen manifest SHA256:", sha(p12))
print("Phase-13 frozen manifest SHA256:", sha(p13))
if failed:
    raise SystemExit("FAILED checks: " + ", ".join(failed))
print("\nFINAL pre-Phase-14 usability/safety validation: PASS")
print("=" * 88)
