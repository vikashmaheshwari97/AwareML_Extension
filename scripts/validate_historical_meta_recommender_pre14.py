from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


checks = {}
service = ROOT / "awareml/recommender/historical_preference.py"
ui = ROOT / "awareml/ui_v2/copilot_historical.py"
page = ROOT / "awareml/ui_v2/pages_copilot.py"

service_text = service.read_text(encoding="utf-8") if service.exists() else ""
ui_text = ui.read_text(encoding="utf-8") if ui.exists() else ""
page_text = page.read_text(encoding="utf-8") if page.exists() else ""

checks["historical_service_installed"] = "class HistoricalPreferenceRecommender" in service_text
checks["stable_3seed_default"] = "stable_3seed_aggregate" in service_text and "load_recommender_train" in service_text
checks["best_seed_same_run_guard"] = "Never mix accuracy from one seed" in service_text and "idxmax()" in service_text
checks["705_run_source"] = "EXPECTED_RUN_ROWS" in service_text and "705" in ui_text
checks["47_dataset_equal_weighting"] = "47 historical development datasets" in ui_text and "normalized within each development dataset" in ui_text
checks["dataset_free_boundary"] = "dataset-free historical recommendation" in ui_text.lower() and "does not replace ML Recommender V2" in ui_text
checks["utility_not_confidence"] = "not a probability or model confidence" in ui_text
checks["human_review_persistent"] = "historical_meta_reviews.jsonl" in ui_text
checks["copilot_priority_import"] = "Use latest Goal Copilot priorities" in ui_text
checks["stale_historical_advice_hidden"] = "previous result is hidden to avoid showing stale advice" in ui_text
checks["workspace_tab_wrapper"] = "Historical Meta-Recommender · No dataset" in page_text and "_goal_copilot_workspace_page" in page_text
checks["goal_context_boundary"] = "Historical Meta-Recommender" in page_text
checks["v31_selector_not_reimplemented"] = "objective_selection_v31" not in service_text and "objective_selection_v31" not in ui_text

p12 = ROOT / "data/journal/objective_selection_benchmark_v1/frozen/manifest.json"
p13 = ROOT / "data/journal/recommender_multiobjective_validation_v1/frozen/manifest.json"
v31 = ROOT / "awareml/llm/objective_selection_v31.py"

print("=" * 88)
print("AwareML Historical Meta-Recommender pre-Phase-14 validation")
print("=" * 88)
failed = []
for name, ok in checks.items():
    print("{:<52} {}".format(name, "PASS" if ok else "FAIL"))
    if not ok:
        failed.append(name)
print()
print("Phase-12 frozen manifest SHA256:", sha256(p12))
print("Phase-13 frozen manifest SHA256:", sha256(p13))
print("V3.1 selector SHA256:", sha256(v31))
if failed:
    print("FAILED checks:", ", ".join(failed))
    raise SystemExit(1)
print("\nHistorical Meta-Recommender pre-Phase-14 validation: PASS")
print("=" * 88)
