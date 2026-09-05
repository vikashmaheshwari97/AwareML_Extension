from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def test_three_path_ui_module():
    text = (ROOT / "awareml/ui_v2/copilot_three_path.py").read_text(encoding="utf-8")
    assert "Historical aggregation · not machine learning" in text
    assert "actual learned meta-recommender" in text

def test_audit_module_present():
    assert (ROOT / "awareml/recommender/meta_logs_v2_audit.py").exists()

def test_final_protocol():
    p = ROOT / "data/journal/recommender_final_test_protocol_v1/design/predeclared_protocol.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["held_out_dataset_count"] == 23
    assert data["expected_framework_runs"] == 345
    assert data["seeds"] == [42, 43, 44]
    assert data["run_protocol"]["max_samples"] == 30000
    assert data["run_protocol"]["window_size"] == 1000
    assert data["run_protocol"]["time_budget_sec"] == 60

def test_ground_truth_is_preference_conditioned():
    p = ROOT / "data/journal/recommender_final_test_protocol_v1/design/predeclared_protocol.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "preference profile" in data["ground_truth_definition"]
    assert len(data["preference_profiles"]) == 6
