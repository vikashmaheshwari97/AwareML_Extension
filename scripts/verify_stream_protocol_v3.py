"""Fast, read-only installation checks; not benchmark evidence."""
from pathlib import Path
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    checks = []
    for name in ("awareml.protocol_v3.campaign", "awareml.protocol_v3.execution",
                 "awareml.protocol_v3.collection", "awareml.protocol_v3.models",
                 "awareml.protocol_v3.evaluation", "awareml.protocol_v3.service"):
        importlib.import_module(name)
        checks.append({"check": name, "status": "pass"})
    cfg = json.loads((ROOT / "configs/stream_protocol_v3.json").read_text())
    assert cfg["scientific_time_limit_sec"] is None
    assert cfg["seeds"] == list(range(42, 47))
    assert cfg["profile_size"] == 1000 and cfg["max_samples"] == 100000
    assert cfg["track_energy"] and cfg["tracking_mode"] == "machine"
    assert cfg["safety_timeout_sec"] == 7200
    assert "Copilot Workspace" in (ROOT / "awareml/ui_v2/pages.py").read_text()
    assert "render_dataset_aware_tab()" in (ROOT / "awareml/ui_v2/pages_copilot.py").read_text()
    assert (ROOT / "awareml/ui_v2/recommender_bridge.py").is_file()
    checks.append({"check": "default protocol and application route", "status": "pass"})
    print(json.dumps({"status": "pass", "checks": checks,
                      "benchmark_results_created": False,
                      "next": "Run the unmeasured smoke test, then prepare a new campaign on the measurement host."}, indent=2))


if __name__ == "__main__":
    main()
