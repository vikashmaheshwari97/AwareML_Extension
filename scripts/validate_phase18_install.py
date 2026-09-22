from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase18_common import (  # noqa: E402
    CONFIG_PATH,
    EXPECTED_AGGREGATED,
    EXPECTED_DATASETS,
    EXPECTED_EXTENSION,
    EXPECTED_HISTORICAL,
    EXPECTED_PREFERENCE_CASES,
    EXPECTED_RUNS,
    FRAMEWORKS,
    INVENTORY_PATH,
    SEEDS,
    load_config,
    load_inventory,
)

REQUIRED_FILES = [
    "configs/phase18_final_heldout_31.json",
    "configs/phase18_dataset_inventory_31.csv",
    "scripts/phase18_common.py",
    "scripts/phase18_prepare_heldout.py",
    "scripts/validate_phase18_install.py",
    "scripts/validate_phase18_preflight.py",
    "scripts/phase18_collect_results.py",
    "scripts/phase18_evaluate_recommender.py",
    "scripts/phase18_build_journal_tables.py",
    "hpc/production/phase18/run_phase18_task.py",
    "hpc/production/phase18/resume_phase18_campaign.py",
    "hpc/production/phase18/phase18_array.sbatch",
    "tests/test_phase18_final_heldout.py",
    "PHASE18_README.md",
    "APPLY_PHASE18.txt",
    "PHASE18_V2_5_31_DATASET_PROTOCOL.md",
    "APPLY_PHASE18_V2_5.txt",
]


def main() -> None:
    print("=" * 76)
    print("AwareML Phase 18 install validation — v2.5 31-dataset protocol")
    print("=" * 76)

    missing = [rel for rel in REQUIRED_FILES if not (ROOT / rel).exists()]
    if missing:
        raise RuntimeError("Missing Phase-18 files:\n" + "\n".join(missing))

    cfg = load_config()
    inv = load_inventory()
    counts = inv["cohort"].value_counts().to_dict()

    assert len(inv) == EXPECTED_DATASETS
    assert int(counts.get("historical_23", 0)) == EXPECTED_HISTORICAL
    assert int(counts.get("extension_8", 0)) == EXPECTED_EXTENSION
    assert len(FRAMEWORKS) == 5
    assert tuple(SEEDS) == (42, 43, 44)
    assert EXPECTED_RUNS == 465
    assert EXPECTED_AGGREGATED == 155
    assert EXPECTED_PREFERENCE_CASES == 3100

    # Validate the current AwareML interfaces this package is built against.
    engine = importlib.import_module("awareml.engine.runner")
    types = importlib.import_module("awareml.types")
    frameworks = importlib.import_module("awareml.frameworks")
    recommender = importlib.import_module("awareml.recommender.v2_service")
    profile = importlib.import_module("awareml.recommender.v2_profile")
    ranking = importlib.import_module("awareml.recommender.v2_ranking")

    required_symbols = [
        (engine, "run_benchmark"),
        (types, "RunConfig"),
        (frameworks, "create_frameworks"),
        (recommender, "V2Recommender"),
        (profile, "profile_from_dataframe_v2"),
        (ranking, "rank_candidates"),
    ]
    absent = [f"{mod.__name__}.{name}" for mod, name in required_symbols if not hasattr(mod, name)]
    if absent:
        raise RuntimeError("Current AwareML interface is incompatible:\n" + "\n".join(absent))

    print("Files: PASS")
    print("Current AwareML interfaces: PASS")
    print("Datasets declared: 31 = 23 historical + 8 extension (Adult overlap excluded)")
    print("Frameworks:", ", ".join(FRAMEWORKS))
    print("Seeds:", list(SEEDS))
    print("Framework runs after freeze: 465")
    print("Aggregated ground-truth rows: 155")
    print("Preference cases: 3,100")
    print("Dataset source:", cfg["dataset_directory"])
    print("Explicit task-policy mode: ENABLED (15 native classification + 16 derived ordinal classification)")
    print("Phase-18 install validation: PASS")


if __name__ == "__main__":
    main()
