from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "journal" / "recommender_multiobjective_validation_v1"

ITEMS = [
    ("Baseline comparison", BASE / "results" / "baseline_comparison_table.csv"),
    ("Energy/CO2 report", BASE / "results" / "energy_co2_sensitivity_report.csv"),
    ("Near-Pareto spec", BASE / "results" / "near_pareto_specification.json"),
    ("Frozen manifest", BASE / "frozen" / "manifest.json"),
    ("Frozen SHA256", BASE / "frozen" / "manifest.json.sha256"),
]


def main() -> None:
    print("=" * 72)
    print("AwareML Phase 13 status")
    print("=" * 72)
    for label, path in ITEMS:
        print("{:<28} {}".format(label, "READY" if path.exists() else "PENDING"))
    print("=" * 72)


if __name__ == "__main__":
    main()
