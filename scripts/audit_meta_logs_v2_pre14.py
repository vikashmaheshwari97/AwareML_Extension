from __future__ import annotations

import json
from pathlib import Path

from awareml.recommender.meta_logs_v2_audit import audit_meta_logs_v2, historical_winner_sensitivity

ROOT = Path(__file__).resolve().parents[1]

def main():
    audit = audit_meta_logs_v2(ROOT)
    sensitivity = historical_winner_sensitivity()

    print("=" * 88)
    print("AwareML meta_logs_v2 audit — pre-Phase-14")
    print("=" * 88)
    print("Rows:", audit["rows"])
    print("Datasets:", audit["datasets"])
    print("Frameworks:", audit["frameworks"])
    print("Seeds:", audit["seeds"])
    print("Duplicate run keys:", audit["duplicate_dataset_framework_seed_rows"])
    print("Primary metric nulls:", audit["primary_metric_nulls"])
    print("Core integrity:", "PASS" if audit["core_integrity_pass"] else "CHECK")
    print("Time-budget values:", audit["time_budget_values"])
    print("Window-size values:", audit["window_size_values"])
    print("Runtime >=95% budget:", audit["runtime_near_budget_fraction_95"])
    print("Energy/CO2 Spearman:", audit["energy_co2_spearman"])
    print()
    print("Warnings / limitations:")
    for warning in audit["warnings"]:
        print(" -", warning)
    print()
    print("Historical winner sensitivity:")
    print(sensitivity.to_string(index=False))

    out = ROOT / "artifacts" / "pre14" / "meta_logs_v2_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(audit)
    payload["historical_winner_sensitivity"] = sensitivity.to_dict(orient="records")
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("Saved:", out)
    print("=" * 88)

if __name__ == "__main__":
    main()
