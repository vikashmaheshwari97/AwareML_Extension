from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "meta" / "snapshots" / "meta_logs_v2.json"


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def main() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    records = payload["records"]

    print("=== TOP LEVEL ===")
    print("snapshot_id:", payload.get("snapshot_id"))
    print("schema_version:", payload.get("schema_version"))
    print("records:", len(records))
    print("payload keys:", sorted(payload.keys()))

    print("\n=== RUN MATRIX ===")
    keys = {
        (r["dataset_id"], r["framework"], int(r["seed"]))
        for r in records
    }
    print("unique (dataset, framework, seed):", len(keys))
    print("datasets:", len({r["dataset_id"] for r in records}))
    print(
        "framework counts:",
        dict(sorted(Counter(r["framework"] for r in records).items())),
    )
    print(
        "seed counts:",
        dict(sorted(Counter(int(r["seed"]) for r in records).items())),
    )

    print("\n=== RECORD TOP-LEVEL KEYS ===")
    all_keys = sorted({key for r in records for key in r})
    for key in all_keys:
        values = [r.get(key) for r in records]
        nulls = sum(v is None for v in values)
        types = Counter(type_name(v) for v in values)
        print(
            "{:<36s} nulls={:3d} types={}".format(
                key, nulls, dict(types)
            )
        )

    print("\n=== NESTED ROOTS ===")
    for key in all_keys:
        values = [r.get(key) for r in records]
        if any(isinstance(v, dict) for v in values):
            nested_keys = sorted(
                {
                    nested
                    for value in values
                    if isinstance(value, dict)
                    for nested in value.keys()
                }
            )
            print("{}: dict keys={}".format(key, nested_keys))
        elif any(isinstance(v, list) for v in values):
            lengths = [
                len(v)
                for v in values
                if isinstance(v, list)
            ]
            print(
                "{}: list min_len={} max_len={}".format(
                    key, min(lengths), max(lengths)
                )
            )

    print("\n=== DATASET PROVENANCE SAMPLE ===")
    prov = records[0].get("dataset_provenance") or {}
    print("keys:", sorted(prov.keys()))
    print("rows:", prov.get("rows"))
    print("feature_count:", prov.get("feature_count"))
    print("numeric_features:", len(prov.get("numeric_features") or []))
    print(
        "categorical_features:",
        len(prov.get("categorical_features") or []),
    )
    print("missing_fraction:", prov.get("missing_fraction"))
    print("target_distribution:", prov.get("target_distribution"))

    print("\n=== PRIMARY TARGET COVERAGE ===")
    fields = [
        "accuracy",
        "f1_macro",
        "runtime_sec",
        "energy_kwh",
        "co2_kg",
        "samples",
        "samples_processed",
        "throughput_samples_sec",
        "mean_prediction_latency_ms",
        "p95_prediction_latency_ms",
    ]
    for field in fields:
        n = sum(r.get(field) is not None for r in records)
        print("{:<36s} {}/{} non-null".format(field, n, len(records)))


if __name__ == "__main__":
    main()
