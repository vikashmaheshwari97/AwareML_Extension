from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import yaml


PRIMARY_OBJECTIVES = ("accuracy", "runtime_sec", "energy_kwh", "co2_kg")
EXPECTED_FRAMEWORKS = ("AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha")
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_DATASETS = 47
EXPECTED_RUN_ROWS = 705
EXPECTED_RECOMMENDER_ROWS = 235


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


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


def load_meta_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(
            "meta_logs_v2.json must be an object with a 'records' list."
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(
            "meta_logs_v2.json is missing the top-level 'records' list."
        )
    return payload


def load_registry(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if not isinstance(datasets, list):
        raise ValueError(
            "Audited registry must contain a 'datasets' list."
        )
    out = {}
    for row in datasets:
        dataset_id = str(row.get("dataset_id", "")).strip()
        if not dataset_id:
            raise ValueError("Registry row missing dataset_id.")
        if dataset_id in out:
            raise ValueError(
                "Duplicate dataset_id in registry: {}".format(dataset_id)
            )
        out[dataset_id] = dict(row)
    return out


def validate_run_matrix(
    payload: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> None:
    if len(records) != EXPECTED_RUN_ROWS:
        raise ValueError(
            "Expected {} run records; found {}.".format(
                EXPECTED_RUN_ROWS, len(records)
            )
        )

    keys = []
    framework_counts = Counter()
    seed_counts = Counter()
    dataset_counts = Counter()

    for r in records:
        dataset_id = str(r.get("dataset_id", ""))
        framework = str(r.get("framework", ""))
        seed = int(r.get("seed"))
        keys.append((dataset_id, framework, seed))
        framework_counts[framework] += 1
        seed_counts[seed] += 1
        dataset_counts[dataset_id] += 1

    if len(set(keys)) != EXPECTED_RUN_ROWS:
        duplicates = [
            key
            for key, count in Counter(keys).items()
            if count > 1
        ]
        raise ValueError(
            "Duplicate run keys: {}".format(duplicates[:10])
        )

    if set(framework_counts) != set(EXPECTED_FRAMEWORKS):
        raise ValueError(
            "Unexpected framework set: {}".format(
                sorted(framework_counts)
            )
        )

    for framework in EXPECTED_FRAMEWORKS:
        if framework_counts[framework] != 141:
            raise ValueError(
                "{} count must be 141, got {}.".format(
                    framework, framework_counts[framework]
                )
            )

    if set(seed_counts) != set(EXPECTED_SEEDS):
        raise ValueError(
            "Unexpected seeds: {}".format(dict(seed_counts))
        )

    for seed in EXPECTED_SEEDS:
        if seed_counts[seed] != 235:
            raise ValueError(
                "Seed {} count must be 235, got {}.".format(
                    seed, seed_counts[seed]
                )
            )

    if len(dataset_counts) != EXPECTED_DATASETS:
        raise ValueError(
            "Expected 47 datasets; found {}.".format(
                len(dataset_counts)
            )
        )

    bad = {
        dataset_id: count
        for dataset_id, count in dataset_counts.items()
        if count != 15
    }
    if bad:
        raise ValueError(
            "Each dataset must have 15 runs: {}".format(bad)
        )


def build_schema_report(
    payload: Dict[str, Any],
    records: List[Dict[str, Any]],
    input_path: Path,
) -> Dict[str, Any]:
    all_keys = sorted({key for r in records for key in r})
    key_report = {}

    for key in all_keys:
        values = [r.get(key) for r in records]
        key_report[key] = {
            "present_count": int(sum(key in r for r in records)),
            "null_count": int(sum(v is None for v in values)),
            "types": dict(
                sorted(
                    Counter(type_name(v) for v in values).items()
                )
            ),
        }

    nested_roots = sorted(
        key
        for key in all_keys
        if any(
            isinstance(r.get(key), (dict, list))
            for r in records
        )
    )

    return {
        "schema_report_version": "1.0",
        "input_file": input_path.name,
        "input_sha256": sha256_file(input_path),
        "snapshot_id": payload.get("snapshot_id"),
        "schema_version": payload.get("schema_version"),
        "record_count": len(records),
        "top_level_payload_keys": sorted(payload.keys()),
        "record_top_level_keys": all_keys,
        "nested_roots": nested_roots,
        "record_key_inventory": key_report,
    }


def flatten_records(
    records: List[Dict[str, Any]],
) -> pd.DataFrame:
    df = pd.json_normalize(records, sep="__")

    for col in list(df.columns):
        has_nested = df[col].map(
            lambda value: isinstance(
                value, (dict, list, tuple, set)
            )
        ).any()
        if has_nested:
            df[col] = df[col].map(
                lambda value: (
                    canonical_json(value)
                    if isinstance(
                        value, (dict, list, tuple, set)
                    )
                    else value
                )
            )

    first = [
        "dataset_id",
        "framework",
        "seed",
        "experiment_id",
        "campaign_id",
        "campaign_task_id",
        "backend",
        "status",
        "accuracy",
        "f1_macro",
        "runtime_sec",
        "energy_kwh",
        "co2_kg",
        "samples",
        "samples_processed",
        "max_samples_requested",
        "window_size",
        "time_budget_sec",
        "throughput_samples_sec",
        "mean_prediction_latency_ms",
        "p95_prediction_latency_ms",
        "dataset_sha256",
        "source_tree_sha256",
    ]
    ordered = [c for c in first if c in df.columns]
    ordered.extend(
        sorted(c for c in df.columns if c not in ordered)
    )
    return df[ordered].copy()


def target_distribution_features(
    provenance: Mapping[str, Any],
) -> Dict[str, float]:
    distribution = provenance.get("target_distribution") or {}
    values = distribution.get("values") or []

    fractions = []
    for item in values:
        try:
            fraction = float(item.get("fraction"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(fraction) and fraction > 0:
            fractions.append(fraction)

    n_classes = int(
        distribution.get("n_unique")
        or len(fractions)
        or 0
    )

    if not fractions:
        return {
            "n_classes": float(n_classes),
            "majority_class_fraction": np.nan,
            "minority_class_fraction": np.nan,
            "class_imbalance_ratio": np.nan,
            "class_entropy_normalized": np.nan,
        }

    majority = max(fractions)
    minority = min(fractions)
    imbalance = (
        majority / minority
        if minority > 0
        else np.nan
    )

    entropy = -sum(
        p * math.log(p)
        for p in fractions
        if p > 0
    )
    entropy_norm = (
        entropy / math.log(n_classes)
        if n_classes > 1
        else 0.0
    )

    return {
        "n_classes": float(n_classes),
        "majority_class_fraction": float(majority),
        "minority_class_fraction": float(minority),
        "class_imbalance_ratio": float(imbalance),
        "class_entropy_normalized": float(entropy_norm),
    }


def dataset_feature_row(
    record: Dict[str, Any],
    registry_row: Mapping[str, Any],
) -> Dict[str, Any]:
    provenance = record.get("dataset_provenance") or {}
    numeric_features = provenance.get("numeric_features") or []
    categorical_features = (
        provenance.get("categorical_features") or []
    )

    n_features = int(
        provenance.get("feature_count")
        or (
            len(numeric_features)
            + len(categorical_features)
        )
    )
    n_numeric = len(numeric_features)
    n_categorical = len(categorical_features)

    out = {
        "dataset_id": str(record["dataset_id"]),
        "dataset_family": registry_row.get("family"),
        "source_type": registry_row.get("source_type"),
        "drift_type": registry_row.get("drift_type"),
        "generator": registry_row.get("generator"),
        "is_synthetic": registry_row.get("is_synthetic"),
        "n_samples_dataset": int(
            provenance.get("rows") or 0
        ),
        "n_features": n_features,
        "n_numeric_features": int(n_numeric),
        "n_categorical_features": int(n_categorical),
        "numeric_feature_fraction": (
            float(n_numeric / n_features)
            if n_features
            else 0.0
        ),
        "categorical_feature_fraction": (
            float(n_categorical / n_features)
            if n_features
            else 0.0
        ),
        "missing_fraction": float(
            provenance.get("missing_fraction") or 0.0
        ),
        "source_size_bytes": int(
            provenance.get("source_size_bytes") or 0
        ),
        "dataset_sha256": (
            record.get("dataset_sha256")
            or provenance.get("source_sha256")
        ),
    }
    out.update(
        target_distribution_features(provenance)
    )
    return out


def assert_dataset_feature_consistency(
    records: List[Dict[str, Any]],
    registry: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    by_dataset = defaultdict(list)
    for record in records:
        by_dataset[str(record["dataset_id"])].append(
            record
        )

    if set(by_dataset) != set(registry):
        raise ValueError(
            "Registry and run datasets do not match."
        )

    rows = []
    compare_fields = [
        "n_samples_dataset",
        "n_features",
        "n_numeric_features",
        "n_categorical_features",
        "missing_fraction",
        "source_size_bytes",
        "n_classes",
        "majority_class_fraction",
        "minority_class_fraction",
        "class_imbalance_ratio",
        "class_entropy_normalized",
        "dataset_sha256",
    ]

    for dataset_id, group in sorted(
        by_dataset.items()
    ):
        candidates = [
            dataset_feature_row(
                record, registry[dataset_id]
            )
            for record in group
        ]
        first = candidates[0]

        for other in candidates[1:]:
            for field in compare_fields:
                a = first.get(field)
                b = other.get(field)

                if a is None and b is None:
                    continue

                if isinstance(a, (int, float)) and isinstance(
                    b, (int, float)
                ):
                    if not np.isclose(
                        float(a),
                        float(b),
                        equal_nan=True,
                        rtol=1e-12,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            "Inconsistent dataset feature "
                            "{}.{}: {!r} vs {!r}".format(
                                dataset_id, field, a, b
                            )
                        )
                elif a != b:
                    raise ValueError(
                        "Inconsistent dataset feature "
                        "{}.{}: {!r} vs {!r}".format(
                            dataset_id, field, a, b
                        )
                    )

        rows.append(first)

    return pd.DataFrame(rows)


def require_positive(
    df: pd.DataFrame,
    column: str,
) -> None:
    if column not in df.columns:
        raise ValueError(
            "Required column missing: {}".format(column)
        )
    values = pd.to_numeric(
        df[column], errors="coerce"
    )
    if values.isna().any():
        raise ValueError(
            "{} contains missing/non-numeric values.".format(
                column
            )
        )
    if (values <= 0).any():
        raise ValueError(
            "{} contains non-positive values.".format(
                column
            )
        )


def build_recommender_frame(
    run_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "dataset_id",
        "framework",
        "seed",
        "backend",
        "campaign_id",
        "accuracy",
        "f1_macro",
        "runtime_sec",
        "energy_kwh",
        "co2_kg",
        "samples_processed",
        "throughput_samples_sec",
        "mean_prediction_latency_ms",
        "p95_prediction_latency_ms",
        "window_size",
        "time_budget_sec",
        "max_samples_requested",
    ]
    missing = [
        c for c in required
        if c not in run_df.columns
    ]
    if missing:
        raise ValueError(
            "Run table missing recommender columns: {}".format(
                missing
            )
        )

    numeric = [
        "accuracy",
        "f1_macro",
        "runtime_sec",
        "energy_kwh",
        "co2_kg",
        "samples_processed",
        "throughput_samples_sec",
        "mean_prediction_latency_ms",
        "p95_prediction_latency_ms",
    ]
    for col in numeric:
        run_df[col] = pd.to_numeric(
            run_df[col], errors="coerce"
        )

    require_positive(run_df, "runtime_sec")
    require_positive(run_df, "energy_kwh")
    require_positive(run_df, "co2_kg")
    require_positive(run_df, "samples_processed")

    if run_df["accuracy"].isna().any():
        raise ValueError("accuracy contains nulls.")
    if (
        (run_df["accuracy"] < 0)
        | (run_df["accuracy"] > 1)
    ).any():
        raise ValueError(
            "accuracy must be in [0, 1]."
        )

    metric_cols = [
        "accuracy",
        "f1_macro",
        "runtime_sec",
        "energy_kwh",
        "co2_kg",
        "samples_processed",
        "throughput_samples_sec",
        "mean_prediction_latency_ms",
        "p95_prediction_latency_ms",
    ]

    rows = []
    grouped = run_df.groupby(
        ["dataset_id", "framework"],
        sort=True,
        observed=True,
    )

    for (
        dataset_id,
        framework,
    ), block in grouped:
        seeds = sorted(
            int(seed)
            for seed in block["seed"].tolist()
        )
        if seeds != list(EXPECTED_SEEDS):
            raise ValueError(
                "{}/{} has seeds {}, expected {}.".format(
                    dataset_id,
                    framework,
                    seeds,
                    list(EXPECTED_SEEDS),
                )
            )

        row = {
            "dataset_id": dataset_id,
            "framework": framework,
            "seed_count": int(len(block)),
            "seed_values": ",".join(
                str(seed) for seed in seeds
            ),
        }

        for col in metric_cols:
            values = pd.to_numeric(
                block[col], errors="coerce"
            )
            row[col + "_mean"] = float(
                values.mean()
            )
            row[col + "_std"] = float(
                values.std(ddof=1)
            )
            row[col + "_min"] = float(
                values.min()
            )
            row[col + "_max"] = float(
                values.max()
            )

        for protocol_col in [
            "window_size",
            "time_budget_sec",
            "max_samples_requested",
        ]:
            values = (
                block[protocol_col]
                .dropna()
                .unique()
                .tolist()
            )
            if len(values) != 1:
                raise ValueError(
                    "{}/{}: {} must be constant, got {}.".format(
                        dataset_id,
                        framework,
                        protocol_col,
                        values,
                    )
                )
            row[protocol_col] = values[0]

        backends = sorted(
            set(
                str(value)
                for value in block[
                    "backend"
                ].dropna().tolist()
            )
        )
        row["backend"] = (
            backends[0]
            if len(backends) == 1
            else canonical_json(backends)
        )

        campaigns = sorted(
            set(
                str(value)
                for value in block[
                    "campaign_id"
                ].dropna().tolist()
            )
        )
        row["campaign_id"] = (
            campaigns[0]
            if len(campaigns) == 1
            else canonical_json(campaigns)
        )

        rows.append(row)

    aggregated = pd.DataFrame(rows)

    if len(aggregated) != EXPECTED_RECOMMENDER_ROWS:
        raise ValueError(
            "Expected 235 dataset/framework rows; got {}.".format(
                len(aggregated)
            )
        )

    out = dataset_df.merge(
        aggregated,
        on="dataset_id",
        how="inner",
        validate="one_to_many",
    )

    if len(out) != EXPECTED_RECOMMENDER_ROWS:
        raise ValueError(
            "Recommender merge produced {} rows, expected 235.".format(
                len(out)
            )
        )

    leading = [
        "dataset_id",
        "framework",
        "dataset_family",
        "source_type",
        "drift_type",
        "generator",
        "is_synthetic",
        "n_samples_dataset",
        "n_features",
        "n_numeric_features",
        "n_categorical_features",
        "numeric_feature_fraction",
        "categorical_feature_fraction",
        "missing_fraction",
        "n_classes",
        "majority_class_fraction",
        "minority_class_fraction",
        "class_imbalance_ratio",
        "class_entropy_normalized",
        "source_size_bytes",
        "window_size",
        "time_budget_sec",
        "max_samples_requested",
        "seed_count",
        "seed_values",
        "backend",
        "campaign_id",
        "dataset_sha256",
    ]
    ordered = [
        c for c in leading
        if c in out.columns
    ]
    ordered.extend(
        sorted(
            c for c in out.columns
            if c not in ordered
        )
    )
    return out[ordered].copy()


def recommender_schema_report(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "schema_version": "2.0",
        "row_granularity": (
            "one row per dataset/framework, "
            "aggregated over seeds 42/43/44"
        ),
        "row_count": int(len(df)),
        "dataset_count": int(
            df["dataset_id"].nunique()
        ),
        "framework_count": int(
            df["framework"].nunique()
        ),
        "framework_counts": {
            str(key): int(value)
            for key, value in (
                df["framework"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "primary_objectives": [
            "accuracy",
            "runtime",
            "energy",
            "co2",
        ],
        "input_features": [
            "framework",
            "n_samples_dataset",
            "n_features",
            "n_numeric_features",
            "n_categorical_features",
            "numeric_feature_fraction",
            "categorical_feature_fraction",
            "missing_fraction",
            "n_classes",
            "majority_class_fraction",
            "minority_class_fraction",
            "class_imbalance_ratio",
            "class_entropy_normalized",
            "window_size",
            "time_budget_sec",
        ],
        "optional_categorical_features": [
            "dataset_family",
            "source_type",
            "drift_type",
        ],
        "targets": {
            "accuracy": [
                "accuracy_mean",
                "accuracy_std",
            ],
            "runtime": [
                "runtime_sec_mean",
                "runtime_sec_std",
            ],
            "energy": [
                "energy_kwh_mean",
                "energy_kwh_std",
            ],
            "co2": [
                "co2_kg_mean",
                "co2_kg_std",
            ],
        },
        "note": (
            "No fixed utility target is stored. "
            "Phase 6 predicts the four primary "
            "objectives separately and constructs "
            "preference-aware utility at inference "
            "or evaluation time."
        ),
        "columns": [
            {
                "name": str(col),
                "dtype": str(df[col].dtype),
                "null_count": int(
                    df[col].isna().sum()
                ),
                "unique_count": int(
                    df[col].nunique(dropna=True)
                ),
            }
            for col in df.columns
        ],
    }


def write_sha256(path: Path) -> Path:
    checksum_path = Path(
        str(path) + ".sha256"
    )
    checksum_path.write_text(
        "{}  {}\n".format(
            sha256_file(path),
            path.name,
        ),
        encoding="utf-8",
    )
    return checksum_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Build AwareML Phase-6.1 canonical "
            "Meta-Dataset V2 Parquet snapshots."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            root
            / "data"
            / "meta"
            / "snapshots"
            / "meta_logs_v2.json"
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=(
            root
            / "data"
            / "meta"
            / "registry"
            / "datasets_v1_47_audited.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            root
            / "data"
            / "meta"
            / "snapshots"
        ),
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    registry_path = args.registry.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = load_meta_payload(input_path)
    records = payload["records"]
    validate_run_matrix(payload, records)

    registry = load_registry(registry_path)
    if len(registry) != EXPECTED_DATASETS:
        raise ValueError(
            "Expected 47 audited registry rows; "
            "found {}.".format(len(registry))
        )

    schema_report = build_schema_report(
        payload,
        records,
        input_path,
    )

    run_df = flatten_records(records)
    dataset_df = assert_dataset_feature_consistency(
        records,
        registry,
    )
    recommender_df = build_recommender_frame(
        run_df.copy(),
        dataset_df,
    )

    meta_parquet = (
        output_dir / "meta_logs_v2.parquet"
    )
    recommender_parquet = (
        output_dir
        / "recommender_train_v2.parquet"
    )
    schema_json = (
        output_dir
        / "meta_logs_v2_schema_report.json"
    )
    recommender_schema_json = (
        output_dir
        / "recommender_train_v2_schema.json"
    )

    run_df.to_parquet(
        meta_parquet,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    recommender_df.to_parquet(
        recommender_parquet,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    schema_report[
        "flattened_column_count"
    ] = int(len(run_df.columns))
    schema_report["flattened_columns"] = [
        {
            "name": str(col),
            "dtype": str(run_df[col].dtype),
            "null_count": int(
                run_df[col].isna().sum()
            ),
            "unique_count": int(
                run_df[col].nunique(
                    dropna=True
                )
            ),
        }
        for col in run_df.columns
    ]

    schema_json.write_text(
        json.dumps(
            schema_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    recommender_schema_json.write_text(
        json.dumps(
            recommender_schema_report(
                recommender_df
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    outputs = [
        meta_parquet,
        recommender_parquet,
        schema_json,
        recommender_schema_json,
    ]
    for path in outputs:
        write_sha256(path)

    print("=" * 72)
    print(
        "AwareML Phase 6.1 — "
        "Meta-Dataset V2 build: SUCCESS"
    )
    print("=" * 72)
    print("Input:", input_path)
    print(
        "Input SHA256:",
        sha256_file(input_path),
    )
    print("Canonical run rows:", len(run_df))
    print(
        "Unique run keys:",
        len(
            run_df[
                [
                    "dataset_id",
                    "framework",
                    "seed",
                ]
            ].drop_duplicates()
        ),
    )
    print(
        "Datasets:",
        int(run_df["dataset_id"].nunique()),
    )
    print(
        "Frameworks:",
        int(run_df["framework"].nunique()),
    )
    print()
    print(
        "Recommender rows:",
        len(recommender_df),
    )
    print(
        "Recommender datasets:",
        int(
            recommender_df[
                "dataset_id"
            ].nunique()
        ),
    )
    print(
        "Recommender frameworks:",
        int(
            recommender_df[
                "framework"
            ].nunique()
        ),
    )
    print(
        "Seed count values:",
        sorted(
            recommender_df[
                "seed_count"
            ].astype(int).unique().tolist()
        ),
    )
    print()

    for metric in PRIMARY_OBJECTIVES:
        mean_col = metric + "_mean"
        std_col = metric + "_std"
        print(
            "{:<12s} mean_nulls={} std_nulls={}".format(
                metric,
                int(
                    recommender_df[
                        mean_col
                    ].isna().sum()
                ),
                int(
                    recommender_df[
                        std_col
                    ].isna().sum()
                ),
            )
        )

    print()
    for path in outputs:
        print("Wrote:", path)
        print(
            "SHA256:",
            sha256_file(path),
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
