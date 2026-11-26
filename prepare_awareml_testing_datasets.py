#!/usr/bin/env python
"""
Prepare four benchmark CSVs for AwareML Extension dashboard testing.

The script NEVER overwrites the source datasets.

It creates two versions of each dataset:

1) *_awareml_ready.csv
   - cleaned, reproducible benchmark version
   - canonical binary target column: target
   - canonical sensitive attribute: sensitive_group
   - obvious identifier/target-leakage columns removed
   - original row order preserved
   - no synthetic drift added

2) *_awareml_stress.csv
   - starts from the ready version
   - contains controlled SYNTHETIC shifts for testing temporal/drift/fairness UI
   - should NOT be reported as a real dataset result
   - phase boundaries and modifications are recorded in JSON

Outputs are written to:
    data/Supplementary Data/External Evaluation Dataset/AwareML_Ready/

Typical usage from the AwareML_Extension project root:

    python scripts/prepare_awareml_testing_datasets.py

or:

    python scripts/prepare_awareml_testing_datasets.py ^
        --input-dir "data/Supplementary Data/External Evaluation Dataset" ^
        --output-dir "data/Supplementary Data/External Evaluation Dataset/AwareML_Ready"

Requirements:
    pandas
    numpy
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data") / "Testing Datasets"
DEFAULT_OUTPUT = DEFAULT_INPUT / "AwareML_Ready"
RNG_SEED = 42


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    preferred_filename: str
    target_source: str
    positive_values: Tuple[str, ...]
    negative_values: Tuple[str, ...]
    sensitive_source: str
    drop_columns: Tuple[str, ...]
    suggested_window: int
    stress_numeric_features: Tuple[str, ...]
    sensitive_labels: Optional[Dict[str, str]] = None


SPECS: Tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="census_income",
        preferred_filename="census_income_stream.csv",
        target_source="income_binary",
        positive_values=("1", "yes", "true"),
        negative_values=("0", "no", "false"),
        sensitive_source="sex",
        # "income" directly duplicates the classification label and would leak the target.
        drop_columns=("income",),
        suggested_window=1000,
        stress_numeric_features=("age", "wage_per_hour", "weeks_worked_in_year"),
        sensitive_labels={"female": "Female", "male": "Male"},
    ),
    DatasetSpec(
        key="credit_card",
        preferred_filename="credit_card_stream.csv",
        target_source="default_next_month",
        positive_values=("1", "yes", "true"),
        negative_values=("0", "no", "false"),
        # Prefer the readable sensitive attribute and remove the duplicate numeric sex code.
        sensitive_source="sex_label",
        drop_columns=("id", "sex"),
        suggested_window=1000,
        stress_numeric_features=("limit_bal", "bill_amt1", "pay_amt1"),
        sensitive_labels={"female": "Female", "male": "Male"},
    ),
    DatasetSpec(
        key="bank",
        preferred_filename="bank_stream.csv",
        target_source="y",
        positive_values=("yes", "1", "true"),
        negative_values=("no", "0", "false"),
        sensitive_source="age_group",
        drop_columns=(),
        suggested_window=500,
        stress_numeric_features=("balance", "campaign", "previous"),
        sensitive_labels={
            "age_lt_40": "Age < 40",
            "age_ge_40": "Age ≥ 40",
        },
    ),
    DatasetSpec(
        key="law",
        preferred_filename="law_dataset_stream.csv",
        target_source="pass_bar",
        positive_values=("1", "yes", "true"),
        negative_values=("0", "no", "false"),
        sensitive_source="race_group",
        drop_columns=(),
        suggested_window=1000,
        stress_numeric_features=("lsat", "ugpa", "fam_inc"),
        sensitive_labels=None,
    ),
)


def snake_case(name: str) -> str:
    text = str(name).strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def discover_source(input_dir: Path, spec: DatasetSpec) -> Path:
    """Find the intended source file while ignoring already-generated outputs."""
    exact = input_dir / spec.preferred_filename
    if exact.exists():
        return exact

    stem = Path(spec.preferred_filename).stem
    candidates = sorted(
        p
        for p in input_dir.glob(stem + "*.csv")
        if "_awareml_ready" not in p.name.lower()
        and "_awareml_stress" not in p.name.lower()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {spec.preferred_filename!r} in {input_dir}"
        )
    return candidates[0]


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim textual cells without treating legitimate 'unknown' categories as missing."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            s = out[col].astype("string").str.strip()
            # Only truly empty/placeholder missing values are converted to NA.
            lowered = s.str.lower()
            s = s.mask(lowered.isin({"", "?", "nan", "none", "null", "n/a", "na"}))
            out[col] = s
    return out


def encode_binary_target(series: pd.Series, spec: DatasetSpec) -> pd.Series:
    pos = {str(v).strip().lower() for v in spec.positive_values}
    neg = {str(v).strip().lower() for v in spec.negative_values}

    def convert(v):
        if pd.isna(v):
            return np.nan
        text = str(v).strip().lower()
        if text in pos:
            return 1
        if text in neg:
            return 0
        # Also accept numeric 0/1 values represented as floats.
        try:
            number = float(text)
            if number == 1:
                return 1
            if number == 0:
                return 0
        except Exception:
            pass
        return np.nan

    encoded = series.map(convert)
    bad = encoded.isna() & series.notna()
    if bad.any():
        examples = series[bad].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{spec.key}: unsupported target values in {spec.target_source}: {examples}"
        )
    return encoded.astype("Int64")


def readable_sensitive(series: pd.Series, spec: DatasetSpec) -> pd.Series:
    values = series.astype("string").str.strip()
    if spec.sensitive_labels:
        mapping = {str(k).lower(): v for k, v in spec.sensitive_labels.items()}
        values = values.map(
            lambda x: mapping.get(str(x).lower(), str(x)) if pd.notna(x) else pd.NA
        )
    return values.astype("string")


def drop_constant_columns(df: pd.DataFrame, protected: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    removed: List[str] = []
    keep_protected = set(protected)
    for col in list(df.columns):
        if col in keep_protected:
            continue
        if df[col].nunique(dropna=False) <= 1:
            removed.append(col)
    return df.drop(columns=removed, errors="ignore"), removed


def prepare_ready(source: Path, spec: DatasetSpec) -> Tuple[pd.DataFrame, Dict]:
    raw = pd.read_csv(source, low_memory=False)
    raw.columns = [snake_case(c) for c in raw.columns]
    raw = normalize_strings(raw)

    if spec.target_source not in raw.columns:
        raise KeyError(
            f"{spec.key}: expected target column {spec.target_source!r}, "
            f"found {list(raw.columns)}"
        )
    if spec.sensitive_source not in raw.columns:
        raise KeyError(
            f"{spec.key}: expected sensitive column {spec.sensitive_source!r}, "
            f"found {list(raw.columns)}"
        )

    original_rows = len(raw)
    duplicate_rows = int(raw.duplicated().sum())
    if duplicate_rows:
        raw = raw.drop_duplicates(keep="first").reset_index(drop=True)

    target = encode_binary_target(raw[spec.target_source], spec)
    sensitive = readable_sensitive(raw[spec.sensitive_source], spec)

    # Build model table. The original target is removed so selecting the canonical
    # "target" column cannot accidentally leave a duplicate label in the feature set.
    to_drop = set(spec.drop_columns) | {spec.target_source, spec.sensitive_source}
    ready = raw.drop(columns=[c for c in to_drop if c in raw.columns], errors="ignore")

    # Canonical columns expected by the testing workflow.
    ready["sensitive_group"] = sensitive
    ready["target"] = target

    # Rows without a valid target cannot be used for supervised evaluation.
    before_target_drop = len(ready)
    ready = ready[ready["target"].notna()].reset_index(drop=True)
    dropped_missing_target = before_target_drop - len(ready)
    ready["target"] = ready["target"].astype(int)

    # Fairness calculations require an observed group value. Keep the record but mark
    # an explicit category instead of silently dropping it.
    ready["sensitive_group"] = ready["sensitive_group"].fillna("Missing / unknown")

    ready, constant_columns = drop_constant_columns(
        ready, protected=("target", "sensitive_group")
    )

    # Put the two dashboard control columns at the end to make selection easy.
    feature_cols = [c for c in ready.columns if c not in {"sensitive_group", "target"}]
    ready = ready[feature_cols + ["sensitive_group", "target"]]

    target_counts = ready["target"].value_counts().sort_index().to_dict()
    group_counts = ready["sensitive_group"].value_counts(dropna=False).to_dict()

    report = {
        "dataset": spec.key,
        "source_file": source.name,
        "output_role": "cleaned_real_data",
        "synthetic": False,
        "original_rows": original_rows,
        "rows": len(ready),
        "columns": len(ready.columns),
        "duplicates_removed": duplicate_rows,
        "rows_dropped_missing_target": dropped_missing_target,
        "removed_columns": sorted([c for c in to_drop if c in raw.columns]),
        "constant_columns_removed": constant_columns,
        "canonical_target": "target",
        "canonical_sensitive_attribute": "sensitive_group",
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "positive_rate": float(ready["target"].mean()),
        "sensitive_group_counts": {str(k): int(v) for k, v in group_counts.items()},
        "suggested_window_size": spec.suggested_window,
        "preserves_original_row_order": True,
    }
    return ready, report


def _numeric_feature_candidates(df: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    result: List[str] = []
    for col in requested:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() >= max(50, int(0.5 * len(df))):
            result.append(col)
    return result


def make_stress_variant(
    ready: pd.DataFrame,
    spec: DatasetSpec,
    seed: int,
) -> Tuple[pd.DataFrame, Dict]:
    """Create a controlled synthetic shift version strictly for dashboard stress testing."""
    out = ready.copy(deep=True)
    rng = np.random.default_rng(seed)

    n = len(out)
    if n < 500:
        raise ValueError(f"{spec.key}: at least 500 rows are needed for the stress variant")

    # Four ordered phases. No phase label is written into the CSV because that could
    # become an accidental predictor; boundaries live only in the JSON manifest.
    b1 = int(round(n * 0.45))
    b2 = int(round(n * 0.65))
    b3 = int(round(n * 0.82))

    # 1) Covariate shift: move selected numeric variables by ~0.65 SD in phase 2 onward.
    shifted_numeric: Dict[str, Dict[str, float]] = {}
    for col in _numeric_feature_candidates(out, spec.stress_numeric_features):
        # Stress variants may require fractional shifts even when the original
        # feature is integer-valued, so promote the selected column to float first.
        numeric = pd.to_numeric(out[col], errors="coerce").astype(float)
        out[col] = numeric
        sd = float(numeric.iloc[:b1].std(skipna=True))
        if not math.isfinite(sd) or sd <= 0:
            continue
        delta = 0.65 * sd
        mask = numeric.iloc[b1:b2].notna()
        idx = numeric.iloc[b1:b2][mask].index
        out.loc[idx, col] = numeric.loc[idx] + delta
        shifted_numeric[col] = {"delta": float(delta), "sd_reference": sd}

    # 2) Concept stress: flip a reproducible 18% of labels in phase 3.
    phase3_idx = np.arange(b2, b3)
    concept_flip_n = max(1, int(round(len(phase3_idx) * 0.18)))
    concept_flip_idx = rng.choice(phase3_idx, size=concept_flip_n, replace=False)
    out.loc[concept_flip_idx, "target"] = 1 - out.loc[concept_flip_idx, "target"].astype(int)

    # 3) Fairness stress: find the group with the lower baseline positive rate and
    # reduce favorable labels in that group during the final phase.
    baseline = out.iloc[:b1]
    group_rates = (
        baseline.groupby("sensitive_group", dropna=False)["target"]
        .agg(["mean", "count"])
        .sort_values(["mean", "count"], ascending=[True, False])
    )
    eligible_groups = group_rates[group_rates["count"] >= 20]
    disadvantaged_group = None
    fairness_changed = 0
    if not eligible_groups.empty:
        disadvantaged_group = str(eligible_groups.index[0])
        final_mask = (
            (out.index >= b3)
            & (out["sensitive_group"].astype(str) == disadvantaged_group)
            & (out["target"].astype(int) == 1)
        )
        candidates = out.index[final_mask].to_numpy()
        if len(candidates):
            fairness_changed = max(1, int(round(len(candidates) * 0.35)))
            chosen = rng.choice(candidates, size=fairness_changed, replace=False)
            out.loc[chosen, "target"] = 0

    report = {
        "dataset": spec.key,
        "output_role": "synthetic_dashboard_stress_test",
        "synthetic": True,
        "warning": (
            "Synthetic test-only variant. Do not report its benchmark numbers as real "
            "dataset results."
        ),
        "seed": seed,
        "rows": n,
        "phase_boundaries_zero_based": {
            "baseline": [0, b1],
            "covariate_shift": [b1, b2],
            "concept_shift": [b2, b3],
            "fairness_stress": [b3, n],
        },
        "covariate_shift": {
            "features": shifted_numeric,
            "method": "add 0.65 reference-window standard deviations in covariate-shift phase",
        },
        "concept_shift": {
            "target_flip_fraction_requested": 0.18,
            "target_rows_flipped": int(concept_flip_n),
        },
        "fairness_stress": {
            "selected_group": disadvantaged_group,
            "positive_labels_changed_to_negative": int(fairness_changed),
            "method": "35% of available positive labels for baseline-lower-rate group in final phase",
        },
        "suggested_window_size": spec.suggested_window,
    }
    return out, report


def validate_output(df: pd.DataFrame, dataset_name: str) -> Dict:
    errors: List[str] = []
    warnings: List[str] = []

    if "target" not in df.columns:
        errors.append("missing canonical target column")
    if "sensitive_group" not in df.columns:
        errors.append("missing canonical sensitive_group column")

    if "target" in df.columns:
        values = sorted(pd.Series(df["target"]).dropna().unique().tolist())
        if values != [0, 1]:
            errors.append(f"target is not binary 0/1: {values}")
        counts = df["target"].value_counts()
        if len(counts) < 2:
            errors.append("target contains only one class")
        elif int(counts.min()) < 50:
            warnings.append("minority target class has fewer than 50 rows")

    if "sensitive_group" in df.columns:
        groups = df["sensitive_group"].value_counts(dropna=False)
        if len(groups) < 2:
            errors.append("sensitive_group has fewer than 2 groups")
        if len(groups) and int(groups.min()) < 30:
            warnings.append("at least one sensitive group has fewer than 30 rows")

    if df.duplicated().any():
        warnings.append("exact duplicate rows remain")

    # Detect very high missingness, but do not fail; some real benchmark columns are sparse.
    missing = df.isna().mean().sort_values(ascending=False)
    high_missing = missing[missing >= 0.80]
    if not high_missing.empty:
        warnings.append(
            "columns with >=80% true missing values: "
            + ", ".join(f"{k}={v:.1%}" for k, v in high_missing.items())
        )

    return {
        "dataset": dataset_name,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare AwareML-ready and synthetic stress-test CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--mode",
        choices=("ready", "stress", "both"),
        default="both",
        help="Which outputs to generate (default: both)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RNG_SEED,
        help=f"Seed for synthetic stress variants (default: {RNG_SEED})",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "artifact": "awareml_testing_dataset_preparation_v1",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "canonical_target": "target",
        "canonical_sensitive_attribute": "sensitive_group",
        "datasets": {},
    }

    print("=" * 78)
    print("AwareML testing dataset preparation")
    print("Input :", input_dir)
    print("Output:", output_dir)
    print("=" * 78)

    for i, spec in enumerate(SPECS):
        source = discover_source(input_dir, spec)
        ready, ready_report = prepare_ready(source, spec)
        validation = validate_output(ready, spec.key)

        entry: Dict[str, object] = {
            "source": str(source),
            "ready_report": ready_report,
            "ready_validation": validation,
        }

        if args.mode in {"ready", "both"}:
            ready_path = output_dir / f"{spec.key}_awareml_ready.csv"
            write_csv(ready, ready_path)
            entry["ready_file"] = str(ready_path)
            print(
                f"[READY] {spec.key:14s} rows={len(ready):7d} "
                f"positive={ready['target'].mean():6.2%} "
                f"groups={ready['sensitive_group'].nunique():2d} "
                f"-> {ready_path.name}"
            )

        if args.mode in {"stress", "both"}:
            stress, stress_report = make_stress_variant(
                ready, spec, seed=args.seed + i
            )
            stress_validation = validate_output(stress, spec.key + "_stress")
            stress_path = output_dir / f"{spec.key}_awareml_stress.csv"
            write_csv(stress, stress_path)
            entry["stress_file"] = str(stress_path)
            entry["stress_report"] = stress_report
            entry["stress_validation"] = stress_validation
            print(
                f"[STRESS] {spec.key:13s} rows={len(stress):7d} "
                f"positive={stress['target'].mean():6.2%} "
                f"-> {stress_path.name}"
            )

        manifest["datasets"][spec.key] = entry

    manifest_path = output_dir / "awareml_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    failures = []
    for name, entry in manifest["datasets"].items():
        for key in ("ready_validation", "stress_validation"):
            result = entry.get(key)
            if result and result.get("status") != "PASS":
                failures.append((name, key, result.get("errors")))

    print("-" * 78)
    print("Manifest:", manifest_path)
    if failures:
        print("Validation failures:")
        for failure in failures:
            print(" -", failure)
        return 2

    print("All generated datasets passed structural validation.")
    print()
    print("Dashboard selection:")
    print("  Target column           : target")
    print("  Sensitive attribute     : sensitive_group")
    print("  READY files             : normal benchmark / fairness / XAI testing")
    print("  STRESS files            : drift / temporal / fairness stress testing only")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
