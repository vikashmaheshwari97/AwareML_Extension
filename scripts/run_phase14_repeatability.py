from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import warnings

import pandas as pd
import psutil

from awareml.analysis.repeatability import (
    hardware_table,
    summarize_repeatability,
)
from awareml.analysis.repeatability_registry import (
    PAPER_READY_MIN_REPETITIONS,
    build_dataset_identity,
    canonical_dataframe_sha256,
    file_sha256,
    register_run,
    timestamp_id,
)
from awareml.engine.runner import run_benchmark
from awareml.types import RunConfig


DUTCH_FILENAME = "dutch_census_stream_awareml.csv"
DUTCH_TARGET = "occupation_binary"
DUTCH_SENSITIVE = "sex"
DUTCH_POSITIVE_LABEL = 1


def _resolve_csv(value):
    raw = Path(value).expanduser()
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            Path.cwd() / raw,
            Path.cwd() / "data" / "demo" / raw.name,
            Path.home() / "Downloads" / raw.name,
        ])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    attempted = "\n".join(
        "  - {}".format(candidate)
        for candidate in candidates
    )
    raise FileNotFoundError(
        "Dataset was not found. Checked:\n{}\n"
        "Tip: pass only the filename when it is in Downloads."
        .format(attempted)
    )


def _load_demo_profile(csv_path):
    profile_path = (
        Path.cwd()
        / "data"
        / "demo"
        / "dutch_census_test_profile.json"
    )
    if (
        csv_path.name.lower() == DUTCH_FILENAME
        and profile_path.exists()
    ):
        return json.loads(
            profile_path.read_text(encoding="utf-8")
        )
    return {}


def _resolve_schema(
    df,
    csv_path,
    target,
    sensitive,
    positive_label,
):
    profile = _load_demo_profile(csv_path)

    if csv_path.name.lower() == DUTCH_FILENAME and not profile:
        profile = {
            "target": DUTCH_TARGET,
            "sensitive_attribute": DUTCH_SENSITIVE,
            "positive_label": DUTCH_POSITIVE_LABEL,
        }

    target = target or profile.get("target")
    sensitive = (
        sensitive
        if sensitive is not None
        else profile.get("sensitive_attribute")
    )
    if positive_label is None:
        positive_label = profile.get("positive_label")

    if not target:
        raise ValueError(
            "Target was not supplied. Available columns: {}"
            .format(", ".join(map(str, df.columns)))
        )
    if target not in df.columns:
        raise ValueError(
            "Target column {!r} was not found. Available columns: {}"
            .format(
                target,
                ", ".join(map(str, df.columns)),
            )
        )
    if sensitive and sensitive not in df.columns:
        raise ValueError(
            "Sensitive attribute {!r} was not found. Available columns: {}"
            .format(
                sensitive,
                ", ".join(map(str, df.columns)),
            )
        )

    classes = df[target].dropna().unique().tolist()

    if positive_label is None:
        if 1 in classes:
            positive_label = 1
        elif "1" in classes:
            positive_label = "1"
        elif classes:
            positive_label = classes[-1]

    for class_value in classes:
        if str(class_value) == str(positive_label):
            positive_label = class_value
            break

    if positive_label not in classes:
        raise ValueError(
            "Positive label {!r} is not present in target {!r}. "
            "Classes: {}".format(
                positive_label,
                target,
                classes,
            )
        )

    return target, sensitive, positive_label, profile


def _active_streamlit_processes():
    found = []
    current_pid = os.getpid()

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info["pid"] == current_pid:
                continue
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "streamlit" in cmd.lower():
                found.append((proc.info["pid"], cmd))
        except Exception:
            continue

    return found


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run dataset-specific Phase-14 repeatability measurements. "
            "The default paper-ready protocol uses five independent repetitions."
        )
    )

    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--sensitive", default=None)
    parser.add_argument("--positive-label", default=None)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=PAPER_READY_MIN_REPETITIONS,
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44,45,46",
    )
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--time-budget-sec", type=float, default=60.0)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument(
        "--output-root",
        default="artifacts/phase14/repeatability",
    )
    parser.add_argument(
        "--allow-streamlit-running",
        action="store_true",
        help=(
            "Exploratory only. Allows Streamlit to coexist with "
            "CodeCarbon measurement."
        ),
    )

    args = parser.parse_args()

    active_streamlit = _active_streamlit_processes()
    if active_streamlit and not args.allow_streamlit_running:
        print("ERROR: Streamlit is still running.", file=sys.stderr)
        for pid, cmd in active_streamlit[:5]:
            print(
                "  PID {}: {}".format(pid, cmd),
                file=sys.stderr,
            )
        print(
            "\nStop Streamlit before paper-facing repeatability "
            "measurement. Concurrent application load can contaminate "
            "Runtime/Energy/CO2.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    warnings.filterwarnings(
        "ignore",
        message="divide by zero encountered in log",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in divide",
        category=RuntimeWarning,
    )

    csv_path = _resolve_csv(args.csv)
    df = pd.read_csv(csv_path)

    (
        target,
        sensitive,
        positive_label,
        profile,
    ) = _resolve_schema(
        df,
        csv_path,
        args.target,
        args.sensitive,
        args.positive_label,
    )

    repetitions = max(1, int(args.repetitions))
    seeds = [
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    ]

    if len(seeds) < repetitions:
        raise SystemExit(
            "Provide at least {} seeds. Current --seeds contains {}."
            .format(repetitions, len(seeds))
        )
    seeds = seeds[:repetitions]

    dataset_file_hash = file_sha256(csv_path)
    dataset_content_hash = canonical_dataframe_sha256(df)

    identity = build_dataset_identity(
        dataset_name=csv_path.name,
        dataset_content_sha256=dataset_content_hash,
        target=target,
        sensitive_attribute=sensitive,
        positive_label=positive_label,
    )

    output_root = Path(args.output_root)
    identity_root = output_root / identity["directory_name"]
    run_id = timestamp_id()
    run_dir = identity_root / run_id

    # Never overwrite an existing repeatability run.
    suffix = 1
    while run_dir.exists():
        run_dir = identity_root / "{}_{}".format(run_id, suffix)
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)

    paper_ready = repetitions >= PAPER_READY_MIN_REPETITIONS

    print("=" * 92)
    print("Phase-14 dataset-specific repeatability")
    print("Dataset:", csv_path)
    print("File SHA256:", dataset_file_hash)
    print("Content SHA256:", dataset_content_hash)
    print("Identity key:", identity["identity_key"])
    print("Rows:", len(df))
    print("Target:", target)
    print("Sensitive attribute:", sensitive or "not requested")
    print("Positive label:", positive_label)
    if profile:
        print(
            "Schema source: data/demo/dutch_census_test_profile.json"
        )
    print("Repetitions:", repetitions)
    print("Seeds:", seeds)
    print(
        "Paper-ready minimum repetitions:",
        PAPER_READY_MIN_REPETITIONS,
    )
    print(
        "Paper-ready status:",
        "YES" if paper_ready else "NO · exploratory/development evidence",
    )
    print("Output directory:", run_dir)
    print(
        "IMPORTANT: CodeCarbon measurement is exclusive; "
        "concurrent trackers are not allowed."
    )
    print("=" * 92)

    all_rows = []

    for repetition, seed in enumerate(seeds, start=1):
        config = RunConfig(
            target=target,
            sensitive_attribute=sensitive,
            window_size=args.window_size,
            max_samples=args.max_samples,
            seed=seed,
            time_budget_sec=args.time_budget_sec,
            positive_label=positive_label,
            track_sustainability=True,
            sustainability_warmup_sec=args.warmup_sec,
            sustainability_repetition_id=repetition,
            sustainability_repetitions_planned=repetitions,
        )

        print(
            "[{}/{}] seed={} ...".format(
                repetition,
                repetitions,
                seed,
            ),
            flush=True,
        )

        results = run_benchmark(df, config)

        for result in results:
            row = result.to_dict()
            row["repeatability_seed"] = seed
            row["repeatability_repetition"] = repetition
            row["source_csv"] = str(csv_path)
            row["target"] = target
            row["sensitive_attribute"] = sensitive
            row["positive_label"] = positive_label
            row["dataset_file_sha256"] = dataset_file_hash
            row["dataset_content_sha256"] = dataset_content_hash
            row["dataset_identity_key"] = identity["identity_key"]
            row["dataset_identity_directory"] = identity["directory_name"]

            provenance = dict(row.get("dataset_provenance") or {})
            provenance.update({
                "source_file_name": csv_path.name,
                "source_size_bytes": int(csv_path.stat().st_size),
                "source_sha256": dataset_file_hash,
                "content_sha256": dataset_content_hash,
                "dataset_identity_key": identity["identity_key"],
            })
            row["dataset_provenance"] = provenance

            all_rows.append(row)

    created_utc = datetime.now(timezone.utc).isoformat()

    results_path = run_dir / "phase14_repeated_results.json"
    results_path.write_text(
        json.dumps(
            all_rows,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    manifest = {
        "phase": 14,
        "schema_version": "phase14-repeatability-run-v2",
        "created_utc": created_utc,
        "run_id": run_dir.name,
        "dataset_path": str(csv_path),
        "dataset_name": csv_path.name,
        "dataset_file_sha256": dataset_file_hash,
        "dataset_content_sha256": dataset_content_hash,
        "dataset_identity_key": identity["identity_key"],
        "dataset_identity_directory": identity["directory_name"],
        "rows": int(len(df)),
        "target": target,
        "sensitive_attribute": sensitive,
        "positive_label": positive_label,
        "repetitions": repetitions,
        "paper_ready_min_repetitions": PAPER_READY_MIN_REPETITIONS,
        "paper_ready": paper_ready,
        "seeds": seeds,
        "window_size": int(args.window_size),
        "max_samples": int(args.max_samples),
        "time_budget_sec": float(args.time_budget_sec),
        "warmup_sec": float(args.warmup_sec),
        "measurement_concurrency": "exclusive",
        "profile_source": (
            "data/demo/dutch_census_test_profile.json"
            if profile
            else None
        ),
    }

    (run_dir / "repeatability_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    summary = summarize_repeatability(
        all_rows,
        min_repetitions=PAPER_READY_MIN_REPETITIONS,
    )
    hardware = hardware_table(all_rows)

    summary.to_csv(
        run_dir / "repeatability_table.csv",
        index=False,
    )
    hardware.to_csv(
        run_dir / "hardware_table.csv",
        index=False,
    )

    register_run(
        output_root,
        identity=identity,
        run_dir=run_dir,
        manifest=manifest,
    )

    print()
    print("Repeatability table")
    print(summary.to_string(index=False))
    print()
    print("Registry:", (output_root / "registry.json").resolve())
    print("Saved run:", run_dir.resolve())
    print()
    if paper_ready:
        print(
            "Phase-14 paper-ready repetition gate: PASS "
            "(>= {} repetitions).".format(
                PAPER_READY_MIN_REPETITIONS
            )
        )
    else:
        print(
            "Phase-14 paper-ready repetition gate: PENDING. "
            "This run is retained as exploratory evidence; it is not overwritten."
        )


if __name__ == "__main__":
    main()
