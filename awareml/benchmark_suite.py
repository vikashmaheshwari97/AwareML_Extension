from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import pandas as pd

from awareml.engine.runner import run_benchmark
from awareml.types import RunConfig


def run_repeated_suite(
    datasets: dict[str, pd.DataFrame],
    target_by_dataset: dict[str, str],
    seeds: Iterable[int] = (42, 43, 44),
    frameworks: Iterable[str] | None = None,
    window_size: int = 500,
    max_samples: int = 5000,
    time_budget_sec: float = 60.0,
    track_sustainability: bool = False,
) -> pd.DataFrame:
    """Run a dataset x seed x framework suite with a consistent prequential protocol."""
    rows = []
    for dataset_name, df in datasets.items():
        target = target_by_dataset[dataset_name]
        for seed in seeds:
            cfg = RunConfig(
                target=target,
                window_size=window_size,
                max_samples=max_samples,
                seed=int(seed),
                time_budget_sec=time_budget_sec,
                track_sustainability=track_sustainability,
            )
            results = run_benchmark(df, cfg, frameworks=frameworks)
            for r in results:
                rows.append({
                    "dataset_name": dataset_name,
                    "seed": seed,
                    "framework": r.framework,
                    "backend": r.backend,
                    "accuracy": r.accuracy,
                    "f1_macro": r.f1_macro,
                    "runtime_sec": r.runtime_sec,
                    "energy_kwh": r.energy_kwh,
                    "co2_kg": r.co2_kg,
                    "samples": r.samples,
                    "drift_count": len(r.drift_events),
                    "status": r.status,
                })
    return pd.DataFrame(rows)
