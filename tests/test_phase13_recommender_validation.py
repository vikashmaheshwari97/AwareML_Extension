from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from awareml.engine.pareto_spec import (
    CANONICAL_EPSILON,
    PARETO_SPEC_ID,
    epsilon_nondominated_mask,
    epsilon_pareto_mask,
    specification_dict,
)
from awareml.journal.recommender_validation import (
    BASELINES,
    EXPECTED_DATASETS,
    EXPECTED_RECOMMENDER_ROWS,
    build_selected_oof_wide,
    load_phase13_inputs,
    run_baseline_validation,
    run_energy_co2_sensitivity,
    write_near_pareto_specification,
)
from awareml.recommender.v2_ranking import rank_candidates


def test_canonical_pareto_spec_is_frozen_to_point_zero_five():
    spec = specification_dict()
    assert spec["spec_id"] == PARETO_SPEC_ID == "epsilon_pareto_v1"
    assert abs(float(spec["epsilon"]) - 0.05) < 1e-12
    assert abs(CANONICAL_EPSILON - 0.05) < 1e-12
    assert "all-higher-is-better" in str(spec["space"])


def test_epsilon_zero_recovers_exact_nondominance_in_normalized_space():
    normalized = pd.DataFrame(
        {
            "accuracy": [1.0, 0.8, 0.2],
            "speed": [1.0, 0.7, 0.9],
        },
        index=["A", "B", "C"],
    )
    mask = epsilon_nondominated_mask(normalized, epsilon=0.0)
    assert bool(mask.loc["A"])
    assert not bool(mask.loc["B"])
    assert not bool(mask.loc["C"])


def test_raw_epsilon_pareto_handles_minimize_and_maximize_objectives():
    frame = pd.DataFrame(
        {
            "accuracy": [0.90, 0.85, 0.70],
            "runtime": [10.0, 12.0, 20.0],
        },
        index=["A", "B", "C"],
    )
    mask = epsilon_pareto_mask(
        frame,
        directions={"accuracy": "max", "runtime": "min"},
        epsilon=0.0,
    )
    assert bool(mask.loc["A"])
    assert not bool(mask.loc["B"])
    assert not bool(mask.loc["C"])


def test_v2_ranking_emits_canonical_near_pareto_metadata():
    candidates = pd.DataFrame(
        {
            "framework": ["A", "B", "C", "D", "E"],
            "accuracy": [0.90, 0.88, 0.85, 0.82, 0.80],
            "runtime": [15.0, 12.0, 10.0, 9.0, 8.0],
            "energy": [0.30, 0.25, 0.22, 0.20, 0.18],
            "co2": [0.12, 0.10, 0.09, 0.08, 0.07],
        }
    )
    ranked, meta = rank_candidates(
        candidates,
        weights={"accuracy": 0.25, "runtime": 0.25, "energy": 0.25, "co2": 0.25},
    )
    assert "near_pareto" in ranked.columns
    assert "pareto_efficient" in ranked.columns
    assert ranked["near_pareto"].equals(ranked["pareto_efficient"])
    assert set(ranked["pareto_spec_id"]) == {PARETO_SPEC_ID}
    assert np.allclose(ranked["pareto_epsilon"], 0.05)
    assert meta["pareto_spec_id"] == PARETO_SPEC_ID
    assert abs(float(meta["pareto_epsilon"]) - 0.05) < 1e-12


def test_phase13_inputs_use_only_47_development_datasets():
    train, oof, manifest = load_phase13_inputs()
    assert len(train) == EXPECTED_RECOMMENDER_ROWS == 235
    assert train["dataset_id"].nunique() == EXPECTED_DATASETS == 47
    wide = build_selected_oof_wide(oof, manifest)
    assert len(wide) == 235
    assert wide["dataset_id"].nunique() == 47


def test_phase13_end_to_end_results_can_be_built_in_temp_directory(tmp_path):
    baseline_dir = tmp_path / "results"
    baseline = run_baseline_validation(output_dir=baseline_dir)
    assert set(baseline["table"]["baseline"].unique()) == set(BASELINES)
    assert (baseline["table"]["scenario"] == "OVERALL").any()
    assert (baseline_dir / "baseline_comparison_table.csv").exists()
    assert (baseline_dir / "baseline_conclusion.json").exists()

    sensitivity = run_energy_co2_sensitivity(output_dir=baseline_dir)
    assert set(sensitivity["summary"]["scenario"].unique()) == {
        "energy_only",
        "co2_only",
        "energy_plus_co2",
    }
    assert len(sensitivity["pairwise"]) == 3
    assert sensitivity["decision"]["held_out_23_dataset_contents_used"] is False

    spec = write_near_pareto_specification(output_dir=baseline_dir)
    assert spec["spec_id"] == PARETO_SPEC_ID
    assert (baseline_dir / "near_pareto_specification.json").exists()
    assert (baseline_dir / "near_pareto_specification.md").exists()
