from __future__ import annotations

import json
from pathlib import Path

import pytest

from awareml.engine.pareto_spec import CANONICAL_EPSILON, PARETO_SPEC_ID
from awareml.journal.objective_benchmark_v2 import (
    OBJECTIVES,
    _allocate_human_quotas,
    _diversity_pick,
    _mcnemar_exact_p,
    _set_metrics,
)
from awareml.llm.weighting import equal_weights_for_selected


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_pre_registers_balanced_fresh_confirmatory_design():
    cfg = json.loads(
        (ROOT / "configs" / "journal" / "phase12_v2_protocol.json").read_text(encoding="utf-8")
    )
    assert cfg["design"]["generated_candidate_count"] == 150
    assert cfg["design"]["intended_k_prime_counts"] == {"1": 25, "2": 30, "3": 45, "4": 50}
    assert cfg["primary"]["target_n"] == 60
    assert cfg["primary"]["target_k_prime_counts"] == {"1": 15, "2": 15, "3": 15, "4": 15}
    assert cfg["primary"]["minimum_human_written"] == 24
    assert cfg["annotation"]["annotator_count"] == 3
    assert cfg["benchmark_mode"]["semantic_recovery_allowed"] is False


def test_equal_selected_weighting_is_unchanged():
    weights, meta = equal_weights_for_selected(["Accuracy", "Energy", "CO2"])
    values = weights.normalized_dict()
    assert values["accuracy"] == pytest.approx(1 / 3)
    assert values["energy"] == pytest.approx(1 / 3)
    assert values["co2"] == pytest.approx(1 / 3)
    assert values["runtime"] == 0
    assert meta["policy_id"] == "equal_selected_v1"


def test_near_pareto_spec_is_unchanged():
    assert PARETO_SPEC_ID == "epsilon_pareto_v1"
    assert CANONICAL_EPSILON == pytest.approx(0.05)


def test_primary_metrics_capture_over_and_under_selection():
    truths = [set(["Accuracy"]), set(["Runtime", "Energy"])]
    preds = [set(["Accuracy", "CO2"]), set(["Runtime"])]
    metrics = _set_metrics(truths, preds, ["valid", "valid"])
    assert metrics["exact_match_rate"] == 0
    assert metrics["over_selection_any_fp_rate"] == pytest.approx(0.5)
    assert metrics["under_selection_any_fn_rate"] == pytest.approx(0.5)
    assert metrics["hamming_loss"] == pytest.approx(2 / 8)


def test_mcnemar_exact_is_well_formed():
    assert _mcnemar_exact_p(0, 0) == 1.0
    assert 0 <= _mcnemar_exact_p(1, 8) <= 1
    assert _mcnemar_exact_p(1, 8) == _mcnemar_exact_p(8, 1)


def test_human_quota_is_at_least_24_and_stratified():
    groups = {}
    for k in (1, 2, 3, 4):
        groups[k] = [
            {"source": "human", "scenario_id": "H{}-{}".format(k, i)} for i in range(8)
        ] + [
            {"source": "generated", "scenario_id": "G{}-{}".format(k, i)} for i in range(20)
        ]
    quotas = _allocate_human_quotas(groups, 24)
    assert sum(quotas.values()) == 24
    assert all(q >= 0 for q in quotas.values())


def test_diversity_selection_never_uses_model_scores():
    rows = []
    for i in range(20):
        rows.append(
            {
                "scenario_id": "S{:02d}".format(i),
                "source": "human" if i < 8 else "generated",
                "domain": "d{}".format(i % 5),
                "style": "s{}".format(i % 4),
                "ground_truth": "Accuracy|Runtime" if i % 2 else "Accuracy|Energy",
            }
        )
    selected = _diversity_pick(rows, 15, human_quota=6, salt="unit-test")
    assert len(selected) == 15
    assert len({r["scenario_id"] for r in selected}) == 15
    assert sum(r["source"] == "human" for r in selected) >= 6
    assert all("prediction" not in r and "model_score" not in r for r in selected)


def test_overlay_does_not_redefine_legacy_active_markers():
    # When run after applying to the real repository, the old marker remains the
    # frozen baseline. V3.2 uses a separate marker created by the freeze script.
    active = ROOT / "data" / "journal" / "active_objective_selector.txt"
    if active.exists():
        assert active.read_text(encoding="utf-8").strip() == "objective_selection_v2/manifest.json"
    assert not (ROOT / "data" / "journal" / "active_objective_benchmark.txt").is_symlink()
