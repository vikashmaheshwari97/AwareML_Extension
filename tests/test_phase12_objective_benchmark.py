from pathlib import Path

from awareml.journal.objective_benchmark import (
    binary_metrics,
    classify_adversarial_case,
    evaluate_predictions,
    fleiss_kappa_binary,
    validate_generated_design,
)


ROOT = Path(__file__).resolve().parents[1]


def test_generated_design_has_required_counts_and_no_explicit_objective_names():
    summary = validate_generated_design(ROOT)
    assert summary["candidate_count"] == 120
    assert summary["k_prime_design_counts"] == {
        "1": 30,
        "2": 30,
        "3": 30,
        "4": 30,
    }
    assert summary["paraphrase_families"] == 10
    assert summary["paraphrase_evaluations"] == 50
    assert 10 <= summary["adversarial_cases"] <= 15


def test_binary_metrics_are_correct():
    m = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


def test_multilabel_evaluation_reports_exact_micro_macro_and_k():
    gt = [
        {
            "scenario_id": "S1",
            "ground_truth_objectives": ["Accuracy"],
            "k_prime": 1,
        },
        {
            "scenario_id": "S2",
            "ground_truth_objectives": ["Accuracy", "Energy"],
            "k_prime": 2,
        },
    ]
    pred = [
        {
            "scenario_id": "S1",
            "selected_objectives": ["Accuracy"],
            "status": "valid",
        },
        {
            "scenario_id": "S2",
            "selected_objectives": ["Accuracy", "Runtime"],
            "status": "valid",
        },
    ]
    m = evaluate_predictions(gt, pred)
    assert m["n"] == 2
    assert m["exact_match_rate"] == 0.5
    assert m["by_k_prime"]["1"]["exact_match_rate"] == 1.0
    assert m["by_k_prime"]["2"]["exact_match_rate"] == 0.0
    assert 0.0 <= m["micro_f1"] <= 1.0
    assert 0.0 <= m["macro_f1"] <= 1.0


def test_fleiss_kappa_perfect_agreement():
    kappa = fleiss_kappa_binary(
        [
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ]
    )
    assert kappa == 1.0


def test_adversarial_taxonomy():
    assert (
        classify_adversarial_case(
            "ambiguous", "sensible_default", "ambiguous", []
        )
        == "sensible_default"
    )
    assert (
        classify_adversarial_case(
            "out_of_scope", "out_of_scope_handling", "valid", ["Accuracy"]
        )
        == "over_selection"
    )
    assert (
        classify_adversarial_case(
            "contradictory", "contradiction_handling", "malformed", []
        )
        == "silent_failure"
    )
