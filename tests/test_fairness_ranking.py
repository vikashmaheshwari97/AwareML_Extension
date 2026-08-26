from awareml.engine.pareto import fairness_score_from_result


def test_selected_fairness_criterion_changes_score():
    fair = {
        "status": "ok",
        "dp_diff": 0.20,
        "equal_opportunity_diff": 0.05,
        "equalized_odds_gap": 0.10,
        "predictive_parity_diff": 0.30,
        "error_rate_gap": 0.08,
    }
    assert fairness_score_from_result(fair, "demographic_parity") == 0.80
    assert fairness_score_from_result(fair, "equal_opportunity") == 0.95
    composite = fairness_score_from_result(fair, "composite")
    assert 0.0 < composite < 1.0
