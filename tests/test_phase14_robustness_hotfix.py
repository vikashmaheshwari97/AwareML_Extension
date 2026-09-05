from pathlib import Path

from awareml.analysis.fairness import SlidingFairness


ROOT = Path(__file__).resolve().parents[1]


def test_constant_predictor_is_flagged_without_changing_zero_gap():
    fair = SlidingFairness(
        window_size=100,
        positive_label=1,
        min_group_n=2,
    )

    for _ in range(5):
        fair.update(1, 1, "A", {1: 0.5, 0: 0.5})
        fair.update(0, 1, "B", {1: 0.5, 0: 0.5})

    result = fair.compute()

    assert result["prediction_behavior_status"] == "constant"
    assert result["probability_behavior_status"] == "constant"
    assert result["dp_diff"] == 0.0
    # Group B has no positive ground-truth cases in this synthetic
    # fixture, so its TPR is undefined. Equal Opportunity therefore
    # remains None/N/A rather than being fabricated as zero.
    assert result["equal_opportunity_diff"] is None
    assert result["fairness_interpretation_warning"]


def test_dutch_profile_declares_positive_label_one():
    profile = (
        ROOT
        / "data"
        / "demo"
        / "dutch_census_test_profile.json"
    )

    if profile.exists():
        import json

        data = json.loads(
            profile.read_text(encoding="utf-8")
        )

        assert data["target"] == "occupation_binary"
        assert data["sensitive_attribute"] == "sex"
        assert data["positive_label"] == 1
