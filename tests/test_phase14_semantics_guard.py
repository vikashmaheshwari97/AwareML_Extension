from pathlib import Path

from awareml.analysis.fairness import SlidingFairness


def _build_fixture():
    fair = SlidingFairness(
        window_size=100,
        positive_label=1,
        min_group_n=2,
    )
    for _ in range(5):
        fair.update(1, 1, "A", {1: 0.5, 0: 0.5})
        fair.update(0, 1, "B", {1: 0.5, 0: 0.5})
    return fair.compute()


def test_equal_opportunity_is_na_when_any_group_has_no_positive_truth():
    result = _build_fixture()

    assert result["prediction_behavior_status"] == "constant"
    assert result["probability_behavior_status"] == "constant"
    assert result["dp_diff"] == 0.0

    # Group B contains no y_true == positive_label observations, therefore
    # TPR_B is undefined. Equal Opportunity must remain None/N/A.
    assert result["equal_opportunity_diff"] is None


def test_legacy_phase14_test_cannot_reintroduce_fake_zero():
    root = Path(__file__).resolve().parents[1]
    legacy = root / "tests" / "test_phase14_robustness_hotfix.py"
    text = legacy.read_text(encoding="utf-8")

    assert 'assert result["equal_opportunity_diff"] is None' in text
    assert 'assert result["equal_opportunity_diff"] == 0.0' not in text
