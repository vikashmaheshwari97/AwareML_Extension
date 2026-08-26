from awareml.analysis.fairness import SlidingFairness


def test_fairness_reports_worst_group_performance():
    f = SlidingFairness(window_size=100, positive_label=1, min_group_n=2)
    rows = [
        (1, 1, "A"),
        (0, 0, "A"),
        (1, 0, "B"),
        (0, 0, "B"),
    ]
    for yt, yp, g in rows:
        f.update(yt, yp, g)
    out = f.compute()
    assert out["status"] == "ok"
    assert out["worst_group_accuracy"] == 0.5
    assert 0.0 <= out["worst_group_macro_f1"] <= 1.0
    assert set(out["group_performance"]) == {"A", "B"}


def test_fairness_insufficient_support_is_not_zero_fairness():
    f = SlidingFairness(window_size=100, positive_label=1, min_group_n=3)
    f.update(1, 1, "A")
    f.update(0, 0, "A")
    f.update(1, 1, "B")
    out = f.compute()
    assert out["status"] == "insufficient_group_support"
    assert out["dp_diff"] is None
    assert out["worst_group_accuracy"] is None
