from awareml.analysis.fairness import SlidingFairness


def test_fairness_reports_group_support_and_gaps():
    f = SlidingFairness(window_size=100, positive_label=1, min_group_n=2)
    rows = [
        (1,1,"A"),(0,1,"A"),(1,1,"A"),(0,0,"A"),
        (1,0,"B"),(0,0,"B"),(1,1,"B"),(0,0,"B"),
    ]
    for r in rows: f.update(*r)
    out = f.compute()
    assert out["status"] == "ok"
    assert out["dp_diff"] is not None
    assert out["equalized_odds_gap"] is not None
