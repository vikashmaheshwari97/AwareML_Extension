from awareml.studies.trust import TrustCalibrationStudy


def test_wrong_condition_uses_lowest_utility_but_same_template():
    rows = [
        {"framework":"A", "utility":0.9},
        {"framework":"B", "utility":0.7},
        {"framework":"C", "utility":0.2},
    ]
    s = TrustCalibrationStudy(seed=1)
    case = s.build_case(rows, condition="wrong")
    assert case.shown_framework == "C"
    assert case.oracle_framework == "A"
    assert "balanced result" in case.explanation
