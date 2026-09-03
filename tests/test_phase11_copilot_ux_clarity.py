from awareml.ui_v2.pages_copilot import _weighting_explanation


def test_weighting_explanation_for_all_four_objectives():
    text = _weighting_explanation(["Accuracy", "Runtime", "Energy", "CO2"])
    assert "4 objectives selected" in text
    assert "1/4 = 0.250" in text
    assert "0.000" in text


def test_weighting_explanation_for_three_objectives():
    text = _weighting_explanation(["Accuracy", "Energy", "CO2"])
    assert "3 objectives selected" in text
    assert "1/3 = 0.333" in text


def test_weighting_explanation_for_one_objective():
    text = _weighting_explanation(["Accuracy"])
    assert "1 objective selected" in text
    assert "1/1 = 1.000" in text
