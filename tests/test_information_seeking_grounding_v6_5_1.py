from awareml.studies.information_seeking_grounding import grounded_information_answer

RESULTS = [
    {"framework":"OAML","accuracy":0.722,"f1_macro":0.7181,"runtime_sec":33.2709,"energy_kwh":0.0005776,"co2_kg":0.0002407},
    {"framework":"AutoClass","accuracy":0.7248,"f1_macro":0.72,"runtime_sec":52.9833,"energy_kwh":0.0009191,"co2_kg":0.0003830},
    {"framework":"AutoStreamML","accuracy":0.71,"runtime_sec":40.0,"energy_kwh":0.0007,"co2_kg":0.00029},
    {"framework":"EvoAutoML","accuracy":0.70,"runtime_sec":60.0,"energy_kwh":0.0010,"co2_kg":0.00042},
    {"framework":"ChaCha","accuracy":0.65,"runtime_sec":20.0,"energy_kwh":0.0005,"co2_kg":0.00021},
]
RANKING = [
    {"framework":"OAML","rank":1,"utility":0.9506},
    {"framework":"AutoClass","rank":2,"utility":0.7292},
    {"framework":"AutoStreamML","rank":3,"utility":0.66},
    {"framework":"ChaCha","rank":4,"utility":0.60},
    {"framework":"EvoAutoML","rank":5,"utility":0.55},
]

def test_compare_and_alternatives_are_distinct():
    compare, _, ct = grounded_information_answer("Please compare the top 2", "counterfactual_or_comparison", RESULTS, RANKING)
    alternatives, _, at = grounded_information_answer("Also please provide the alternatives", "counterfactual_or_comparison", RESULTS, RANKING)
    assert "Top-2 measured comparison" in compare
    assert "Ranked alternatives to OAML" in alternatives
    assert compare != alternatives
    assert ct == ["OAML", "AutoClass"]
    assert "EvoAutoML" in alternatives and "ChaCha" in alternatives


def test_energy_co2_question_answers_concept_not_only_numbers():
    answer, topics, targets = grounded_information_answer(
        "In sustainability what is better CO2 or energy in OAML?",
        "clarification",
        RESULTS,
        RANKING,
    )
    assert "neither is universally 'better'" in answer.lower()
    assert "electricity" in answer.lower()
    assert "climate impact" in answer.lower()
    assert topics == ["sustainability"]
    assert targets == ["OAML"]


def test_simple_explanation_is_human_readable():
    answer, _, _ = grounded_information_answer(
        "Explain the recommendation in simpler terms",
        "clarification",
        RESULTS,
        RANKING,
    )
    assert "In simple terms" in answer
    assert "not saying OAML is best at every single metric" in answer
