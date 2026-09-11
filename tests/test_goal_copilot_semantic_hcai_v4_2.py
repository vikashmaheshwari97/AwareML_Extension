from awareml.llm.objective_selection import infer_hcai_evidence, infer_hcai_requirements
from awareml.llm.objective_selection_v31 import semantic_support_map

SCENARIO_1 = (
    "A real-time healthcare screening service will run on resource-constrained hospital edge devices. "
    "It should make highly reliable predictions, respond quickly enough for clinical use, avoid unnecessary "
    "power consumption and environmental impact, treat patient groups consistently, and provide clear reasons "
    "that clinicians can understand before acting on its recommendations."
)
SCENARIO_2 = (
    "A roadside hazard-detection system will operate continuously on battery-powered edge hardware. "
    "It needs dependable predictions, very fast responses, long operating time between charges, "
    "and a small environmental footprint."
)
SCENARIO_3 = (
    "A conservation monitoring system will analyze wildlife observations on remote low-power devices. "
    "Its predictions should be dependable, preserve battery life, minimize environmental impact, "
    "work consistently across different monitored regions and populations, and provide understandable "
    "reasons for important alerts."
)

def supported(scenario):
    mapping = semantic_support_map(scenario)
    return {name for name, evidence in mapping.items() if evidence}

def test_scenario_1_objectives():
    assert supported(SCENARIO_1) == {"Accuracy", "Runtime", "Energy", "CO2"}

def test_scenario_2_objectives():
    assert supported(SCENARIO_2) == {"Accuracy", "Runtime", "Energy", "CO2"}

def test_scenario_3_objectives():
    assert supported(SCENARIO_3) == {"Accuracy", "Energy", "CO2"}

def test_scenario_1_hcai():
    result = infer_hcai_requirements(SCENARIO_1)
    assert result.fairness_required is True
    assert result.explainability_level == "high"

def test_scenario_3_hcai():
    result = infer_hcai_requirements(SCENARIO_3)
    assert result.fairness_required is True
    assert result.explainability_level == "high"

def test_defaults_are_marked_as_baselines():
    evidence = infer_hcai_evidence("The model should make dependable predictions.")
    assert evidence["drift"]["status"] == "platform_baseline"
    assert evidence["explainability"]["status"] == "platform_baseline"
    assert evidence["fairness"]["status"] == "not_requested"

def test_clear_reasons_is_high_explainability():
    evidence = infer_hcai_evidence(
        "Provide clear reasons that clinicians can understand before acting on recommendations."
    )
    assert evidence["explainability"]["status"] == "scenario_supported"
    assert evidence["explainability"]["value"] == "high"

def test_group_consistency_is_fairness():
    evidence = infer_hcai_evidence("Treat patient groups consistently.")
    assert evidence["fairness"]["required"] is True
    assert evidence["fairness"]["status"] == "scenario_supported"
