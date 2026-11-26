from __future__ import annotations

from pathlib import Path

from awareml.llm.objective_selection_v3 import EvidenceGroundedObjectiveSelectorV3, SELECTOR_ID
from awareml.llm.grounded_chat import GroundedChat


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.root = ROOT
        self.model = "llama3:8b"
        self.calls = 0

    def generate_json(self, prompt):
        self.calls += 1
        return self.payload, {
            "source": "journal-ollama",
            "model": "llama3:8b",
            "model_digest": "fake-digest",
            "ollama_version": "0.32.14",
        }

    def verify_runtime(self):
        return {
            "model": "llama3:8b",
            "model_digest": "fake-digest",
            "ollama_version": "0.32.14",
        }


def payload(**selected):
    decisions = {}
    for objective in ["Accuracy", "Runtime", "Energy", "CO2"]:
        value = selected.get(objective)
        if value is None:
            decisions[objective] = {
                "selected": False,
                "evidence": None,
                "confidence": "medium",
            }
        else:
            evidence, confidence = value
            decisions[objective] = {
                "selected": True,
                "evidence": evidence,
                "confidence": confidence,
            }
    return {"status": "valid", "decisions": decisions, "uncertainties": []}


def test_v3_rejects_over_selection_without_semantic_support():
    scenario = "The city incident detector should react quickly enough for live operation."
    client = FakeClient(
        payload(
            Accuracy=("city incident detector", "high"),
            Runtime=("react quickly enough", "high"),
            Energy=("live operation", "high"),
            CO2=("city incident detector", "medium"),
        )
    )
    selector = EvidenceGroundedObjectiveSelectorV3(client=client)
    result = selector.select(scenario)
    assert result.selected_objectives == ["Runtime"]
    assert selector.last_audit["decisions"]["Accuracy"]["accepted"] is False
    assert selector.last_audit["decisions"]["Runtime"]["accepted"] is True


def test_v3_separates_energy_and_co2_semantics():
    scenario = "The field unit must preserve its small battery and keep its environmental footprint low."
    client = FakeClient(
        payload(
            Energy=("small battery", "high"),
            CO2=("environmental footprint", "high"),
        )
    )
    selector = EvidenceGroundedObjectiveSelectorV3(client=client)
    result = selector.select(scenario)
    assert result.selected_objectives == ["Energy", "CO2"]


def test_v3_low_confidence_selection_is_rejected():
    scenario = "The service must respond immediately."
    client = FakeClient(payload(Runtime=("respond immediately", "low")))
    selector = EvidenceGroundedObjectiveSelectorV3(client=client)
    result = selector.select(scenario)
    assert result.status == "ambiguous"
    assert result.selected_objectives == []


def test_v3_preflight_handles_generic_request_without_llm_call():
    client = FakeClient(payload(Accuracy=("good", "high")))
    selector = EvidenceGroundedObjectiveSelectorV3(client=client)
    result = selector.select("Make it good.")
    assert result.status == "ambiguous"
    assert result.selected_objectives == []
    assert client.calls == 0


def test_v3_preflight_handles_out_of_scope_without_llm_call():
    client = FakeClient(payload())
    selector = EvidenceGroundedObjectiveSelectorV3(client=client)
    result = selector.select("The only goal is employee morale.")
    assert result.status == "out_of_scope"
    assert client.calls == 0


def test_information_seeking_routes_different_questions_to_different_answers():
    facts = {
        "frameworks": {
            "AutoClass": {
                "accuracy": 0.80,
                "runtime_sec": 10.0,
                "energy_kwh": 0.002,
                "co2_kg": 0.001,
                "fairness": {"dp_diff": 0.1, "equal_opportunity_diff": 0.05},
                "drift_events": [1000],
                "explainability": {"status": "ok", "fidelity": 0.2, "stability": 0.6, "consistency": 0.7},
            },
            "OAML": {
                "accuracy": 0.75,
                "runtime_sec": 8.0,
                "energy_kwh": 0.001,
                "co2_kg": 0.0005,
                "fairness": {"dp_diff": 0.05, "equal_opportunity_diff": 0.03},
                "drift_events": [],
                "explainability": {"status": "ok", "fidelity": 0.18, "stability": 0.7, "consistency": 0.8},
            },
        },
        "ranking": [
            {"rank": 1, "framework": "AutoClass", "utility": 0.82},
            {"rank": 2, "framework": "OAML", "utility": 0.78},
        ],
    }
    chat = GroundedChat()
    why, _ = chat.answer("Why was AutoClass recommended?", facts, category="explanation_probe")
    compare, _ = chat.answer("Compare AutoClass and OAML", facts, category="counterfactual_or_comparison")
    fairness, _ = chat.answer("Show fairness evidence", facts, category="evidence_request")
    assert why != compare
    assert compare != fairness
    assert "AutoClass" in why
    assert "OAML" in compare
    assert "Fairness" in fairness or "DP=" in fairness


def test_selector_id_is_stable():
    assert SELECTOR_ID == "evidence_grounded_objective_selector_v3"
