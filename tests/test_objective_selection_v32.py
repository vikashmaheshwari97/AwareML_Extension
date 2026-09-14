from __future__ import annotations

from pathlib import Path

import pytest

from awareml.llm.journal_client import JournalLLMResponseError
from awareml.llm.objective_selection_v32 import (
    EvidenceGroundedObjectiveSelectorV32,
    semantic_support_map_v32,
)


class FakeClient:
    model = "llama3:8b"

    def __init__(self, root: Path, payload=None, error=None):
        self.root = root
        self.payload = payload
        self.error = error
        self.calls = 0

    def generate_json(self, prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return self.payload, {
            "model": self.model,
            "model_digest": "test-digest",
            "ollama_version": "test-version",
        }


def write_prompt(tmp_path: Path) -> Path:
    path = tmp_path / "prompt.txt"
    path.write_text("USER SCENARIO:\n{{USER_SCENARIO}}\n", encoding="utf-8")
    return path


def payload(**decisions):
    rows = {}
    for objective in ("Accuracy", "Runtime", "Energy", "CO2"):
        selected, evidence = decisions.get(objective, (False, None))
        rows[objective] = {
            "selected": selected,
            "evidence": evidence,
            "confidence": "high" if selected else "medium",
        }
    return {"status": "valid", "decisions": rows, "uncertainties": []}


def test_concept_guards_separate_energy_from_co2():
    support = semantic_support_map_v32(
        "The remote sensor uses a small battery and must preserve power between visits."
    )
    assert support["Energy"] is not None
    assert support["CO2"] is None


def test_benchmark_guard_rejects_unsupported_overselection(tmp_path):
    scenario = "The remote monitor uses a small battery and needs dependable predictions."
    client = FakeClient(
        tmp_path,
        payload=payload(
            Accuracy=(True, "dependable predictions"),
            Energy=(True, "small battery"),
            CO2=(True, "small battery"),
        ),
    )
    selector = EvidenceGroundedObjectiveSelectorV32(
        client=client, prompt_path=write_prompt(tmp_path), benchmark_mode=True
    )
    result = selector.select(scenario)
    assert result.status == "valid"
    assert result.selected_objectives == ["Accuracy", "Energy"]
    assert result.fallback_used is False
    assert selector.last_audit["decisions"]["CO2"]["accepted"] is False


def test_benchmark_mode_does_not_recover_llm_omissions(tmp_path):
    scenario = "The device uses a small battery and needs dependable predictions."
    client = FakeClient(
        tmp_path,
        payload=payload(Accuracy=(True, "dependable predictions")),
    )
    selector = EvidenceGroundedObjectiveSelectorV32(
        client=client, prompt_path=write_prompt(tmp_path), benchmark_mode=True
    )
    result = selector.select(scenario)
    assert result.selected_objectives == ["Accuracy"]
    assert result.fallback_used is False


def test_interactive_mode_can_recover_with_explicit_audit(tmp_path):
    scenario = "The device uses a small battery and needs dependable predictions."
    client = FakeClient(
        tmp_path,
        payload=payload(Accuracy=(True, "dependable predictions")),
    )
    selector = EvidenceGroundedObjectiveSelectorV32(
        client=client,
        prompt_path=write_prompt(tmp_path),
        benchmark_mode=False,
        allow_semantic_recovery=True,
    )
    result = selector.select(scenario)
    assert result.selected_objectives == ["Accuracy", "Energy"]
    assert result.fallback_used is True
    assert selector.last_audit["semantic_recovery_used"] is True


def test_malformed_llm_response_remains_malformed_in_benchmark(tmp_path):
    client = FakeClient(
        tmp_path,
        error=JournalLLMResponseError("bad json"),
    )
    selector = EvidenceGroundedObjectiveSelectorV32(
        client=client, prompt_path=write_prompt(tmp_path), benchmark_mode=True
    )
    result = selector.select("The warning must respond quickly when a hazard appears.")
    assert result.status == "malformed"
    assert result.selected_objectives == []
    assert result.fallback_used is False


def test_fabricated_evidence_quote_is_rejected(tmp_path):
    client = FakeClient(
        tmp_path,
        payload=payload(Runtime=(True, "low latency")),
    )
    selector = EvidenceGroundedObjectiveSelectorV32(
        client=client, prompt_path=write_prompt(tmp_path), benchmark_mode=True
    )
    result = selector.select("The warning should arrive promptly when a hazard appears.")
    assert result.status == "ambiguous"
    assert result.selected_objectives == []


def test_generic_request_abstains_before_llm(tmp_path):
    client = FakeClient(tmp_path, payload=payload())
    selector = EvidenceGroundedObjectiveSelectorV32(
        client=client, prompt_path=write_prompt(tmp_path), benchmark_mode=True
    )
    result = selector.select("Make it good.")
    assert result.status == "ambiguous"
    assert result.selected_objectives == []
    assert client.calls == 0


def test_explicit_same_objective_conflict_is_contradictory(tmp_path):
    client = FakeClient(tmp_path, payload=payload())
    selector = EvidenceGroundedObjectiveSelectorV32(
        client=client, prompt_path=write_prompt(tmp_path), benchmark_mode=True
    )
    result = selector.select(
        "Response time does not matter, but the alert must respond immediately when danger appears."
    )
    assert result.status == "contradictory"
    assert result.selected_objectives == []
    assert client.calls == 0
