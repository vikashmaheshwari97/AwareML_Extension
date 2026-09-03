from __future__ import annotations

import pytest

from awareml.llm.copilot import CopilotService
from awareml.llm.goal_parser import GoalParser, deterministic_goal_parse
from awareml.llm.journal_client import (
    JournalLLMResponseError,
    JournalModelLockError,
    StrictJournalOllamaClient,
)
from awareml.llm.objective_selection import (
    JournalObjectiveSelector,
    deterministic_objective_selection,
)
from awareml.llm.weighting import equal_weights_for_selected


def test_scenario_maps_to_clean_objective_subset():
    result = deterministic_objective_selection(
        "Suitable for deployment in a low-impact edge environment "
        "while still providing strong performance."
    )
    assert result.status == "valid"
    assert result.selected_objectives == ["Accuracy", "Energy", "CO2"]


def test_selected_subset_maps_to_equal_weights():
    weights, meta = equal_weights_for_selected(
        ["Accuracy", "Energy", "CO2"]
    )
    values = weights.normalized_dict()
    assert values["runtime"] == 0.0
    assert values["accuracy"] == pytest.approx(1.0 / 3.0)
    assert values["energy"] == pytest.approx(1.0 / 3.0)
    assert values["co2"] == pytest.approx(1.0 / 3.0)
    assert meta["policy_id"] == "equal_selected_v1"


def test_phase7_compatibility_properties_remain_available():
    parsed = deterministic_goal_parse(
        "High accuracy, low energy, fair and explainable."
    )
    assert parsed.fairness_required is True
    assert parsed.explainability_level == "high"
    assert set(parsed.primary_weights.normalized_dict()) == {
        "accuracy", "runtime", "energy", "co2"
    }


class _MalformedClient:
    model = "llama3:8b"
    root = None
    protocol = {"journal_llm": {"prompt_file": "unused"}}

    def generate_json(self, prompt):
        raise JournalLLMResponseError("malformed test JSON")


def test_malformed_llm_json_is_explicit_not_silent(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{{USER_SCENARIO}}", encoding="utf-8")

    client = _MalformedClient()
    client.root = tmp_path
    client.protocol["journal_llm"]["prompt_file"] = "prompt.txt"

    selector = JournalObjectiveSelector(client=client)
    result = selector.select("strong performance on a battery device")
    assert result.status == "malformed"
    assert result.selected_objectives == []
    assert result.fallback_used is False

    parser = GoalParser(selector=selector)
    interpretation, meta = parser.parse(
        "strong performance on a battery device",
        use_llm=True,
        allow_malformed_fallback=True,
    )
    assert interpretation.selection_status == "malformed"
    assert interpretation.fallback_used is True
    assert meta["fallback_used"] is True
    assert set(interpretation.selected_objectives) == {"Accuracy", "Energy"}


def test_wrong_model_inventory_fails_benchmark_lock():
    with pytest.raises(JournalModelLockError):
        StrictJournalOllamaClient.validate_runtime_inventory(
            models=[{"name": "llama3.2:3b", "digest": "wrong"}],
            required_model_tag="llama3:8b",
            frozen_model_digest="expected-digest",
            ollama_version="0.32.14",
            frozen_ollama_version="0.32.14",
        )


def test_wrong_digest_fails_benchmark_lock():
    with pytest.raises(JournalModelLockError):
        StrictJournalOllamaClient.validate_runtime_inventory(
            models=[{"name": "llama3:8b", "digest": "wrong-digest"}],
            required_model_tag="llama3:8b",
            frozen_model_digest="expected-digest",
            ollama_version="0.32.14",
            frozen_ollama_version="0.32.14",
        )


def test_ambiguous_and_out_of_scope_are_explicit():
    ambiguous = deterministic_objective_selection("Make it good.")
    assert ambiguous.status == "ambiguous"
    assert ambiguous.selected_objectives == []

    out = deterministic_objective_selection(
        "Please optimize employee happiness at the office."
    )
    assert out.status in {"ambiguous", "out_of_scope"}


class _NeverUseRecommender:
    def recommend_profile(self, *args, **kwargs):
        raise AssertionError(
            "Context-free objective interpretation must not call Recommender V2."
        )


def test_copilot_goal_interpretation_does_not_require_dataset_or_recommender():
    service = CopilotService(
        recommender=_NeverUseRecommender(),
        goal_parser=GoalParser(),
    )
    interpretation, meta = service.interpret_goal(
        "High predictive performance with low energy use.",
        use_llm=False,
    )

    assert interpretation.selected_objectives == ["Accuracy", "Energy"]
    assert meta["dataset_context_used"] is False
    assert meta["framework_ranking_generated"] is False
    assert meta["weighting"]["policy_id"] == "equal_selected_v1"

    weights = interpretation.primary_weights.normalized_dict()
    assert weights["accuracy"] == pytest.approx(0.5)
    assert weights["energy"] == pytest.approx(0.5)
    assert weights["runtime"] == 0.0
    assert weights["co2"] == 0.0
