from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_v31_selector_has_no_private_intent_dependency():
    path = ROOT / "awareml" / "llm" / "objective_selection_v31.py"
    text = path.read_text(encoding="utf-8")
    assert "candidate_generation_intent.PRIVATE.csv" not in text
    assert "paraphrase_generation_intent.PRIVATE.csv" not in text
    assert "generation_intent.PRIVATE" not in text


def test_v31_semantic_recovery_and_nondecisive_phrases(tmp_path):
    from awareml.llm.objective_selection_v31 import HybridEvidenceGroundedObjectiveSelectorV31

    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{{USER_SCENARIO}}", encoding="utf-8")

    class FakeClient:
        root = tmp_path
        model = "llama3:8b"

        def generate_json(self, prompt_text):
            return {
                "status": "valid",
                "decisions": {
                    name: {"selected": False, "evidence": None, "confidence": "medium"}
                    for name in ["Accuracy", "Runtime", "Energy", "CO2"]
                },
                "uncertainties": [],
            }, {"model": self.model, "model_digest": "fake", "ollama_version": "fake"}

    selector = HybridEvidenceGroundedObjectiveSelectorV31(client=FakeClient(), prompt_path=prompt)
    result = selector.select(
        "The wearable needs trustworthy predictions and should keep operating for a long time between charges."
    )
    assert result.selected_objectives == ["Accuracy", "Energy"]
    assert selector.last_audit["semantic_recovery_used"] is True

    result2 = selector.select("The system must be suitable for sustained deployment.")
    assert result2.status == "ambiguous"
    assert result2.selected_objectives == []


def test_v31_malformed_llm_recovers_only_explicit_support(tmp_path):
    from awareml.llm.journal_client import JournalLLMResponseError
    from awareml.llm.objective_selection_v31 import HybridEvidenceGroundedObjectiveSelectorV31

    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{{USER_SCENARIO}}", encoding="utf-8")

    class BadClient:
        root = tmp_path
        model = "llama3:8b"

        def generate_json(self, prompt_text):
            raise JournalLLMResponseError("bad json")

    selector = HybridEvidenceGroundedObjectiveSelectorV31(client=BadClient(), prompt_path=prompt)
    result = selector.select("The road platform should return decisions promptly.")
    assert result.status == "valid"
    assert result.selected_objectives == ["Runtime"]
    assert result.fallback_used is True


def test_v31_paraphrase_denominator_is_50():
    text = (ROOT / "scripts" / "run_phase12_v31_paraphrase_diagnostic.py").read_text(encoding="utf-8")
    assert "if para_total != 50" in text
    assert '"paraphrase_evaluations": para_total' in text
    assert "60 inference calls = 10 BASE + 50 paraphrases" in text


def test_fresh_benchmark_design_restores_k4_and_human_coverage():
    path = ROOT / "data" / "journal" / "objective_selection_v31_development" / "benchmark_design_v31.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["target_k_prime_counts"]["4"] > 0
    assert payload["minimum_human_written_fraction"] >= 0.40
    assert payload["fresh_adversarial_set_required"] is True
    assert payload["paraphrase_reviewers_minimum"] >= 2


def test_clean_human_written_derivative_contains_no_blank_scenarios():
    path = ROOT / "data" / "journal" / "objective_selection_v31_development" / "human" / "human_written_scenarios.clean.csv"
    assert path.exists()
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert all(str(row.get("scenario") or "").strip() for row in rows)
    assert len(rows) == 6


def test_copilot_v31_ui_and_state_safety_markers():
    text = (ROOT / "awareml" / "ui_v2" / "pages_copilot.py").read_text(encoding="utf-8")
    helper = (ROOT / "awareml" / "ui_v2" / "copilot_v31_components.py").read_text(encoding="utf-8")
    assert "HybridEvidenceGroundedObjectiveSelectorV31" in text
    assert "clear_previous_copilot_result(state)" in text
    assert "render_copilot_clarification(state)" in text
    assert "Objective clarification required" in helper
    assert "Hybrid evidence-grounded V3.1 selection audit" in text
    assert "Objective-specific support" in helper
    assert "Copilot proposal failed:" not in text


def test_existing_pre14_fairness_and_pareto_fixes_remain_present():
    specialist = (ROOT / "awareml" / "ui_v2" / "pages_specialist.py").read_text(encoding="utf-8")
    core = (ROOT / "awareml" / "ui_v2" / "pages_core.py").read_text(encoding="utf-8")
    assert '"Equal opportunity": "equal_opportunity_diff"' in specialist
    assert "Composite fairness score ↑" in specialist
    assert "All fairness metrics" in specialist
    assert "ε-Pareto candidate" in core
