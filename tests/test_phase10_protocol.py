from __future__ import annotations

import json
from pathlib import Path

from awareml.journal.protocol import (
    CANONICAL_OBJECTIVE_KEYS,
    CANONICAL_OBJECTIVE_LABELS,
    phase10_only_dirty,
    validate_static_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def test_phase10_objective_vocabulary_exact():
    cfg = json.loads(
        (ROOT / "configs/journal/objectives_v1.json").read_text(encoding="utf-8")
    )
    assert tuple(cfg["display_labels"]) == CANONICAL_OBJECTIVE_LABELS
    assert tuple(cfg["internal_keys"]) == CANONICAL_OBJECTIVE_KEYS


def test_phase10_schema_uses_literal_objective_labels():
    schema = json.loads(
        (ROOT / "configs/journal/objective_selection_schema_v1.json").read_text(
            encoding="utf-8"
        )
    )
    enum = schema["properties"]["selected_objectives"]["items"]["enum"]
    assert tuple(enum) == CANONICAL_OBJECTIVE_LABELS


def test_phase10_prompt_is_selection_not_weight_generation():
    prompt = (
        ROOT / "prompts/journal_objective_selection_v1.txt"
    ).read_text(encoding="utf-8")
    for label in CANONICAL_OBJECTIVE_LABELS:
        assert label in prompt
    assert "Do not assign weights." in prompt
    assert "Do not choose an AutoML framework." in prompt


def test_phase10_heldout_policy_resolves_role_without_inventing_ids():
    result = validate_static_inputs(ROOT, verify_git=False)
    split = result["dataset_policy"]
    heldout = split["canonical_role_resolution"]["final_heldout_evaluation"]
    assert heldout["expected_count"] == 23
    assert split["heldout_access_allowed_before_phase18"] is False
    assert split["heldout_dataset_contents_read_by_phase10"] is False


def test_phase10_journal_llm_is_exact_and_no_fallback():
    cfg = json.loads(
        (ROOT / "configs/journal/journal_llm_v1.json").read_text(encoding="utf-8")
    )
    assert cfg["required_model_tag"] == "llama3:8b"
    assert cfg["strict_exact_model"] is True
    assert cfg["allow_fallback"] is False
    assert cfg["generation"]["temperature"] == 0.0
    assert cfg["generation"]["seed"] == 42


def test_phase10_dirty_path_guard():
    ok, unexpected = phase10_only_dirty(
        [
            "awareml/journal/protocol.py",
            "configs/journal/objectives_v1.json",
            "data/journal/protocol_v1/journal_experimental_protocol_v1.json",
        ]
    )
    assert ok
    assert unexpected == []

    ok, unexpected = phase10_only_dirty(
        [
            "awareml/journal/protocol.py",
            "awareml/llm/goal_parser.py",
        ]
    )
    assert not ok
    assert "awareml/llm/goal_parser.py" in unexpected
