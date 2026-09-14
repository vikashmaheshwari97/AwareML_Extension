from __future__ import annotations

import json
from pathlib import Path

import pytest

from awareml.llm.confirmatory_runtime_v32 import ConfirmatoryOllamaClientV32
from awareml.llm.journal_client import JournalModelLockError


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "journal" / "objective_selection_v32_runtime_lock.json"


def test_confirmatory_lock_preserves_model_digest_but_versions_runtime_separately():
    data = json.loads(LOCK.read_text(encoding="utf-8"))
    assert data["model_tag"] == "llama3:8b"
    assert data["model_digest"] == data["legacy_phase10_runtime"]["model_digest"]
    assert data["ollama_version"] == "0.34.0"
    assert data["legacy_phase10_runtime"]["ollama_version"] == "0.32.14"
    assert data["paired_evaluation_policy"]["do_not_describe_baseline_as_literal_phase10_runtime_replay"] is True


def test_confirmatory_generation_contract_matches_phase10_protocol():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    active = (ROOT / "data" / "journal" / "active_protocol.txt").read_text(encoding="utf-8").strip()
    protocol = json.loads((ROOT / "data" / "journal" / active).read_text(encoding="utf-8"))
    assert lock["generation"] == protocol["journal_llm"]["generation"]


def test_client_class_is_separate_from_legacy_runtime_lock():
    assert ConfirmatoryOllamaClientV32.__name__ == "ConfirmatoryOllamaClientV32"
