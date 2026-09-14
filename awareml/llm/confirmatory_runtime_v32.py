from __future__ import annotations

"""Phase-11R / Phase-12-v2 common confirmatory Ollama runtime.

This module deliberately does *not* alter the frozen Phase-10 journal protocol.
It subclasses the original strict journal client so that the Phase-11 V2 method
continues to use the exact frozen Phase-10 prompt/schema/generation contract,
while runtime verification is performed against a separate Phase-11R lock for
the fresh paired V2-method-vs-V3.2 confirmatory evaluation.

The current confirmatory lock records the same llama3:8b model digest as Phase
10 but Ollama 0.34.0 rather than the historical 0.32.14 engine. Therefore the
baseline must be reported as a V2 *method replay under the common confirmatory
runtime*, not as a literal rerun of the original Phase-10 runtime.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .journal_client import JournalModelLockError, StrictJournalOllamaClient


RUNTIME_LOCK_REL = Path("configs/journal/objective_selection_v32_runtime_lock.json")


class ConfirmatoryOllamaClientV32(StrictJournalOllamaClient):
    """Strict common runtime for Phase-12-v2 paired evaluation.

    Parent initialization still validates the immutable Phase-10 protocol,
    prompt and schema. Only the runtime inventory lock is replaced by the
    separately versioned Phase-11R/V3.2 confirmatory lock.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        base_url: Optional[str] = None,
        timeout_sec: float = 90.0,
        session=None,
    ):
        super().__init__(
            root=root,
            base_url=base_url,
            timeout_sec=timeout_sec,
            session=session,
        )
        lock_path = self.root / RUNTIME_LOCK_REL
        if not lock_path.exists():
            raise JournalModelLockError(
                "Phase-11R/V3.2 confirmatory runtime lock is missing: {}".format(lock_path)
            )
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise JournalModelLockError(
                "Could not parse V3.2 confirmatory runtime lock: {}: {}".format(
                    type(exc).__name__, exc
                )
            )

        if lock.get("release_status") != "pre_benchmark_lock":
            raise JournalModelLockError(
                "V3.2 confirmatory runtime lock must be pre_benchmark_lock before collection."
            )
        if str(lock.get("model_tag")) != str(self.model):
            raise JournalModelLockError(
                "Confirmatory model tag differs from the frozen journal model tag."
            )

        locked_generation = dict(lock.get("generation") or {})
        if locked_generation != dict(self.generation):
            raise JournalModelLockError(
                "Confirmatory generation settings differ from the frozen Phase-10 generation contract."
            )

        legacy = dict(lock.get("legacy_phase10_runtime") or {})
        if str(legacy.get("model_digest")) != str(lock.get("model_digest")):
            raise JournalModelLockError(
                "Confirmatory model digest must match the recorded Phase-10 llama3:8b digest."
            )

        self.confirmatory_runtime_lock: Dict[str, Any] = lock
        self.runtime_lock_id = str(lock.get("runtime_lock_id"))
        self.frozen_runtime = {
            "base_url": str(lock.get("base_url") or self.base_url),
            "ollama_version": str(lock["ollama_version"]),
            "model_digest": str(lock["model_digest"]),
        }
        if base_url is None:
            self.base_url = str(lock.get("base_url") or self.base_url).rstrip("/")

    def verify_runtime(self) -> Dict[str, Any]:
        result = super().verify_runtime()
        result["runtime_lock_id"] = self.runtime_lock_id
        result["runtime_role"] = "phase11r_phase12v2_common_confirmatory_runtime"
        result["legacy_phase10_ollama_version"] = str(
            self.confirmatory_runtime_lock["legacy_phase10_runtime"]["ollama_version"]
        )
        result["same_model_digest_as_phase10"] = True
        return result
