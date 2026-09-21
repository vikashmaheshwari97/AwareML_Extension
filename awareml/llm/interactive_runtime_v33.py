from __future__ import annotations

"""
Interactive Ollama runtime for AwareML Goal Copilot V3.3.

Why this exists
---------------
The frozen Phase-11R / Phase-12-v2 confirmatory runtime intentionally requires
an exact Ollama engine version (0.34.0). That is correct for confirmatory
research reproducibility, but it makes the interactive Goal Copilot fail when
Ollama receives a patch/update such as 0.34.1.

V3.3 is an interactive DEVELOPMENT selector, not a frozen confirmatory method.
This client therefore keeps the important hard guarantees while treating the
Ollama engine version as recorded provenance rather than a fatal condition.

Hard requirements retained:
- exact model tag
- exact model digest
- no silent model fallback
- same frozen generation settings inherited from the journal protocol
- same prompt/schema static hash verification inherited from the journal client

Relaxed requirement:
- Ollama engine version is recorded but not required to equal 0.34.0

Do NOT use this client for frozen Phase-12-v2 confirmatory reruns.
"""

from typing import Any, Dict, Optional
from pathlib import Path

from .confirmatory_runtime_v32 import ConfirmatoryOllamaClientV32
from .journal_client import JournalModelLockError


class InteractiveOllamaClientV33(ConfirmatoryOllamaClientV32):
    """Development-only runtime for interactive Goal Copilot V3.3."""

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

    def verify_runtime(self) -> Dict[str, Any]:
        """
        Verify the live runtime without enforcing the frozen Ollama engine
        version. Exact model identity/digest remains mandatory.
        """
        try:
            version_response = self.session.get(
                self.base_url + "/api/version",
                timeout=5.0,
            )
            version_response.raise_for_status()

            tags_response = self.session.get(
                self.base_url + "/api/tags",
                timeout=5.0,
            )
            tags_response.raise_for_status()
        except Exception as exc:
            raise JournalModelLockError(
                "Interactive V3.3 Ollama runtime is not reachable at {}: {}: {}".format(
                    self.base_url,
                    type(exc).__name__,
                    exc,
                )
            )

        actual_version = str(version_response.json().get("version"))
        models = tags_response.json().get("models") or []

        exact = None
        for row in list(models):
            name = row.get("name") or row.get("model")
            if name == self.model:
                exact = row
                break

        if exact is None:
            raise JournalModelLockError(
                "Required exact V3.3 model '{}' is not installed. "
                "Silent model fallback is forbidden.".format(self.model)
            )

        actual_digest = str(exact.get("digest") or "")
        expected_digest = str(self.frozen_runtime["model_digest"])

        if actual_digest != expected_digest:
            raise JournalModelLockError(
                "V3.3 model digest mismatch for '{}'. Expected {}, got {}. "
                "Model fallback or model replacement is forbidden.".format(
                    self.model,
                    expected_digest,
                    actual_digest,
                )
            )

        reference_version = str(
            self.confirmatory_runtime_lock.get("ollama_version")
            or self.frozen_runtime.get("ollama_version")
            or ""
        )

        return {
            "reachable": True,
            "model": self.model,
            "model_digest": actual_digest,
            "ollama_version": actual_version,
            "reference_ollama_version": reference_version,
            "runtime_version_match": actual_version == reference_version,
            "runtime_version_policy": "record_only_for_v33_interactive_development",
            "runtime_role": "v33_interactive_development_not_confirmatory",
            "runtime_lock_id": self.runtime_lock_id,
            "same_model_digest_as_phase10": True,
            "fallback_used": False,
        }
