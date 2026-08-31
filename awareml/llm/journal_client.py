from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import requests


class JournalModelLockError(RuntimeError):
    pass


class JournalLLMResponseError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class StrictJournalOllamaClient:
    """Exact Phase-10 Ollama runtime for journal objective selection.

    Unlike the interactive Phase-7 client, this client never resolves to a
    different installed model. The exact tag, model digest, Ollama version,
    prompt hash, and schema hash come from Journal Experimental Protocol v1.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        base_url: Optional[str] = None,
        timeout_sec: float = 90.0,
        session=None,
    ):
        self.root = (
            Path(root).resolve()
            if root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.timeout_sec = float(timeout_sec)
        self.session = session or requests
        self.protocol = self._load_protocol()
        llm = self.protocol["journal_llm"]
        self.model = str(llm["required_model_tag"])
        self.frozen_runtime = dict(llm["runtime_lock"])
        self.generation = dict(llm["generation"])
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL")
            or self.frozen_runtime.get("base_url")
            or "http://localhost:11434"
        ).rstrip("/")
        self._verify_static_hashes()

    def _load_protocol(self) -> Dict[str, Any]:
        marker = self.root / "data" / "journal" / "active_protocol.txt"
        if not marker.exists():
            raise JournalModelLockError(
                "Phase-10 active journal protocol marker is missing."
            )

        rel = marker.read_text(encoding="utf-8").strip()
        protocol_path = self.root / "data" / "journal" / rel
        if not protocol_path.exists():
            raise JournalModelLockError(
                "Frozen journal protocol is missing: {}".format(protocol_path)
            )

        sha_path = Path(str(protocol_path) + ".sha256")
        if not sha_path.exists():
            raise JournalModelLockError("Journal protocol checksum file is missing.")

        expected_sha = sha_path.read_text(encoding="utf-8").strip().split()[0]
        if _sha256(protocol_path) != expected_sha:
            raise JournalModelLockError("Journal protocol checksum mismatch.")

        payload = json.loads(protocol_path.read_text(encoding="utf-8"))
        if payload.get("release_status") != "frozen":
            raise JournalModelLockError("Journal protocol is not frozen.")
        return payload

    def _verify_static_hashes(self) -> None:
        llm = self.protocol["journal_llm"]
        for file_key, sha_key in (
            ("prompt_file", "prompt_sha256"),
            ("schema_file", "schema_sha256"),
        ):
            path = self.root / llm[file_key]
            if not path.exists():
                raise JournalModelLockError(
                    "Frozen journal {} is missing: {}".format(file_key, path)
                )
            if _sha256(path) != llm[sha_key]:
                raise JournalModelLockError(
                    "Frozen journal {} changed after Phase 10.".format(file_key)
                )

    @staticmethod
    def validate_runtime_inventory(
        models,
        required_model_tag: str,
        frozen_model_digest: str,
        ollama_version: str,
        frozen_ollama_version: str,
    ) -> Dict[str, Any]:
        exact = None
        for row in list(models or []):
            name = row.get("name") or row.get("model")
            if name == required_model_tag:
                exact = row
                break

        if exact is None:
            raise JournalModelLockError(
                "Required exact journal model '{}' is not installed. "
                "Silent model fallback is forbidden.".format(required_model_tag)
            )

        digest = exact.get("digest")
        if digest != frozen_model_digest:
            raise JournalModelLockError(
                "Journal model digest mismatch for '{}'. Expected {}, got {}.".format(
                    required_model_tag,
                    frozen_model_digest,
                    digest,
                )
            )

        if str(ollama_version) != str(frozen_ollama_version):
            raise JournalModelLockError(
                "Ollama version mismatch. Expected {}, got {}.".format(
                    frozen_ollama_version,
                    ollama_version,
                )
            )

        return {
            "reachable": True,
            "model": required_model_tag,
            "model_digest": digest,
            "ollama_version": str(ollama_version),
            "fallback_used": False,
        }

    def verify_runtime(self) -> Dict[str, Any]:
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
                "Journal Ollama runtime is not reachable at {}: {}: {}".format(
                    self.base_url,
                    type(exc).__name__,
                    exc,
                )
            )

        version = version_response.json().get("version")
        models = tags_response.json().get("models") or []
        return self.validate_runtime_inventory(
            models=models,
            required_model_tag=self.model,
            frozen_model_digest=str(self.frozen_runtime["model_digest"]),
            ollama_version=str(version),
            frozen_ollama_version=str(self.frozen_runtime["ollama_version"]),
        )

    def status(self) -> Dict[str, Any]:
        try:
            result = self.verify_runtime()
            result["error"] = None
            return result
        except Exception as exc:
            return {
                "reachable": False,
                "model": self.model,
                "model_digest": self.frozen_runtime.get("model_digest"),
                "ollama_version": self.frozen_runtime.get("ollama_version"),
                "fallback_used": False,
                "error": "{}: {}".format(type(exc).__name__, exc),
            }

    def generate_json(self, prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        runtime = self.verify_runtime()

        options = {
            "temperature": float(self.generation["temperature"]),
            "top_p": float(self.generation["top_p"]),
            "seed": int(self.generation["seed"]),
            "num_predict": int(self.generation["num_predict"]),
        }
        payload = {
            "model": self.model,
            "prompt": str(prompt),
            "stream": False,
            "format": "json",
            "options": options,
        }

        try:
            response = self.session.post(
                self.base_url + "/api/generate",
                json=payload,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise JournalLLMResponseError(
                "Journal Ollama generation failed: {}: {}".format(
                    type(exc).__name__,
                    exc,
                )
            )

        text = str(body.get("response", "")).strip()
        if not text:
            raise JournalLLMResponseError("Journal Ollama returned an empty response.")

        # Deliberately strict: no regex extraction or fenced-JSON rescue in the
        # journal selector. Malformed JSON must remain visible to evaluation.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise JournalLLMResponseError(
                "Journal Ollama returned malformed JSON: {}".format(exc)
            )

        if not isinstance(parsed, dict):
            raise JournalLLMResponseError(
                "Journal Ollama response must be one JSON object."
            )

        return parsed, {
            "source": "journal-ollama",
            "model": self.model,
            "model_digest": runtime["model_digest"],
            "ollama_version": runtime["ollama_version"],
            "fallback_used": False,
            "generation": options,
        }
