from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

import requests

from awareml.config import settings


class OllamaClient:
    """Small local-first Ollama client used by the Phase-7 Copilot.

    The client only receives user goals and compact evidence dictionaries
    constructed by AwareML. Dataset rows are never accepted by this API.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_sec: float = 60.0,
    ):
        self.base_url = (
            base_url or settings.ollama_base_url
        ).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout_sec = float(timeout_sec)

    def status(self) -> Dict[str, Any]:
        try:
            response = requests.get(
                self.base_url + "/api/tags",
                timeout=3.0,
            )
            response.raise_for_status()
            payload = response.json()
            models = []
            for row in payload.get("models", []) or []:
                name = row.get("name") or row.get("model")
                if name:
                    models.append(str(name))
            return {
                "reachable": True,
                "models": models,
                "configured_model": self.model,
                "resolved_model": self.resolve_model(models),
                "error": None,
            }
        except Exception as exc:
            return {
                "reachable": False,
                "models": [],
                "configured_model": self.model,
                "resolved_model": self.model,
                "error": "{}: {}".format(
                    type(exc).__name__,
                    exc,
                ),
            }

    def resolve_model(self, models=None) -> str:
        models = list(models or [])
        if not models:
            return self.model
        if self.model in models:
            return self.model

        # Ollama may expose a fully-qualified tag while configuration contains
        # only the model stem.
        configured_stem = self.model.split(":")[0]
        for candidate in models:
            if candidate.split(":")[0] == configured_stem:
                return candidate
        return models[0]

    def _request(
        self,
        prompt: str,
        json_mode: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        status = self.status()
        if not status["reachable"]:
            raise RuntimeError(
                "Ollama is not reachable at {}.".format(
                    self.base_url
                )
            )

        model = status["resolved_model"]
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"

        response = requests.post(
            self.base_url + "/api/generate",
            json=payload,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        body = response.json()
        text = str(
            body.get("response", "")
        ).strip()
        if not text:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return text, {
            "source": "ollama",
            "model": model,
            "base_url": self.base_url,
        }

    def generate_text(
        self,
        prompt: str,
    ) -> Tuple[str, Dict[str, Any]]:
        return self._request(
            prompt,
            json_mode=False,
        )

    def generate_json(
        self,
        prompt: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        text, meta = self._request(
            prompt,
            json_mode=True,
        )

        try:
            return json.loads(text), meta
        except json.JSONDecodeError:
            # Local models occasionally wrap otherwise valid JSON in a short
            # preamble or fenced block. Extract one object, then validate it in
            # the caller with Pydantic.
            match = re.search(
                r"\{.*\}",
                text,
                flags=re.DOTALL,
            )
            if not match:
                raise
            return json.loads(
                match.group(0)
            ), meta
