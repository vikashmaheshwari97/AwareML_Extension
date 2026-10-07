from __future__ import annotations

import json
import requests


class OllamaProbeClient:
    """Exact model selection, explicit stochastic settings, no silent fallback."""
    def __init__(self, model, base_url="http://127.0.0.1:11434", temperature=0.7, timeout=60):
        if not model.strip() or not 0 < temperature <= 2:
            raise ValueError("Supply an exact model tag and positive sampling temperature.")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.temperature, self.timeout = temperature, timeout

    def check_model(self):
        response = requests.get(self.base_url + "/api/tags", timeout=5)
        response.raise_for_status()
        names = {r.get("name") for r in response.json().get("models", [])}
        if self.model not in names:
            raise ValueError("Exact model tag is not installed: " + self.model)

    def generate(self, prompt, seed):
        options = {"temperature": self.temperature, "seed": int(seed), "num_predict": 240}
        response = requests.post(self.base_url + "/api/generate", timeout=self.timeout,
                                 json={"model": self.model, "prompt": prompt, "format": "json",
                                       "stream": False, "options": options})
        response.raise_for_status()
        body = response.json()
        if body.get("done_reason") == "length":
            raise ValueError("Generation truncated; increase the budget explicitly before rerunning.")
        parsed = json.loads(body["response"])
        if not isinstance(parsed, dict):
            raise ValueError("Model returned a non-object response.")
        parsed["model"] = body.get("model", self.model)
        parsed["generation"] = options
        return parsed
