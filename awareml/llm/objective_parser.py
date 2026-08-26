from __future__ import annotations

import json
import re
from typing import Optional
import requests
from pydantic import ValidationError

from awareml.config import settings
from awareml.recommender.schema import ObjectiveRequest


def _deterministic_parse(text: str) -> ObjectiveRequest:
    t = text.lower()
    w = {"accuracy": 0.55, "runtime": 0.15, "energy": 0.10, "co2": 0.10, "fairness": 0.05, "interpretability": 0.05}
    cues = {
        "accuracy": ["accur", "performance", "quality"],
        "runtime": ["fast", "latency", "runtime", "speed"],
        "energy": ["energy", "power", "efficient"],
        "co2": ["co2", "carbon", "emission", "sustain"],
        "fairness": ["fair", "bias", "equity"],
        "interpretability": ["explain", "interpret", "transparent"],
    }
    for key, words in cues.items():
        hits = sum(1 for word in words if word in t)
        if hits:
            w[key] += 0.15 * hits
    max_runtime = None
    m = re.search(r"(?:under|within|max(?:imum)?)[^0-9]{0,12}(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)", t)
    if m:
        max_runtime = float(m.group(1))
    return ObjectiveRequest(**w, max_runtime_sec=max_runtime)


def parse_objective_text(text: str, use_llm: bool = False, model: Optional[str] = None) -> tuple[ObjectiveRequest, dict]:
    if not text or not text.strip():
        return ObjectiveRequest(), {"source": "defaults", "warnings": []}
    if not use_llm:
        obj = _deterministic_parse(text)
        return obj, {"source": "deterministic", "warnings": []}

    schema = {
        "accuracy": "nonnegative number", "runtime": "nonnegative number", "energy": "nonnegative number",
        "co2": "nonnegative number", "fairness": "nonnegative number", "interpretability": "nonnegative number",
        "max_runtime_sec": "positive number or null", "fairness_metric": "string or null",
    }
    prompt = (
        "Convert the user's streaming AutoML goal to ONLY one JSON object matching this schema. "
        "Do not invent unsupported objectives. If uncertain, preserve conservative defaults.\n"
        f"Schema: {json.dumps(schema)}\nUser goal: {text}"
    )
    selected_model = model or settings.ollama_model
    try:
        resp = requests.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/generate",
            json={"model": selected_model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=45,
        )
        resp.raise_for_status()
        data = json.loads(resp.json().get("response", "{}"))
        obj = ObjectiveRequest.model_validate(data)
        return obj, {"source": "ollama", "model": selected_model, "warnings": []}
    except (requests.RequestException, json.JSONDecodeError, ValidationError, ValueError) as e:
        obj = _deterministic_parse(text)
        return obj, {
            "source": "deterministic-fallback",
            "model": selected_model,
            "warnings": [f"LLM objective parse rejected: {type(e).__name__}"],
        }
