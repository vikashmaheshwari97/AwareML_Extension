from __future__ import annotations

import json
from typing import Any, Optional
import requests

from awareml.config import settings


def ollama_status(base_url: Optional[str] = None) -> dict[str, Any]:
    """Return reachable status and locally installed Ollama model names."""
    url = (base_url or settings.ollama_base_url).rstrip("/")
    try:
        resp = requests.get(f"{url}/api/tags", timeout=2.5)
        resp.raise_for_status()
        payload = resp.json()
        models = []
        for item in payload.get("models", []) or []:
            name = item.get("name") or item.get("model")
            if name:
                models.append(str(name))
        return {"reachable": True, "models": models, "error": None}
    except Exception as exc:
        return {"reachable": False, "models": [], "error": f"{type(exc).__name__}: {exc}"}


class GroundedChat:
    """Local-first chat over derived experiment facts. Raw rows are intentionally excluded."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    @staticmethod
    def build_facts(results: list[dict[str, Any]], ranking: Optional[list[dict]] = None) -> dict[str, Any]:
        facts = {"frameworks": {}}
        for r in results:
            facts["frameworks"][r["framework"]] = {
                "accuracy": r.get("accuracy"),
                "f1_macro": r.get("f1_macro"),
                "runtime_sec": r.get("runtime_sec"),
                "energy_kwh": r.get("energy_kwh"),
                "co2_kg": r.get("co2_kg"),
                "drift_events": r.get("drift_events"),
                "fairness": r.get("fairness"),
                "explainability": r.get("explainability"),
                "backend": r.get("backend"),
                "sustainability": r.get("sustainability"),
            }
        if ranking is not None:
            facts["ranking"] = ranking
        return facts

    def answer(self, question: str, facts: dict[str, Any], use_llm: bool = False) -> tuple[str, dict]:
        if not question.strip():
            return "Ask about accuracy, drift, fairness, explanation diagnostics, sustainability, or ranking.", {"source": "deterministic"}
        if use_llm:
            return self._ollama_answer(question, facts)
        return self._deterministic_answer(question, facts), {"source": "deterministic"}

    def _ollama_answer(self, question: str, facts: dict[str, Any]) -> tuple[str, dict]:
        fact_json = json.dumps(facts, default=str)
        prompt = f"""You are the AwareML grounded explanation assistant.
Use ONLY the FACTS block below. Do not infer experimental results that are absent.
Every numerical or comparative claim must include a bracketed evidence key, for example
[frameworks.AutoClass.accuracy] or [ranking]. If the requested fact is unavailable, say so.
Do not claim that an explanation proves causality. Do not expose raw participant or raw dataset rows.
Explain technical terms plainly but preserve scientific uncertainty.

FACTS:
{fact_json}

QUESTION: {question}
"""
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=60,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            return text or self._deterministic_answer(question, facts), {"source": "ollama", "model": self.model}
        except Exception as e:
            return self._deterministic_answer(question, facts), {
                "source": "deterministic-fallback",
                "model": self.model,
                "warning": f"{type(e).__name__}: {e}",
            }

    def _deterministic_answer(self, question: str, facts: dict[str, Any]) -> str:
        frameworks = facts.get("frameworks", {})
        if not frameworks:
            return "No experiment results are available yet. Run the benchmark first."
        q = question.lower()
        metric = "accuracy"
        if "runtime" in q or "fast" in q: metric = "runtime_sec"
        elif "energy" in q: metric = "energy_kwh"
        elif "carbon" in q or "co2" in q: metric = "co2_kg"
        elif "f1" in q: metric = "f1_macro"
        vals = [(fw, data.get(metric)) for fw, data in frameworks.items() if data.get(metric) is not None]
        if not vals:
            return f"The requested metric '{metric}' is not measured in the current results."
        reverse = metric not in {"runtime_sec", "energy_kwh", "co2_kg"}
        best = sorted(vals, key=lambda x: x[1], reverse=reverse)[0]
        key = f"frameworks.{best[0]}.{metric}"
        return f"{best[0]} has the strongest observed value for {metric}: {best[1]:.4g} [{key}]. This describes the current run only."
