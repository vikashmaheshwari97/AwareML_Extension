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
        return {
            "reachable": False,
            "models": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


class GroundedChat:
    """Local-first chat over derived experiment facts. Raw rows are excluded."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    @staticmethod
    def build_facts(
        results: list[dict[str, Any]],
        ranking: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
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

    def answer(
        self,
        question: str,
        facts: dict[str, Any],
        use_llm: bool = False,
        category: Optional[str] = None,
    ) -> tuple[str, dict]:
        if not question.strip():
            return (
                "Ask about accuracy, drift, fairness, explanation diagnostics, "
                "sustainability, ranking, or a framework comparison.",
                {"source": "deterministic"},
            )
        if use_llm:
            return self._ollama_answer(question, facts, category=category)
        return (
            self._deterministic_answer(question, facts, category=category),
            {"source": "deterministic", "category": category},
        )

    def _ollama_answer(
        self,
        question: str,
        facts: dict[str, Any],
        category: Optional[str] = None,
    ) -> tuple[str, dict]:
        fact_json = json.dumps(facts, default=str)
        prompt = f"""You are the AwareML grounded explanation assistant.
Use ONLY the FACTS block below. Do not infer experimental results that are absent.
Every numerical or comparative claim must include a bracketed evidence key, for example
[frameworks.AutoClass.accuracy] or [ranking]. If the requested fact is unavailable, say so.
Do not claim that an explanation proves causality. Do not expose raw participant or raw dataset rows.
Explain technical terms plainly but preserve scientific uncertainty.

FOLLOW_UP_CATEGORY: {category or 'unspecified'}
Respond to the user's actual information need. A comparison request must compare alternatives;
an evidence request must surface measured evidence; a challenge must acknowledge the strongest
counter-evidence or alternative; a clarification should define the requested concept rather than
repeating the same recommendation sentence.

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
            return (
                text
                or self._deterministic_answer(question, facts, category=category),
                {"source": "ollama", "model": self.model, "category": category},
            )
        except Exception as exc:
            return self._deterministic_answer(
                question,
                facts,
                category=category,
            ), {
                "source": "deterministic-fallback",
                "model": self.model,
                "category": category,
                "warning": f"{type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _ranking_rows(facts: dict[str, Any]) -> list:
        rows = list(facts.get("ranking") or [])
        if not rows:
            return []
        return sorted(
            rows,
            key=lambda row: (
                float(row.get("rank")) if row.get("rank") is not None else 1e9,
                -float(row.get("utility")) if row.get("utility") is not None else 0.0,
            ),
        )

    @staticmethod
    def _mentioned_frameworks(question: str, frameworks: dict[str, Any]) -> list:
        q = question.lower()
        return [name for name in frameworks if name.lower() in q]

    @staticmethod
    def _fmt(value, digits: int = 4) -> str:
        try:
            return ("{:.%dg}" % digits).format(float(value))
        except Exception:
            return "N/A"

    def _framework_summary(self, framework: str, data: dict[str, Any]) -> str:
        pieces = []
        for key, label in [
            ("accuracy", "accuracy"),
            ("runtime_sec", "runtime"),
            ("energy_kwh", "energy"),
            ("co2_kg", "CO2"),
        ]:
            value = data.get(key)
            if value is not None:
                pieces.append(
                    "{}={} [frameworks.{}.{}]".format(
                        label,
                        self._fmt(value),
                        framework,
                        key,
                    )
                )
        return "; ".join(pieces) if pieces else "no core metrics available"

    def _comparison_answer(
        self,
        question: str,
        facts: dict[str, Any],
    ) -> str:
        frameworks = facts.get("frameworks", {})
        mentioned = self._mentioned_frameworks(question, frameworks)
        if len(mentioned) < 2:
            ranking = self._ranking_rows(facts)
            for row in ranking:
                name = str(row.get("framework"))
                if name in frameworks and name not in mentioned:
                    mentioned.append(name)
                if len(mentioned) >= 2:
                    break
        if len(mentioned) < 2:
            mentioned = list(frameworks)[:2]
        if len(mentioned) < 2:
            return "A two-framework comparison is unavailable in the current evidence."

        left, right = mentioned[:2]
        a = frameworks[left]
        b = frameworks[right]
        return (
            "Comparison of {} and {}: {}. {}. Lower is better for runtime, energy "
            "and CO2; higher is better for accuracy. This comparison describes the "
            "current run only.".format(
                left,
                right,
                self._framework_summary(left, a),
                self._framework_summary(right, b),
            )
        )

    def _fairness_answer(self, question: str, facts: dict[str, Any]) -> str:
        frameworks = facts.get("frameworks", {})
        mentioned = self._mentioned_frameworks(question, frameworks)
        targets = mentioned or list(frameworks)
        rows = []
        for name in targets:
            fair = frameworks[name].get("fairness") or {}
            if not fair:
                continue
            vals = []
            for key, label in [
                ("dp_diff", "DP"),
                ("equal_opportunity_diff", "EO"),
                ("equalized_odds_gap", "EOD"),
                ("predictive_parity_diff", "PP"),
                ("error_rate_gap", "error-rate"),
            ]:
                value = fair.get(key)
                vals.append(
                    "{}={}".format(label, "N/A" if value is None else self._fmt(value))
                )
            rows.append(
                "{}: {} [frameworks.{}.fairness]".format(
                    name,
                    ", ".join(vals),
                    name,
                )
            )
        if not rows:
            return "Fairness evidence is not available in the current run."
        return (
            "Fairness is multi-criterion; lower disparity gaps are better and an "
            "unavailable value is not treated as zero. " + " ".join(rows)
        )

    def _drift_answer(self, question: str, facts: dict[str, Any]) -> str:
        frameworks = facts.get("frameworks", {})
        mentioned = self._mentioned_frameworks(question, frameworks)
        targets = mentioned or list(frameworks)
        rows = []
        for name in targets:
            events = frameworks[name].get("drift_events")
            if events is None:
                continue
            count = len(events) if isinstance(events, (list, tuple)) else events
            rows.append(
                "{}: {} recorded drift event(s) [frameworks.{}.drift_events]".format(
                    name,
                    count,
                    name,
                )
            )
        return " ".join(rows) if rows else "No drift evidence is available."

    def _xai_answer(self, question: str, facts: dict[str, Any]) -> str:
        frameworks = facts.get("frameworks", {})
        mentioned = self._mentioned_frameworks(question, frameworks)
        targets = mentioned or list(frameworks)
        rows = []
        for name in targets:
            exp = frameworks[name].get("explainability") or {}
            if not exp:
                continue
            rows.append(
                "{}: status={}, fidelity={}, stability={}, consistency={} "
                "[frameworks.{}.explainability]".format(
                    name,
                    exp.get("status", "N/A"),
                    "N/A" if exp.get("fidelity") is None else self._fmt(exp.get("fidelity")),
                    "N/A" if exp.get("stability") is None else self._fmt(exp.get("stability")),
                    "N/A" if exp.get("consistency") is None else self._fmt(exp.get("consistency")),
                    name,
                )
            )
        if not rows:
            return "Explanation diagnostics are not available in the current run."
        return " ".join(rows)

    def _recommendation_answer(self, facts: dict[str, Any], challenge: bool = False) -> str:
        ranking = self._ranking_rows(facts)
        frameworks = facts.get("frameworks", {})
        if not ranking:
            return self._metric_best_answer("accuracy", facts)
        top = ranking[0]
        name = str(top.get("framework"))
        utility = top.get("utility")
        text = "{} is ranked #1".format(name)
        if utility is not None:
            text += " with utility={} [ranking]".format(self._fmt(utility))
        if name in frameworks:
            text += ". Its current-run evidence is: {}.".format(
                self._framework_summary(name, frameworks[name])
            )
        if challenge and len(ranking) > 1:
            alt = ranking[1]
            alt_name = str(alt.get("framework"))
            text += " The strongest ranked alternative is {}".format(alt_name)
            if alt.get("utility") is not None:
                text += " with utility={} [ranking]".format(self._fmt(alt.get("utility")))
            text += "."
        return text

    def _evidence_answer(self, question: str, facts: dict[str, Any]) -> str:
        frameworks = facts.get("frameworks", {})
        mentioned = self._mentioned_frameworks(question, frameworks)
        if mentioned:
            name = mentioned[0]
            return "Measured evidence for {}: {}.".format(
                name,
                self._framework_summary(name, frameworks[name]),
            )
        return self._recommendation_answer(facts, challenge=True)

    def _metric_best_answer(self, metric: str, facts: dict[str, Any]) -> str:
        frameworks = facts.get("frameworks", {})
        vals = [
            (fw, data.get(metric))
            for fw, data in frameworks.items()
            if data.get(metric) is not None
        ]
        if not vals:
            return "The requested metric '{}' is not measured in the current results.".format(metric)
        reverse = metric not in {"runtime_sec", "energy_kwh", "co2_kg"}
        best = sorted(vals, key=lambda item: item[1], reverse=reverse)[0]
        key = "frameworks.{}.{}".format(best[0], metric)
        return (
            "{} has the strongest observed value for {}: {} [{}]. "
            "This describes the current run only.".format(
                best[0],
                metric,
                self._fmt(best[1]),
                key,
            )
        )

    def _deterministic_answer(
        self,
        question: str,
        facts: dict[str, Any],
        category: Optional[str] = None,
    ) -> str:
        frameworks = facts.get("frameworks", {})
        if not frameworks:
            return "No experiment results are available yet. Run the benchmark first."

        q = question.lower()

        if (
            category == "counterfactual_or_comparison"
            or "compare" in q
            or " versus " in q
            or " vs " in q
        ):
            return self._comparison_answer(question, facts)

        if "fair" in q or "parity" in q or "dispar" in q:
            return self._fairness_answer(question, facts)

        if "drift" in q or "recovery" in q or "temporal" in q:
            return self._drift_answer(question, facts)

        if any(token in q for token in ["xai", "explainability", "fidelity", "stability"]):
            return self._xai_answer(question, facts)

        if category == "clarification" or "what is" in q or "define" in q:
            if "utility" in q:
                return (
                    "Utility is the normalized weighted multi-objective score used to rank "
                    "frameworks. It is not predictive accuracy and it depends on the current "
                    "objective weights [ranking]."
                )
            if "near-pareto" in q or "pareto" in q:
                return (
                    "Near-Pareto means epsilon-nondominated under the canonical journal "
                    "definition; the current ranking stores this status in [ranking]."
                )

        if category == "evidence_request" or any(
            token in q for token in ["source", "evidence", "show me", "prove", "data"]
        ):
            return self._evidence_answer(question, facts)

        if category == "challenge" or any(
            token in q for token in ["wrong", "disagree", "really", "challenge", "sure"]
        ):
            return self._recommendation_answer(facts, challenge=True)

        if category == "explanation_probe" or "why" in q or "recommend" in q:
            return self._recommendation_answer(facts, challenge=False)

        if "runtime" in q or "fast" in q:
            return self._metric_best_answer("runtime_sec", facts)
        if "energy" in q:
            return self._metric_best_answer("energy_kwh", facts)
        if "carbon" in q or "co2" in q:
            return self._metric_best_answer("co2_kg", facts)
        if "f1" in q:
            return self._metric_best_answer("f1_macro", facts)
        return self._metric_best_answer("accuracy", facts)
