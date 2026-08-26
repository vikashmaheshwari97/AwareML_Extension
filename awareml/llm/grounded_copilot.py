from __future__ import annotations

import json
import re
from typing import Optional

from .client import OllamaClient
from .evidence import EvidenceBundle
from .schemas import GroundedAnswer


EVIDENCE_PATTERN = re.compile(
    r"\[(evidence\.[A-Za-z0-9_.-]+)\]"
)


class GroundedCopilotChat:
    """Evidence-only before/during/after assistant."""

    def __init__(
        self,
        client: Optional[
            OllamaClient
        ] = None,
    ):
        self.client = (
            client or OllamaClient()
        )

    def answer(
        self,
        question: str,
        evidence: EvidenceBundle,
        use_llm: bool = False,
    ) -> GroundedAnswer:
        if not question.strip():
            return GroundedAnswer(
                text=(
                    "Ask a question about the "
                    "measured AwareML evidence."
                ),
                source="deterministic",
                evidence_keys=[],
            )

        if use_llm:
            try:
                answer = (
                    self._ollama_answer(
                        question,
                        evidence,
                    )
                )
                if answer is not None:
                    return answer
            except Exception:
                pass

        text, keys = (
            self._deterministic_answer(
                question,
                evidence,
            )
        )
        return GroundedAnswer(
            text=text,
            source="deterministic",
            evidence_keys=keys,
        )

    def _ollama_answer(
        self,
        question: str,
        evidence: EvidenceBundle,
    ):
        payload = (
            evidence.prompt_payload()
        )
        prompt = """You are the AwareML evidence-grounded Copilot.

Answer the QUESTION using ONLY the EVIDENCE JSON.

Rules:
- Never invent experimental values.
- Every numerical or comparative claim must cite at least one exact valid key
  using [evidence....].
- Never claim correlation proves causality.
- If evidence is insufficient, explicitly say so.
- Do not infer raw dataset rows or participant information.
- Distinguish predicted pre-run evidence from observed post-run evidence.
- Keep the answer concise and scientific.

EVIDENCE:
{}

QUESTION:
{}
""".format(
            json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            ),
            question.strip(),
        )

        text, meta = (
            self.client.generate_text(
                prompt
            )
        )

        cited = (
            EVIDENCE_PATTERN.findall(
                text
            )
        )
        valid = set(
            evidence.valid_keys
        )
        invalid = [
            key
            for key in cited
            if key not in valid
        ]

        if invalid:
            return None

        # When evidence exists, an uncited LLM answer is not accepted as a
        # grounded scientific answer.
        if valid and not cited:
            return None

        return GroundedAnswer(
            text=text,
            source="ollama",
            model=meta.get("model"),
            evidence_keys=sorted(
                set(cited)
            ),
        )

    def _deterministic_answer(
        self,
        question: str,
        evidence: EvidenceBundle,
    ):
        q = question.lower()
        facts = evidence.facts

        if evidence.stage == "before":
            return self._before(
                q,
                facts,
            )
        if evidence.stage == "during":
            return self._during(
                q,
                facts,
            )
        return self._after(
            q,
            facts,
        )

    def _before(self, q, facts):
        recommendation = facts.get(
            "recommendation",
            {},
        )
        top = recommendation.get(
            "top_framework"
        )
        candidates = facts.get(
            "candidates",
            {},
        )
        row = candidates.get(
            top,
            {},
        )

        if not top:
            return (
                "No empirical pre-run recommendation is available.",
                [],
            )

        key = (
            "evidence.before.recommendation.top_framework"
        )

        if (
            "energy" in q
            or "carbon" in q
            or "co2" in q
        ):
            objective = (
                "co2"
                if (
                    "co2" in q
                    or "carbon" in q
                )
                else "energy"
            )
            available = [
                (
                    fw,
                    values.get(
                        objective
                    ),
                )
                for fw, values in (
                    candidates.items()
                )
                if values.get(
                    objective
                )
                is not None
            ]
            if available:
                best = min(
                    available,
                    key=lambda item: item[1],
                )
                evidence_key = (
                    "evidence.before.candidates."
                    + best[0]
                    + "."
                    + objective
                )
                return (
                    "{} has the lowest predicted {} among the current "
                    "candidates: {:.4g} [{}]. This is a pre-run prediction, "
                    "not an observed outcome.".format(
                        best[0],
                        objective,
                        best[1],
                        evidence_key,
                    ),
                    [evidence_key],
                )

        accuracy = row.get(
            "accuracy"
        )
        utility = row.get(
            "utility"
        )
        keys = [key]
        details = []

        if accuracy is not None:
            accuracy_key = (
                "evidence.before.candidates."
                + top
                + ".accuracy"
            )
            details.append(
                "predicted accuracy {:.4g} [{}]".format(
                    accuracy,
                    accuracy_key,
                )
            )
            keys.append(
                accuracy_key
            )

        if utility is not None:
            utility_key = (
                "evidence.before.candidates."
                + top
                + ".utility"
            )
            details.append(
                "preference utility {:.4g} [{}]".format(
                    utility,
                    utility_key,
                )
            )
            keys.append(
                utility_key
            )

        return (
            "{} is the current empirical top-ranked framework [{}]. {}"
            .format(
                top,
                key,
                "; ".join(details),
            ),
            keys,
        )

    def _during(self, q, facts):
        if "drift" in q or "recover" in q:
            count = (
                facts.get(
                    "drift",
                    {},
                ).get("count")
            )
            latest = (
                facts.get(
                    "drift",
                    {},
                ).get(
                    "latest_event"
                )
            )
            count_key = (
                "evidence.during.drift.count"
            )
            latest_key = (
                "evidence.during.drift.latest_event"
            )
            return (
                "The current run has {} recorded drift event(s) [{}]; "
                "the latest recorded event is at sample {} [{}].".format(
                    count,
                    count_key,
                    latest,
                    latest_key,
                ),
                [
                    count_key,
                    latest_key,
                ],
            )

        if (
            "fair" in q
            or "bias" in q
        ):
            fairness = facts.get(
                "fairness",
                {},
            )
            if not fairness:
                return (
                    "No fairness evidence is available for the current run.",
                    [],
                )
            status = fairness.get(
                "status"
            )
            key = (
                "evidence.during.fairness.status"
            )
            return (
                "The current fairness evidence status is {} [{}]. "
                "Use the recorded gap fields only when support is sufficient."
                .format(
                    status,
                    key,
                ),
                [key],
            )

        if (
            "feature" in q
            or "explain" in q
        ):
            top = (
                facts.get(
                    "explainability",
                    {},
                ).get(
                    "top_features"
                )
                or []
            )
            key = (
                "evidence.during.explainability.top_features"
            )
            if top:
                return (
                    "The current explanation snapshot reports these top "
                    "features: {} [{}].".format(
                        top[:5],
                        key,
                    ),
                    [key],
                )
            return (
                "No supported feature-importance snapshot is available.",
                [],
            )

        accuracy = (
            facts.get(
                "current",
                {},
            ).get("accuracy")
        )
        key = (
            "evidence.during.current.accuracy"
        )
        return (
            "The current observed accuracy is {} [{}].".format(
                accuracy,
                key,
            ),
            [key],
        )

    def _after(self, q, facts):
        frameworks = facts.get(
            "frameworks",
            {},
        )
        if not frameworks:
            return (
                "No post-run framework evidence is available.",
                [],
            )

        metric = "accuracy"
        minimize = False

        if (
            "runtime" in q
            or "fast" in q
        ):
            metric = "runtime_sec"
            minimize = True
        elif "energy" in q:
            metric = "energy_kwh"
            minimize = True
        elif (
            "co2" in q
            or "carbon" in q
        ):
            metric = "co2_kg"
            minimize = True
        elif "f1" in q:
            metric = "f1_macro"

        available = [
            (
                fw,
                row.get(metric),
            )
            for fw, row in (
                frameworks.items()
            )
            if row.get(metric)
            is not None
        ]

        if not available:
            return (
                "The requested metric is not measured in the post-run evidence.",
                [],
            )

        best = (
            min(
                available,
                key=lambda item: item[1],
            )
            if minimize
            else max(
                available,
                key=lambda item: item[1],
            )
        )
        key = (
            "evidence.after.frameworks."
            + best[0]
            + "."
            + metric
        )
        return (
            "{} has the strongest observed {} value: {:.4g} [{}]. "
            "This comparison refers to the current completed experiment only."
            .format(
                best[0],
                metric,
                best[1],
                key,
            ),
            [key],
        )
