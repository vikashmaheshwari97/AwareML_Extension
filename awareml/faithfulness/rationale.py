from __future__ import annotations

import json
import re
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd

from awareml.llm.client import OllamaClient
from awareml.llm.evidence import (
    EvidenceBundle,
)
from awareml.recommender.v2_ranking import (
    OBJECTIVES,
)

from .attribution import (
    cited_objectives,
)
from .metrics import (
    citation_validity,
    extract_evidence_keys,
)
from .schemas import (
    RationaleRecord,
)


EVIDENCE_PATTERN = re.compile(
    r"\[(evidence\.[A-Za-z0-9_.-]+)\]"
)


def _objective_direction(
    objective: str,
) -> str:
    return (
        "higher"
        if objective == "accuracy"
        else "lower"
    )


class FaithfulRationaleGenerator:
    """Generate a rationale constrained by recommendation evidence influence."""

    def __init__(
        self,
        client: Optional[
            OllamaClient
        ] = None,
    ):
        self.client = (
            client or OllamaClient()
        )

    def generate(
        self,
        evidence: EvidenceBundle,
        ranked: pd.DataFrame,
        influence: Mapping[str, float],
        use_llm: bool = False,
    ) -> RationaleRecord:
        if ranked is None or ranked.empty:
            raise ValueError(
                "A ranking is required."
            )

        if use_llm:
            try:
                llm = self._generate_llm(
                    evidence,
                    ranked,
                    influence,
                )
                if llm is not None:
                    return llm
            except Exception:
                pass

        return self._generate_deterministic(
            evidence,
            ranked,
            influence,
        )

    def _top_objectives(
        self,
        influence,
    ):
        return sorted(
            OBJECTIVES,
            key=lambda objective: float(
                influence.get(
                    objective,
                    0.0,
                )
            ),
            reverse=True,
        )

    def _generate_deterministic(
        self,
        evidence,
        ranked,
        influence,
    ):
        top = str(
            ranked.iloc[0]["framework"]
        )
        runner = (
            str(
                ranked.iloc[1][
                    "framework"
                ]
            )
            if len(ranked) > 1
            else None
        )

        ranked_objectives = (
            self._top_objectives(
                influence
            )
        )
        chosen = ranked_objectives[:2]

        recommendation_key = (
            "evidence.before."
            "recommendation.top_framework"
        )
        keys = [
            recommendation_key
        ]
        phrases = []

        for objective in chosen:
            top_key = (
                "evidence.before."
                "candidates.{}."
                "{}".format(
                    top,
                    objective,
                )
            )
            top_value = float(
                ranked.iloc[0][
                    objective
                ]
            )
            keys.append(top_key)

            if runner is not None:
                runner_row = ranked[
                    ranked["framework"].eq(
                        runner
                    )
                ].iloc[0]
                runner_value = float(
                    runner_row[
                        objective
                    ]
                )
                runner_key = (
                    "evidence.before."
                    "candidates.{}."
                    "{}".format(
                        runner,
                        objective,
                    )
                )
                keys.append(
                    runner_key
                )

                phrases.append(
                    "{} {} evidence: {}={:.4g} [{}] "
                    "vs {}={:.4g} [{}]".format(
                        _objective_direction(
                            objective
                        ),
                        objective,
                        top,
                        top_value,
                        top_key,
                        runner,
                        runner_value,
                        runner_key,
                    )
                )
            else:
                phrases.append(
                    "{}={:.4g} [{}]".format(
                        objective,
                        top_value,
                        top_key,
                    )
                )

        text = (
            "{} is the current top recommendation [{}]. "
            "The most decision-influential objective evidence is: {}. "
            "These are predicted pre-run values, not observed outcomes."
        ).format(
            top,
            recommendation_key,
            "; ".join(phrases),
        )

        valid = citation_validity(
            keys,
            evidence.valid_keys,
        )
        top_influence = (
            ranked_objectives[0]
            if ranked_objectives
            else None
        )

        return RationaleRecord(
            text=text,
            source="deterministic-faithful",
            evidence_keys=keys,
            cited_objectives=(
                cited_objectives(keys)
            ),
            citation_validity=valid,
            top_framework_mentioned=(
                top.lower()
                in text.lower()
            ),
            top_influence_objective=(
                top_influence
            ),
            top_influence_cited=(
                top_influence
                in cited_objectives(
                    keys
                )
                if top_influence
                else False
            ),
        )

    def _generate_llm(
        self,
        evidence,
        ranked,
        influence,
    ):
        top = str(
            ranked.iloc[0]["framework"]
        )
        ranked_objectives = (
            self._top_objectives(
                influence
            )
        )
        top_influence = (
            ranked_objectives[0]
            if ranked_objectives
            else None
        )

        payload = (
            evidence.prompt_payload()
        )
        payload[
            "decision_influence"
        ] = {
            objective: float(
                influence.get(
                    objective,
                    0.0,
                )
            )
            for objective in OBJECTIVES
        }

        prompt = """You are the AwareML faithfulness evaluator's rationale generator.

Explain why the current top framework is recommended.

STRICT RULES:
1. Use only the supplied EVIDENCE.
2. The current top framework is: {top}
3. Prefer the objective evidence with the greatest supplied decision influence.
4. Every numerical/comparative statement must cite exact valid evidence keys
   using [evidence....].
5. Cite the top-framework recommendation key and at least one objective key.
6. Do not invent causal claims.
7. Distinguish predicted pre-run evidence from observed outcomes.
8. Do not discuss raw dataset rows.
9. Keep the explanation under 130 words.

EVIDENCE:
{payload}
""".format(
            top=top,
            payload=json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            ),
        )

        text, meta = (
            self.client.generate_text(
                prompt
            )
        )

        cited = extract_evidence_keys(
            text
        )
        if not cited:
            return None

        valid = set(
            evidence.valid_keys
        )
        if any(
            key not in valid
            for key in cited
        ):
            return None

        recommendation_key = (
            "evidence.before."
            "recommendation.top_framework"
        )
        if recommendation_key not in cited:
            return None

        objectives = cited_objectives(
            cited
        )
        if not objectives:
            return None

        return RationaleRecord(
            text=text,
            source="ollama-faithful",
            model=meta.get("model"),
            evidence_keys=sorted(
                set(cited)
            ),
            cited_objectives=objectives,
            citation_validity=(
                citation_validity(
                    cited,
                    evidence.valid_keys,
                )
            ),
            top_framework_mentioned=(
                top.lower()
                in text.lower()
            ),
            top_influence_objective=(
                top_influence
            ),
            top_influence_cited=(
                top_influence
                in objectives
                if top_influence
                else False
            ),
        )
