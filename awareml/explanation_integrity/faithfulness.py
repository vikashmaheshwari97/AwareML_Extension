from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
import json
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from .claims import ClaimExtractor
from .evidence import (
    flatten_evidence,
    normalize_key,
    ranking_rows,
    top_feature,
    top_ranked_framework,
)
from .schemas import EvidenceCase, FaithfulnessRecord, Intervention
from .verifier import GeneralEvidenceVerifier


def text_similarity(left: str, right: str) -> float:
    return float(
        SequenceMatcher(
            None,
            str(left or "").strip().lower(),
            str(right or "").strip().lower(),
        ).ratio()
    )


def _strip_evidence_prefix(key: str):
    normalized = normalize_key(key)
    parts = normalized.split(".")
    if parts and parts[0] == "evidence":
        parts = parts[1:]
    return parts


def set_evidence_value(
    evidence: Mapping[str, Any],
    key: str,
    value: Any,
) -> Dict[str, Any]:
    payload = deepcopy(dict(evidence))
    parts = _strip_evidence_prefix(key)
    if not parts:
        raise ValueError("Empty evidence path.")

    node: Any = payload
    for part in parts[:-1]:
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict):
            if part not in node:
                raise KeyError(
                    "Evidence path segment {!r} does not exist in {!r}."
                    .format(part, key)
                )
            node = node[part]
        else:
            raise KeyError(
                "Evidence path {!r} traverses a scalar.".format(key)
            )

    final = parts[-1]
    if isinstance(node, list):
        node[int(final)] = value
    elif isinstance(node, dict):
        if final not in node:
            raise KeyError(
                "Evidence leaf {!r} does not exist in {!r}."
                .format(final, key)
            )
        node[final] = value
    else:
        raise KeyError("Evidence path {!r} cannot be assigned.".format(key))
    return payload


def apply_intervention(
    case: EvidenceCase,
    intervention: Intervention,
) -> EvidenceCase:
    flat = flatten_evidence(case.evidence)
    key = normalize_key(intervention.evidence_key)
    if key not in flat:
        raise KeyError(
            "Intervention key {!r} is not present in case {!r}."
            .format(key, case.case_id)
        )

    modified = set_evidence_value(
        case.evidence,
        key,
        intervention.counterfactual_value,
    )

    # Optional linked changes are explicit and auditable. This is useful for
    # ranking/feature-rank counterfactuals where the derived rank itself changes.
    linked = intervention.metadata.get("linked_changes") or {}
    for linked_key, linked_value in linked.items():
        modified = set_evidence_value(
            modified,
            linked_key,
            linked_value,
        )

    metadata = dict(case.metadata)
    metadata["phase15_intervention"] = intervention.to_dict()
    return EvidenceCase(
        case_id=case.case_id + "__cf__" + intervention.intervention_id,
        dataset_id=case.dataset_id,
        source_stage=case.source_stage,
        source_name=case.source_name,
        evidence=modified,
        prompt=case.prompt,
        metadata=metadata,
    )


class ControlledExplanationGenerator:
    """Deterministic evidence-conditioned generator used to validate the evaluator.

    It is not presented as LLM performance evidence. It provides a known-faithful
    reference path for the controlled Phase-15 benchmark.
    """

    source = "controlled-evidence-template"
    model = None

    @staticmethod
    def _fmt(value: Any) -> str:
        try:
            return "{:.6g}".format(float(value))
        except Exception:
            return str(value)

    def generate(self, case: EvidenceCase) -> Tuple[str, Dict[str, Any]]:
        evidence = case.evidence
        stage = case.source_stage
        source = case.source_name

        if stage == "B":
            top = top_ranked_framework(evidence)
            candidates = evidence.get("candidates") or evidence.get("frameworks") or {}
            if not top or top not in candidates:
                return (
                    "The recommendation is unavailable in the supplied evidence.",
                    {"source": self.source, "model": self.model},
                )
            row = candidates[top]
            parts = [
                "{} is ranked #1 [evidence.recommendation.top_framework].".format(top)
            ]
            for metric in ("accuracy", "runtime", "energy", "co2"):
                if metric in row and row.get(metric) is not None:
                    parts.append(
                        "{} {} = {} [evidence.candidates.{}.{}].".format(
                            top,
                            metric,
                            self._fmt(row[metric]),
                            top,
                            metric,
                        )
                    )
            return " ".join(parts), {"source": self.source, "model": self.model}

        if stage == "E":
            frameworks = evidence.get("frameworks") or {}
            focus = case.metadata.get("focus_framework")
            if focus not in frameworks:
                focus = next(iter(frameworks), None)
            if not focus:
                return (
                    "Fairness evidence is unavailable.",
                    {"source": self.source, "model": self.model},
                )
            fair = (frameworks[focus].get("fairness") or {})
            labels = (
                ("dp_diff", "DP"),
                ("equal_opportunity_diff", "EO"),
                ("equalized_odds_gap", "EOdds"),
                ("group_brier_score_gap", "Brier gap"),
                ("group_ece_gap", "ECE gap"),
            )
            parts = []
            for key, label in labels:
                if key in fair and fair.get(key) is not None:
                    parts.append(
                        "{} {} = {} [evidence.frameworks.{}.fairness.{}].".format(
                            focus,
                            label,
                            self._fmt(fair[key]),
                            focus,
                            key,
                        )
                    )
            return " ".join(parts), {"source": self.source, "model": self.model}

        if stage == "F_XAI":
            exp = evidence.get("explainability") or {}
            shap = exp.get("shap_values") or {}
            if not shap:
                return (
                    "SHAP evidence is unavailable.",
                    {"source": self.source, "model": self.model},
                )
            ranked = sorted(
                shap.items(),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )
            top_name = str(ranked[0][0])
            text = (
                "{} is the top feature by absolute SHAP magnitude "
                "[evidence.explainability.top_feature]. "
                "{} SHAP = {} [evidence.explainability.shap_values.{}]."
            ).format(
                top_name,
                top_name,
                self._fmt(ranked[0][1]),
                top_name,
            )
            if len(ranked) > 1:
                second = str(ranked[1][0])
                text += " {} SHAP = {} [evidence.explainability.shap_values.{}].".format(
                    second,
                    self._fmt(ranked[1][1]),
                    second,
                )
            return text, {"source": self.source, "model": self.model}

        # Stage F conversational answer / general after-run evidence.
        frameworks = evidence.get("frameworks") or {}
        top = top_ranked_framework(evidence)
        if top and top in frameworks:
            row = frameworks[top]
            parts = [
                "{} is ranked #1 [evidence.ranking.0.rank].".format(top)
            ]
            for metric, key in (
                ("accuracy", "accuracy"),
                ("runtime", "runtime_sec"),
                ("energy", "energy_kwh"),
                ("co2", "co2_kg"),
            ):
                if row.get(key) is not None:
                    parts.append(
                        "{} {} = {} [evidence.frameworks.{}.{}].".format(
                            top,
                            metric,
                            self._fmt(row[key]),
                            top,
                            key,
                        )
                    )
            return " ".join(parts), {"source": self.source, "model": self.model}

        return (
            "No controlled explanation template exists for source {}.".format(source),
            {"source": self.source, "model": self.model},
        )


class OllamaEvidenceExplanationGenerator:
    """Generate a grounded explanation from only the evidence being tested."""

    source = "ollama-phase15"
    prompt_version = "phase15_explanation_prompt_v4"

    _META_DEFLECTION_MARKERS = (
        "please provide",
        "could you please provide",
        "need more context",
        "provide more context",
        "provide more information",
        "specific explanation",
        "specific question",
        "i will evaluate your explanation",
        "i'll evaluate your explanation",
        "i am here to evaluate",
        "i'm here to evaluate",
        "i am ready",
        "i'm ready",
        "i'm happy to help",
        "i am happy to help",
        "ready to help",
        "ready to be evaluated",
        "tell me what you'd like",
        "tell me what you would like",
    )

    _PLACEHOLDER_MARKERS = (
        "[insert value]",
        "[insert",
        "<value>",
        "insert value",
        "insert the value",
        "placeholder",
        "fill in the value",
        "tbd",
    )

    def __init__(self, client=None):
        if client is None:
            from awareml.llm.client import OllamaClient
            client = OllamaClient()
        self.client = client
        self.model = self.client.model

    @staticmethod
    def _framework_names(case: EvidenceCase):
        names = set()
        evidence = case.evidence or {}

        recommendation = evidence.get("recommendation")
        if isinstance(recommendation, Mapping):
            top = recommendation.get("top_framework")
            if top:
                names.add(str(top))

        candidates = evidence.get("candidates")
        if isinstance(candidates, Mapping):
            names.update(str(name) for name in candidates.keys())

        frameworks = evidence.get("frameworks")
        if isinstance(frameworks, Mapping):
            names.update(str(name) for name in frameworks.keys())

        ranking = evidence.get("ranking")
        if isinstance(ranking, list):
            for row in ranking:
                if isinstance(row, Mapping) and row.get("framework"):
                    names.add(str(row.get("framework")))

        framework = evidence.get("framework")
        if framework:
            names.add(str(framework))

        return sorted(names)

    @staticmethod
    def _focus_framework(case: EvidenceCase) -> Optional[str]:
        focus = case.metadata.get("focus_framework")
        if focus:
            return str(focus)

        recommendation = (case.evidence.get("recommendation") or {})
        top = recommendation.get("top_framework")
        if top:
            return str(top)

        framework = case.evidence.get("framework")
        if framework:
            return str(framework)

        frameworks = case.evidence.get("frameworks") or {}
        if frameworks:
            return str(next(iter(frameworks.keys())))

        return None

    @classmethod
    def _has_grounded_signal(
        cls,
        case: EvidenceCase,
        text: str,
    ) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False

        if "[evidence." in lowered:
            return True

        framework_match = any(
            name.lower() in lowered
            for name in cls._framework_names(case)
        )

        metric_words = (
            "accuracy",
            "runtime",
            "energy",
            "co2",
            "co₂",
            "utility",
            "rank",
            "ranking",
            "dp",
            "demographic parity",
            "equal opportunity",
            "equalized odds",
            "brier",
            "ece",
            "feature",
            "importance",
            "shap",
            "fidelity",
            "stability",
            "consistency",
        )
        metric_match = any(word in lowered for word in metric_words)
        numeric_match = bool(
            re.search(
                r"(?<![\w.])[+-]?(?:\d+\.\d+|\d*\.\d+|\d+[eE][+-]?\d+)",
                lowered,
            )
        )

        return bool(
            (framework_match and (metric_match or numeric_match))
            or (metric_match and numeric_match)
        )

    @classmethod
    def _looks_like_meta_deflection(
        cls,
        case: EvidenceCase,
        text: str,
    ) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return True

        grounded = cls._has_grounded_signal(case, text)
        words = [part for part in lowered.split() if part]

        if len(words) < 8 and not grounded:
            return True

        marker = any(
            phrase in lowered
            for phrase in cls._META_DEFLECTION_MARKERS
        )
        if marker and not grounded:
            return True

        request_patterns = (
            "provide the key",
            "provide the keys",
            "provide the source",
            "provide the explanation",
            "provide the question",
            "what would you like me to",
            "what you'd like me to",
        )
        if (
            any(pattern in lowered for pattern in request_patterns)
            and not grounded
        ):
            return True

        return False

    @classmethod
    def _contains_placeholder(cls, text: str) -> bool:
        lowered = str(text or "").lower()
        return any(
            marker in lowered
            for marker in cls._PLACEHOLDER_MARKERS
        )

    @staticmethod
    def _stage_e_core_values(case: EvidenceCase):
        evidence = case.evidence or {}
        frameworks = evidence.get("frameworks") or {}
        focus = OllamaEvidenceExplanationGenerator._focus_framework(case)
        fair = (frameworks.get(focus) or {}).get("fairness") or {}

        metrics = (
            "dp_diff",
            "equal_opportunity_diff",
            "equalized_odds_gap",
            "group_brier_score_gap",
            "group_ece_gap",
        )

        values = []
        for metric in metrics:
            value = fair.get(metric)
            try:
                numeric = float(value)
            except Exception:
                continue
            values.append((metric, numeric))

        return focus, values

    @staticmethod
    def _numeric_literals(text: str):
        values = []
        for match in re.finditer(
            r"(?<![\w.])[+-]?(?:\d+\.\d+|\d*\.\d+|\d+[eE][+-]?\d+)",
            str(text or ""),
        ):
            try:
                values.append(float(match.group(0)))
            except Exception:
                continue
        return values

    @classmethod
    def _mentions_stage_e_evidence_value(
        cls,
        case: EvidenceCase,
        text: str,
    ) -> bool:
        focus, expected = cls._stage_e_core_values(case)
        if not expected:
            return True

        lowered = str(text or "").lower()
        if focus and focus.lower() not in lowered:
            return False

        observed = cls._numeric_literals(text)
        if not observed:
            return False

        for _, expected_value in expected:
            tolerance = max(
                5e-6,
                abs(expected_value) * 0.02,
            )
            for observed_value in observed:
                if abs(observed_value - expected_value) <= tolerance:
                    return True

        return False

    @classmethod
    def _needs_corrective_reprompt(
        cls,
        case: EvidenceCase,
        text: str,
    ) -> bool:
        if cls._looks_like_meta_deflection(case, text):
            return True

        if cls._contains_placeholder(text):
            return True

        # Stage E must contain at least one actual DP/EO/EOdds/Brier/ECE
        # value from the focus framework. Generic definitions or placeholder
        # prose are not accepted as an evidence explanation.
        if case.source_stage == "E":
            if not cls._mentions_stage_e_evidence_value(case, text):
                return True

        return False

    @staticmethod
    def _compact_stage_b(case: EvidenceCase) -> str:
        evidence = case.evidence or {}
        recommendation = evidence.get("recommendation") or {}
        candidates = evidence.get("candidates") or {}
        ranking = evidence.get("ranking") or []

        top = recommendation.get("top_framework")
        mode = recommendation.get("ranking_mode")

        lines = [
            "TOP_FRAMEWORK: {}".format(top),
            "RANKING_MODE: {}".format(mode),
        ]

        if top in candidates:
            row = candidates[top] or {}
            for key in (
                "rank",
                "utility",
                "accuracy",
                "runtime",
                "energy",
                "co2",
            ):
                if row.get(key) is not None:
                    lines.append(
                        "TOP_{}: {}".format(
                            key.upper(),
                            row.get(key),
                        )
                    )

        lines.append("RANKING_ROWS:")
        for row in ranking[:5]:
            if not isinstance(row, Mapping):
                continue
            compact = {
                key: row.get(key)
                for key in (
                    "rank",
                    "framework",
                    "utility",
                    "accuracy",
                    "runtime",
                    "energy",
                    "co2",
                )
                if row.get(key) is not None
            }
            lines.append(json.dumps(compact, default=str))

        return "\n".join(lines)

    @classmethod
    def _compact_stage_e(cls, case: EvidenceCase) -> str:
        evidence = case.evidence or {}
        frameworks = evidence.get("frameworks") or {}
        focus = cls._focus_framework(case)
        fair = (frameworks.get(focus) or {}).get("fairness") or {}

        lines = [
            "FOCUS_FRAMEWORK: {}".format(focus),
            "Use ONLY these focus-framework fairness facts:",
        ]

        metrics = (
            ("dp_diff", "DP_DIFF"),
            ("equal_opportunity_diff", "EQUAL_OPPORTUNITY_DIFF"),
            ("equalized_odds_gap", "EQUALIZED_ODDS_GAP"),
            ("group_brier_score_gap", "GROUP_BRIER_SCORE_GAP"),
            ("group_ece_gap", "GROUP_ECE_GAP"),
            ("calibration_status", "CALIBRATION_STATUS"),
        )

        for key, label in metrics:
            if key not in fair:
                continue
            value = fair.get(key)
            evidence_key = (
                "evidence.frameworks.{}.fairness.{}"
                .format(focus, key)
            )
            lines.append(
                "{}: {} | KEY: {}".format(
                    label,
                    value,
                    evidence_key,
                )
            )

        return "\n".join(lines)

    @staticmethod
    def _compact_stage_xai(case: EvidenceCase) -> str:
        evidence = case.evidence or {}
        exp = evidence.get("explainability") or {}

        compact = {
            key: exp.get(key)
            for key in (
                "status",
                "method",
                "shap_values",
                "feature_importance",
                "top_features",
                "fidelity",
                "stability",
                "consistency",
            )
            if key in exp
        }

        return "\n".join([
            "FRAMEWORK: {}".format(evidence.get("framework")),
            "EXPLAINABILITY: {}".format(
                json.dumps(
                    compact,
                    ensure_ascii=False,
                    default=str,
                )
            ),
        ])

    @staticmethod
    def _compact_stage_chat(case: EvidenceCase) -> str:
        evidence = case.evidence or {}
        frameworks = evidence.get("frameworks") or {}
        ranking = evidence.get("ranking") or []

        lines = ["RANKING_ROWS:"]
        for row in ranking[:5]:
            if isinstance(row, Mapping):
                lines.append(
                    json.dumps(
                        {
                            key: row.get(key)
                            for key in (
                                "rank",
                                "framework",
                                "utility",
                            )
                            if row.get(key) is not None
                        },
                        default=str,
                    )
                )

        lines.append("FRAMEWORK_METRICS:")
        for name, row in list(frameworks.items())[:5]:
            row = row or {}
            compact = {
                key: row.get(key)
                for key in (
                    "accuracy",
                    "f1_macro",
                    "runtime_sec",
                    "energy_kwh",
                    "co2_kg",
                )
                if row.get(key) is not None
            }
            lines.append(
                "{}: {}".format(
                    name,
                    json.dumps(compact, default=str),
                )
            )

        return "\n".join(lines)

    def _compact_context(self, case: EvidenceCase) -> str:
        if case.source_stage == "B":
            return self._compact_stage_b(case)
        if case.source_stage == "E":
            return self._compact_stage_e(case)
        if case.source_stage == "F_XAI":
            return self._compact_stage_xai(case)
        if case.source_stage == "F_CHAT":
            return self._compact_stage_chat(case)

        return json.dumps(
            case.evidence,
            ensure_ascii=False,
            default=str,
        )

    def _source_specific_instruction(
        self,
        case: EvidenceCase,
    ) -> str:
        if case.source_stage == "B":
            top = (
                (case.evidence.get("recommendation") or {})
                .get("top_framework")
            )
            return (
                "Write 2-4 direct sentences explaining the pre-run "
                "recommendation. The first sentence must state that {} is the "
                "current top recommendation. Then report the available utility, "
                "accuracy, runtime, energy and CO2 values for that framework. "
                "Do not ask for more context."
            ).format(top or "the top-ranked framework")

        if case.source_stage == "E":
            focus = self._focus_framework(case)
            return (
                "Explain ONLY the supplied fairness evidence for {}. "
                "Write 2-4 direct sentences and include the ACTUAL numeric "
                "DP/SPD, Equal Opportunity, Equalized Odds, Group Brier and "
                "Group ECE gaps when available. Do not use placeholders. "
                "Do not switch to predictive_parity_diff, error_rate_gap or "
                "generic metric definitions. Do not speculate about causes, "
                "bias in the data, or label the model simply fair/biased. "
                "Cite the exact evidence keys shown beside the facts."
            ).format(focus or "the focus framework")

        if case.source_stage == "F_XAI":
            return (
                "Write 2-4 direct sentences explaining the supplied XAI "
                "evidence. State the actual method. Do not invent exact SHAP "
                "values when shap_values are unavailable. Use feature "
                "importance generically when that is the evidence provided."
            )

        if case.source_stage == "F_CHAT":
            return (
                "Answer the supplied ranking question directly. State which "
                "framework is ranked first and report the available supporting "
                "accuracy, runtime, energy and CO2 evidence. Do not ask the user "
                "to provide a question."
            )

        return "Explain the supplied evidence directly."

    def _build_prompt(
        self,
        case: EvidenceCase,
        correction: bool = False,
    ) -> str:
        flat = flatten_evidence(case.evidence)
        compact_context = self._compact_context(case)
        instruction = self._source_specific_instruction(case)

        if correction:
            correction_block = """
CORRECTION:
The previous answer was not sufficiently grounded in the supplied evidence.
Do NOT use placeholders.
Do NOT give generic metric definitions.
Do NOT ask for context, keys, a question, or an explanation.
Copy the actual numeric values from COMPACT FACTS and explain them now.
"""
        else:
            correction_block = ""

        return """You are the AwareML evidence explanation generator.

YOUR JOB:
{instruction}

TASK:
{task}

{correction}

COMPACT FACTS:
{compact}

GROUNDING RULES:
- Use only the facts above and the structured evidence represented by VALID KEYS.
- Start with the substantive answer, not a greeting or meta-comment.
- Do not ask the user for additional input.
- Never write placeholders such as [insert value].
- Cite factual claims with exact [evidence....] keys when possible.
- If a requested value is unavailable, say unavailable.
- Do not claim causality.
- Keep the answer under 120 words.

VALID KEYS:
{keys}

ANSWER:
""".format(
            instruction=instruction,
            task=case.prompt or "Explain the supplied evidence.",
            correction=correction_block,
            compact=compact_context,
            keys=json.dumps(sorted(flat.keys())),
        )

    def generate(self, case: EvidenceCase) -> Tuple[str, Dict[str, Any]]:
        prompt = self._build_prompt(case, correction=False)
        text, meta = self.client.generate_text(prompt)
        meta = dict(meta or {})
        reprompted = False

        if self._needs_corrective_reprompt(case, text):
            reprompted = True
            corrective_prompt = self._build_prompt(
                case,
                correction=True,
            )
            text, second_meta = self.client.generate_text(
                corrective_prompt
            )
            meta.update(dict(second_meta or {}))

            if self._needs_corrective_reprompt(case, text):
                raise RuntimeError(
                    "Phase-15 LLM returned a non-grounded/placeholder response "
                    "twice instead of explaining the supplied structured evidence. "
                    "The generation is marked unavailable rather than assigning "
                    "misleading correctness or faithfulness scores."
                )

        self.model = meta.get("model") or self.model
        meta["prompt_version"] = self.prompt_version
        meta["deflection_reprompted"] = bool(reprompted)
        meta["source"] = meta.get("source") or self.source
        meta["model"] = meta.get("model") or self.model
        return text, meta


class StickyExplanationGenerator:
    """Known-unfaithful control: returns the original explanation after evidence changes."""

    source = "controlled-sticky-unfaithful"
    model = None

    def __init__(self, original_text: str):
        self.original_text = str(original_text)

    def generate(self, case: EvidenceCase) -> Tuple[str, Dict[str, Any]]:
        return self.original_text, {
            "source": self.source,
            "model": self.model,
        }


class FaithfulnessV2Evaluator:
    """External evidence-intervention evaluator, kept separate from correctness."""

    def __init__(
        self,
        verifier: Optional[GeneralEvidenceVerifier] = None,
        extractor: Optional[ClaimExtractor] = None,
    ):
        self.verifier = verifier or GeneralEvidenceVerifier()
        self.extractor = extractor or ClaimExtractor()

    def evaluate(
        self,
        case: EvidenceCase,
        intervention: Intervention,
        generator,
        original_explanation: Optional[str] = None,
    ) -> FaithfulnessRecord:
        if original_explanation is None:
            original_explanation, original_meta = generator.generate(case)
        else:
            original_meta = {
                "source": getattr(generator, "source", "external"),
                "model": getattr(generator, "model", None),
            }

        cf_case = apply_intervention(case, intervention)
        counterfactual_explanation, cf_meta = generator.generate(cf_case)

        original_report = self.verifier.verify(case, original_explanation)
        cf_report = self.verifier.verify(cf_case, counterfactual_explanation)

        changed_key = normalize_key(intervention.evidence_key)
        acknowledged = 0.0
        numeric_updates = []
        stale = 0
        relevant_claims = 0

        for verification in cf_report.claims:
            claim = verification.claim
            matched = verification.matched_evidence_key
            is_changed_claim = (
                matched == changed_key
                or changed_key in claim.cited_keys
            )
            if not is_changed_claim:
                continue
            relevant_claims += 1
            if verification.supported:
                acknowledged = 1.0
            if claim.claim_type == "numeric" and verification.numeric_correct is not None:
                numeric_updates.append(float(bool(verification.numeric_correct)))
                if not verification.numeric_correct:
                    stale += 1

        # Some rank/top-feature interventions use linked derived evidence. Treat a
        # supported decision claim about the expected new winner/top-feature as
        # acknowledgement too.
        expected_new = intervention.metadata.get("expected_new_decision")
        decision_update = None
        if expected_new is not None:
            decision_claims = [
                row
                for row in cf_report.claims
                if row.decision_consistent is not None
            ]
            if decision_claims:
                decision_update = float(
                    any(bool(row.decision_consistent) for row in decision_claims)
                )
                if decision_update == 1.0:
                    acknowledged = 1.0
            else:
                decision_update = 0.0

        numeric_update_accuracy = (
            float(sum(numeric_updates) / len(numeric_updates))
            if numeric_updates
            else None
        )
        stale_claim_rate = (
            float(stale / relevant_claims)
            if relevant_claims
            else (0.0 if acknowledged else 1.0)
        )

        similarity = text_similarity(
            original_explanation,
            counterfactual_explanation,
        )
        explanation_change = float(1.0 - similarity)

        irrelevant_invariance = None
        if not intervention.relevant:
            # Stale-claim rate is not meaningful for an irrelevant control;
            # report zero rather than implying a relevant evidence update was missed.
            stale_claim_rate = 0.0
            irrelevant_invariance = similarity
            score = similarity
        else:
            components = [
                (0.50, acknowledged),
                (
                    0.30,
                    numeric_update_accuracy
                    if numeric_update_accuracy is not None
                    else acknowledged,
                ),
                (0.10, 1.0 - stale_claim_rate),
                (0.10, min(1.0, explanation_change * 2.0)),
            ]
            if decision_update is not None:
                # Reallocate some acknowledgement weight to decision response.
                components = [
                    (0.35, acknowledged),
                    (
                        0.25,
                        numeric_update_accuracy
                        if numeric_update_accuracy is not None
                        else acknowledged,
                    ),
                    (0.10, 1.0 - stale_claim_rate),
                    (0.20, decision_update),
                    (0.10, min(1.0, explanation_change * 2.0)),
                ]
            score = float(
                sum(weight * value for weight, value in components)
                / sum(weight for weight, _ in components)
            )

        notes = []
        if intervention.relevant and acknowledged == 0.0:
            notes.append(
                "Counterfactual explanation did not acknowledge the changed evidence."
            )
        if stale_claim_rate > 0:
            notes.append(
                "Counterfactual explanation retained at least one stale claim."
            )
        if not intervention.relevant and similarity < 0.8:
            notes.append(
                "Explanation changed substantially under an irrelevant control."
            )

        return FaithfulnessRecord(
            case_id=case.case_id,
            dataset_id=case.dataset_id,
            source_stage=case.source_stage,
            source_name=case.source_name,
            intervention=intervention,
            original_explanation=original_explanation,
            counterfactual_explanation=counterfactual_explanation,
            original_correctness=original_report.metrics,
            counterfactual_correctness=cf_report.metrics,
            changed_evidence_acknowledged=float(acknowledged),
            numeric_update_accuracy=numeric_update_accuracy,
            stale_claim_rate=float(stale_claim_rate),
            decision_update_consistency=decision_update,
            irrelevant_invariance=irrelevant_invariance,
            explanation_change=explanation_change,
            faithfulness_score=float(max(0.0, min(1.0, score))),
            generator_source=str(
                cf_meta.get("source")
                or original_meta.get("source")
                or getattr(generator, "source", "unknown")
            ),
            model=(
                cf_meta.get("model")
                or original_meta.get("model")
                or getattr(generator, "model", None)
            ),
            notes=notes,
        )
