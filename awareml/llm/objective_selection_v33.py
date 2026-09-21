from __future__ import annotations

"""Interactive Objective Selection V3.3 development selector.

V3.3 is deliberately NOT a replacement for the frozen Phase-12-v2 confirmatory
V3.2 selector. It combines V3.2-style strict validation of LLM-selected
objectives with V3.1-style controlled semantic recovery for interactive use.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from .interactive_runtime_v33 import InteractiveOllamaClientV33
from .journal_client import JournalLLMResponseError, JournalModelLockError
from .objective_selection_v31 import semantic_support_map as v31_semantic_support_map
from .objective_selection_v32 import (
    CANONICAL_OBJECTIVES,
    EvidenceGroundedPayloadV32,
    _find_concept_support as _v32_find_concept_support,
    _negative_requirement_present as _v32_negative_requirement_present,
    _normalize_payload as _v32_normalize_payload,
    _preflight_status as _v32_preflight_status,
    _quote_is_grounded as _v32_quote_is_grounded,
)
from .schemas import ObjectiveLabel, ObjectiveSelectionResult


SELECTOR_ID = "hybrid_evidence_grounded_objective_selector_v33"
SELECTOR_VERSION = "3.3-development"
EVALUATION_ROLE = "interactive_development_not_confirmatory"

# V3.3_REVERSE_RELIABILITY_SUPPORT_V1
#
# Development-only extension. Do not move these patterns into frozen V3.2.
# Frozen V3.2 mainly recognizes adjective-first constructions such as
# "reliable detection". Interactive V3.3 also recognizes verb-first forms such
# as "detect dangerous conditions reliably".
V33_ACCURACY_REVERSE_RELIABILITY_PATTERNS = (
    r"\b(?:detect|detects|detected|detecting)\b.{0,60}\breliabl(?:e|y)\b",
    r"\b(?:identify|identifies|identified|identifying)\b.{0,60}\breliabl(?:e|y)\b",
    r"\b(?:recognize|recognizes|recognized|recognizing|recognise|recognises|recognised|recognising)\b.{0,60}\breliabl(?:e|y)\b",
    r"\b(?:classify|classifies|classified|classifying)\b.{0,60}\breliabl(?:e|y)\b",
    r"\b(?:predict|predicts|predicted|predicting)\b.{0,60}\breliabl(?:e|y)\b",
    r"\b(?:diagnose|diagnoses|diagnosed|diagnosing)\b.{0,60}\breliabl(?:e|y)\b",
)


def _v33_find_concept_support(
    objective: str,
    text: str,
) -> Optional[Dict[str, str]]:
    # Keep every frozen V3.2 positive concept rule.
    support = _v32_find_concept_support(objective, text)
    if support is not None:
        return support

    # V3.3-only extension is intentionally narrow and Accuracy-specific.
    if objective != "Accuracy":
        return None

    normalized = re.sub(
        r"\s+",
        " ",
        str(text or "").replace("₂", "2").strip().lower(),
    )
    for idx, pattern in enumerate(V33_ACCURACY_REVERSE_RELIABILITY_PATTERNS, 1):
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return {
                "phrase": match.group(0),
                "guard_id": "accuracy_v33_reverse_reliability_{:02d}".format(idx),
                "source": "v33_development_extension",
            }
    return None



class EvidenceGroundedObjectiveSelectorV33:
    """Precision/recall-balanced interactive selector.

    LLM-selected objectives face strict V3.2 validation. Separately, strong
    scenario-local V3.1 semantic cues may recover an omitted objective or recover
    from malformed model JSON. Every recovery is exposed in the audit trail and
    sets ``fallback_used=True`` so human review remains explicit.
    """

    def __init__(
        self,
        client: Optional[InteractiveOllamaClientV33] = None,
        root: Optional[Path] = None,
        prompt_path: Optional[Path] = None,
    ):
        self.client = client or InteractiveOllamaClientV33(root=root)
        self.root = (
            Path(root).resolve()
            if root is not None
            else Path(self.client.root).resolve()
        )
        self.prompt_path = prompt_path or (
            self.root / "prompts" / "objective_selection_evidence_grounded_v33.txt"
        )
        if not self.prompt_path.exists():
            raise JournalModelLockError(
                "V3.3 evidence-grounded prompt is missing: {}".format(
                    self.prompt_path
                )
            )
        self.template = self.prompt_path.read_text(encoding="utf-8")
        if "{{USER_SCENARIO}}" not in self.template:
            raise JournalModelLockError(
                "V3.3 prompt is missing {{USER_SCENARIO}} placeholder."
            )
        self.last_audit: Dict[str, Any] = {}

    def render_prompt(self, scenario: str) -> str:
        return self.template.replace("{{USER_SCENARIO}}", str(scenario).strip())

    def _result(
        self,
        *,
        status: str,
        selected: List[ObjectiveLabel],
        uncertainties: List[str],
        source_suffix: str,
        model: Optional[str],
        fallback_used: bool,
    ) -> ObjectiveSelectionResult:
        return ObjectiveSelectionResult(
            status=status,
            selected_objectives=selected,
            uncertainties=uncertainties,
            source="{}:{}".format(SELECTOR_ID, source_suffix),
            model=model,
            fallback_used=fallback_used,
        )

    def _recovery_supports(self, scenario: str) -> Dict[str, Optional[Dict[str, str]]]:
        supports = v31_semantic_support_map(scenario)

        # V3.3 development-only supplement:
        # recover narrow concept evidence not covered by V3.1, including
        # "detect ... reliably" / "identify ... reliably".
        for objective in CANONICAL_OBJECTIVES:
            if not supports.get(objective):
                concept = _v33_find_concept_support(objective, scenario)
                if concept:
                    supports[objective] = {
                        "phrase": concept.get("phrase"),
                        "pattern_id": concept.get("guard_id"),
                        "strength": "strong",
                        "source": concept.get("source", "v32_or_v33_concept_guard"),
                    }

            # Preserve the frozen V3.2 negative-requirement guard.
            if supports.get(objective) and _v32_negative_requirement_present(
                objective, scenario
            ):
                supports[objective] = None

        return supports

    def _semantic_only_result(
        self,
        scenario: str,
        *,
        error: str,
        stage: str,
    ) -> ObjectiveSelectionResult:
        supports = self._recovery_supports(scenario)
        selected = [
            objective for objective in CANONICAL_OBJECTIVES if supports.get(objective)
        ]

        decisions: Dict[str, Any] = {}
        for objective in CANONICAL_OBJECTIVES:
            support = supports.get(objective)
            decisions[objective] = {
                "llm_selected": False,
                "confidence": "N/A",
                "evidence_returned": None,
                "evidence_grounded": False,
                "evidence_phrase_found": bool(support),
                "objective_specific_support": bool(support),
                "v32_concept_support": _v32_find_concept_support(objective, scenario),
                "v33_concept_support": _v33_find_concept_support(objective, scenario),
                "semantic_support": support.get("phrase") if support else None,
                "support_pattern_id": support.get("pattern_id") if support else None,
                "accepted": bool(support),
                "accepted_by": (
                    "v31_semantic_recovery_after_llm_failure" if support else "none"
                ),
                "reason": (
                    "strong scenario-local semantic evidence recovered the objective after unusable LLM output"
                    if support
                    else "no sufficiently specific objective evidence"
                ),
            }

        if selected:
            uncertainty = (
                "The LLaMA response could not be used at stage '{}'. V3.3 recovered "
                "only objectives with explicit scenario-local semantic evidence. "
                "Human review is required."
            ).format(stage)
            self.last_audit = {
                "selector_id": SELECTOR_ID,
                "version": SELECTOR_VERSION,
                "evaluation_role": EVALUATION_ROLE,
                "stage": stage,
                "llm_error": error,
                "final_status": "valid",
                "selected_objectives": list(selected),
                "semantic_recovery_used": True,
                "semantic_recovered_objectives": list(selected),
                "decisions": decisions,
                "uncertainties": [uncertainty],
            }
            return self._result(
                status="valid",
                selected=selected,  # type: ignore[arg-type]
                uncertainties=[uncertainty],
                source_suffix="semantic-recovery",
                model=getattr(self.client, "model", None),
                fallback_used=True,
            )

        uncertainty = (
            "No sufficiently specific objective evidence survived V3.3 validation. "
            "Please clarify the deployment priorities."
        )
        self.last_audit = {
            "selector_id": SELECTOR_ID,
            "version": SELECTOR_VERSION,
            "evaluation_role": EVALUATION_ROLE,
            "stage": stage,
            "llm_error": error,
            "final_status": "ambiguous",
            "selected_objectives": [],
            "semantic_recovery_used": False,
            "semantic_recovered_objectives": [],
            "decisions": decisions,
            "uncertainties": [uncertainty],
        }
        return self._result(
            status="ambiguous",
            selected=[],
            uncertainties=[uncertainty],
            source_suffix="abstain",
            model=getattr(self.client, "model", None),
            fallback_used=False,
        )

    def select(self, scenario: str) -> ObjectiveSelectionResult:
        preflight = _v32_preflight_status(scenario)
        if preflight is not None:
            status, uncertainties = preflight
            self.last_audit = {
                "selector_id": SELECTOR_ID,
                "version": SELECTOR_VERSION,
                "evaluation_role": EVALUATION_ROLE,
                "stage": "v32_preflight",
                "status": status,
                "final_status": status,
                "selected_objectives": [],
                "semantic_recovery_used": False,
                "semantic_recovered_objectives": [],
                "decisions": {},
                "uncertainties": list(uncertainties),
            }
            return self._result(
                status=status,
                selected=[],
                uncertainties=list(uncertainties),
                source_suffix="preflight",
                model=getattr(self.client, "model", None),
                fallback_used=False,
            )

        recovery_supports = self._recovery_supports(scenario)
        prompt = self.render_prompt(scenario)

        try:
            payload, meta = self.client.generate_json(prompt)
        except JournalModelLockError:
            raise
        except JournalLLMResponseError as exc:
            return self._semantic_only_result(
                scenario, error=str(exc), stage="llm_generation_recovery"
            )

        try:
            parsed = EvidenceGroundedPayloadV32.model_validate(
                _v32_normalize_payload(payload)
            )
        except (ValidationError, ValueError, TypeError) as exc:
            return self._semantic_only_result(
                scenario,
                error="{}: {}".format(type(exc).__name__, exc),
                stage="schema_validation_recovery",
            )

        if parsed.status in {"contradictory", "out_of_scope"}:
            self.last_audit = {
                "selector_id": SELECTOR_ID,
                "version": SELECTOR_VERSION,
                "evaluation_role": EVALUATION_ROLE,
                "stage": "llm_status",
                "raw_status": parsed.status,
                "final_status": parsed.status,
                "selected_objectives": [],
                "semantic_recovery_used": False,
                "semantic_recovered_objectives": [],
                "decisions": {},
                "uncertainties": list(parsed.uncertainties),
            }
            return self._result(
                status=parsed.status,
                selected=[],
                uncertainties=list(parsed.uncertainties),
                source_suffix="llm-status",
                model=meta.get("model"),
                fallback_used=False,
            )

        selected: List[ObjectiveLabel] = []
        recovered: List[str] = []
        uncertainties = list(parsed.uncertainties)
        decisions_audit: Dict[str, Any] = {}

        for objective in CANONICAL_OBJECTIVES:
            decision = parsed.decisions[objective]
            quote = str(decision.evidence or "").strip() or None

            grounded = bool(quote and _v32_quote_is_grounded(quote, scenario))
            quote_support = (
                _v33_find_concept_support(objective, quote or "")
                if grounded
                else None
            )
            scenario_v32_support = _v32_find_concept_support(objective, scenario)
            scenario_v33_support = _v33_find_concept_support(objective, scenario)
            explicit_negative = _v32_negative_requirement_present(objective, scenario)
            recovery_support = recovery_supports.get(objective)

            strict_llm_accept = bool(
                decision.selected
                and grounded
                and quote_support
                and scenario_v33_support
                and not explicit_negative
            )
            semantic_recovery_accept = bool(recovery_support and not explicit_negative)

            accepted = False
            accepted_by = "none"
            reason = "not selected and no recovery evidence"

            if strict_llm_accept:
                accepted = True
                accepted_by = "v33_strict_llm_evidence"
                reason = (
                    "LLM proposal passed exact-quote grounding and V3.3 objective-specific validation"
                )
            elif semantic_recovery_accept:
                accepted = True
                accepted_by = "v31_recovery_with_v32_negative_guard"
                reason = (
                    "strong scenario-local V3.1/V3.3 semantic evidence recovered the objective while retaining the V3.2 negative guard"
                )
                recovered.append(objective)
            elif decision.selected:
                reason_bits = []
                if not grounded:
                    reason_bits.append("LLM evidence was not an exact scenario-local quote")
                if not quote_support:
                    reason_bits.append("quoted evidence failed the V3.3 concept guard")
                if not scenario_v33_support:
                    reason_bits.append("scenario lacked V3.3 concept support")
                if explicit_negative:
                    reason_bits.append("scenario explicitly dismisses this requirement")
                reason = "rejected: " + "; ".join(reason_bits)
                uncertainties.append(
                    "Rejected {} LLM selection: {}.".format(
                        objective, "; ".join(reason_bits)
                    )
                )

            if accepted:
                selected.append(objective)  # type: ignore[arg-type]

            decisions_audit[objective] = {
                "llm_selected": bool(decision.selected),
                "confidence": decision.confidence,
                "evidence_returned": quote,
                "evidence_grounded": grounded,
                "evidence_phrase_found": bool(grounded or recovery_support),
                "objective_specific_support": bool(
                    scenario_v33_support or recovery_support
                ),
                "quote_concept_support": quote_support,
                "v32_concept_support": scenario_v32_support,
                "v33_concept_support": scenario_v33_support,
                "v31_recovery_support": recovery_support,
                "explicit_negative_requirement": explicit_negative,
                "semantic_support": (
                    recovery_support.get("phrase")
                    if recovery_support
                    else (
                        scenario_v33_support.get("phrase")
                        if scenario_v33_support
                        else None
                    )
                ),
                "support_pattern_id": (
                    recovery_support.get("pattern_id")
                    if recovery_support
                    else (
                        scenario_v33_support.get("guard_id")
                        if scenario_v33_support
                        else None
                    )
                ),
                "accepted": accepted,
                "accepted_by": accepted_by,
                "reason": reason,
            }

        final_status = parsed.status
        if selected:
            if final_status == "ambiguous":
                uncertainties.append(
                    "The model marked the scenario ambiguous, but V3.3 found explicit objective-specific evidence. Human review is required."
                )
            final_status = "valid"
        elif final_status == "valid":
            final_status = "ambiguous"
            uncertainties.append(
                "No objective survived V3.3 validation; clarification is required."
            )

        if recovered:
            uncertainties.append(
                "V3.3 semantic recovery added or rescued: {}. Review these objectives before accepting the proposal.".format(
                    ", ".join(recovered)
                )
            )

        fallback_used = bool(recovered)
        self.last_audit = {
            "selector_id": SELECTOR_ID,
            "version": SELECTOR_VERSION,
            "evaluation_role": EVALUATION_ROLE,
            "stage": "v33_hybrid_evidence_grounded_llm",
            "model": meta.get("model"),
            "model_digest": meta.get("model_digest"),
            "ollama_version": meta.get("ollama_version"),
            "raw_status": parsed.status,
            "final_status": final_status,
            "selected_objectives": list(selected),
            "semantic_recovery_used": fallback_used,
            "semantic_recovered_objectives": list(recovered),
            "decisions": decisions_audit,
            "uncertainties": list(uncertainties),
        }

        return self._result(
            status=final_status,
            selected=selected,
            uncertainties=uncertainties,
            source_suffix="interactive-development",
            model=meta.get("model"),
            fallback_used=fallback_used,
        )
