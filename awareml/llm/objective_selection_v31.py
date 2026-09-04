from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .journal_client import (
    JournalLLMResponseError,
    JournalModelLockError,
    StrictJournalOllamaClient,
)
from .schemas import ObjectiveLabel, ObjectiveSelectionResult


SELECTOR_ID = "hybrid_evidence_grounded_objective_selector_v31"
CANONICAL_OBJECTIVES = ("Accuracy", "Runtime", "Energy", "CO2")
_ALLOWED_STATUS = {"valid", "ambiguous", "contradictory", "out_of_scope"}
_ALLOWED_CONFIDENCE = {"high", "medium", "low"}

# V3.1 is intentionally a hybrid selector. The exact locked LLaMA remains one
# evidence source, but explicit scenario-local semantic evidence may recover an
# objective when the LLM omits it. This addresses the V3 under-selection failure
# observed during post-hoc development while preserving the strong V3 rejection
# behavior for unsupported objectives.
#
# IMPORTANT: these rules are development logic. The frozen Phase-12 v1 benchmark
# remains untouched and a fresh independent benchmark is required for a final
# journal claim about V3.1.
CUE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "Accuracy": (
        r"rarely (?:make|be) (?:the )?wrong",
        r"avoid(?:ing)? (?:incorrect|wrong) (?:decisions?|classifications?|predictions?)",
        r"trustworthy (?:predictions?|decisions?|classifications?|detections?)",
        r"dependable (?:outputs?|predictions?|decisions?|classifications?|detections?)",
        r"\bdependable\b",
        r"reliable enough",
        r"reliable .*?(?:predictions?|decisions?|classifications?|flags?|detections?|identification)",
        r"consistently (?:right|reliable)",
        r"correct decisions?",
        r"predictive quality",
        r"prediction quality",
        r"decision quality",
        r"sound decisions?",
        r"avoid costly misclassification",
        r"misclassification",
        r"strong predictive (?:quality|performance)",
        r"strong performance",
        r"high accuracy",
        r"\baccurate\b",
        r"\baccuracy\b",
        r"won['’]?t miss (?:a |an |the )?(?:severe|critical|important) issue",
        r"catch subtle (?:equipment )?glitches",
        r"reliable identification",
    ),
    "Runtime": (
        r"react quickly",
        r"respond (?:almost )?immediately",
        r"respond quickly",
        r"respond promptly",
        r"answer promptly",
        r"answer immediately",
        r"time[- ]critical",
        r"time[- ]to[- ]response",
        r"decision latency",
        r"low latency",
        r"very small latency",
        r"(?:very )?short response times?",
        r"prompt response",
        r"react(?:ing)? promptly",
        r"arrive promptly",
        r"returned promptly",
        r"return .* promptly",
        r"\bprompt\b",
        r"rapid .*?(?:responses?|reactions?|decisions?)",
        r"immediate .*?(?:responses?|reactions?)",
        r"\brapidly\b",
        r"real[- ]time",
        r"live operation",
        r"live traffic",
        r"fast response",
        r"fast enough",
        r"quickly enough",
        r"little delay",
        r"almost no delay",
        r"without .* delay",
        r"avoid delays?",
        r"processing delays?",
        r"too late",
        r"before .* completes",
        r"before .* impact",
        r"right away",
        r"\binstantly\b",
        r"quick responses?",
        r"quick reactions?",
        r"\bruntime\b",
    ),
    "Energy": (
        r"\bbattery\b",
        r"battery[- ]backed",
        r"battery[- ]powered",
        r"battery supply",
        r"battery endurance",
        r"battery replacements?",
        r"preserve (?:its |the )?battery",
        r"battery life",
        r"limited battery",
        r"small battery",
        r"avoid draining",
        r"drain(?:ing)? (?:the )?(?:device|sensor|battery|flight system)",
        r"power draw",
        r"device power draw",
        r"modest (?:device )?power",
        r"tight (?:device )?power budget",
        r"limited power supply",
        r"conserve .*power",
        r"low[- ]power",
        r"energy use",
        r"energy consumption",
        r"energy budget",
        r"energy efficient",
        r"run for months",
        r"recording for months",
        r"last on battery",
        r"between charges",
        r"away from charging",
        r"flight time",
        r"full shift without dying",
        r"long unattended deployment",
        r"long periods away from charging",
        r"remain practical on .* battery",
        r"operate without draining",
        r"long[- ]lived (?:sensor|monitor|device|unit|platform)",
    ),
    "CO2": (
        r"environmental footprint",
        r"environmental burden",
        r"environmental[- ]impact",
        r"climate[- ]conscious",
        r"low[- ]impact (?:computing|deployment|operating|operation|strategy|policy|profile)",
        r"low[- ]impact (?:edge )?environment",
        r"lower[- ]impact (?:computing|deployment|operating|operation|strategy|policy|profile)",
        r"green(?:er)? (?:operating|operation|infrastructure|computing)",
        r"green deployment",
        r"green infrastructure goals",
        r"\bcarbon\b",
        r"\bco2\b",
        r"emissions?",
        r"environmentally conscious",
        r"environmentally considerate",
        r"smaller environmental footprint",
        r"small environmental footprint",
    ),
}

# These phrases are deliberately *not* treated as evidence on their own. They
# are broad deployment descriptions that caused annotation disagreement or could
# refer to multiple concepts.
NON_DECISIVE_PHRASES = (
    "sustained deployment",
    "continuous operation",
    "long-running",
    "long running",
    "resource constrained",
    "resource-constrained",
    "edge environment",
    "edge deployment",
    "sustainable deployment",
    "sustainability",
)


class ObjectiveEvidenceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected: bool
    evidence: Optional[str] = None
    confidence: str = "medium"

    @model_validator(mode="after")
    def validate_decision(self):
        if self.confidence not in _ALLOWED_CONFIDENCE:
            raise ValueError("confidence must be high, medium, or low")
        if self.selected and not str(self.evidence or "").strip():
            raise ValueError("selected objectives require an evidence quote")
        return self


class EvidenceGroundedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    decisions: Dict[str, ObjectiveEvidenceDecision]
    uncertainties: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_payload(self):
        if self.status not in _ALLOWED_STATUS:
            raise ValueError("unsupported status")
        if set(self.decisions) != set(CANONICAL_OBJECTIVES):
            raise ValueError(
                "decisions must contain exactly Accuracy, Runtime, Energy, and CO2"
            )
        return self


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    decisions = dict(out.get("decisions") or {})
    if "CO₂" in decisions and "CO2" not in decisions:
        decisions["CO2"] = decisions.pop("CO₂")
    normalized = {}
    for objective, raw in decisions.items():
        row = dict(raw or {})
        if row.get("confidence") is not None:
            row["confidence"] = str(row.get("confidence")).strip().lower()
        if not bool(row.get("selected")) and not str(row.get("evidence") or "").strip():
            row["evidence"] = None
        normalized[str(objective)] = row
    out["decisions"] = normalized
    if out.get("status") is not None:
        out["status"] = str(out.get("status")).strip().lower()
    return out


def _find_support(objective: str, scenario: str) -> Optional[Dict[str, str]]:
    text = _norm(scenario)
    for idx, pattern in enumerate(CUE_PATTERNS[objective], 1):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return {
                "phrase": match.group(0),
                "pattern_id": "{}_{:02d}".format(objective.lower(), idx),
                "strength": "strong",
            }
    return None


def semantic_support_map(scenario: str) -> Dict[str, Optional[Dict[str, str]]]:
    return {objective: _find_support(objective, scenario) for objective in CANONICAL_OBJECTIVES}


def _quote_is_grounded(quote: str, scenario: str) -> bool:
    q = _norm(quote).strip('"\' ')
    s = _norm(scenario)
    if not q:
        return False
    if q in s:
        return True
    q_tokens = [x for x in re.findall(r"[a-z0-9]+", q) if len(x) > 1]
    s_tokens = set(re.findall(r"[a-z0-9]+", s))
    if len(q_tokens) < 2:
        return False
    coverage = sum(token in s_tokens for token in q_tokens) / float(len(q_tokens))
    return coverage >= 0.85


def _preflight_status(scenario: str) -> Optional[Tuple[str, List[str]]]:
    t = _norm(scenario)
    if not t:
        return "ambiguous", ["No scenario was provided."]
    if not re.search(r"[a-z0-9]", t):
        return "out_of_scope", ["No interpretable deployment requirement was provided."]

    contradiction_patterns = (
        (r"do not care how long it takes", r"answer immediately|respond immediately"),
        (r"response time does not matter", r"respond instantly|respond immediately"),
        (r"ignore (?:device )?power draw", r"battery.*last|tiny battery"),
        (r"ignore both constraints", r"battery.*environmental footprint"),
    )
    for left, right in contradiction_patterns:
        if re.search(left, t) and re.search(right, t):
            return "contradictory", ["The scenario contains explicitly conflicting requirements."]

    unsupported_only = (
        "optimize user happiness",
        "only goal is employee morale",
        "makes the interface prettier",
        "interface prettier",
    )
    if any(phrase in t for phrase in unsupported_only) and not any(
        _find_support(objective, t) for objective in CANONICAL_OBJECTIVES
    ):
        return "out_of_scope", ["The request does not map to the frozen objective vocabulary."]

    if "cheap" in t and "all maximized" in t:
        return "ambiguous", [
            "The request mixes supported and unsupported objectives without a unique valid formulation."
        ]

    generic = (
        "make it good",
        "whatever is best",
        "keep everything balanced",
        "excellent in every possible way",
        "do whatever is best",
    )
    if any(phrase in t for phrase in generic) and not any(
        _find_support(objective, t) for objective in CANONICAL_OBJECTIVES
    ):
        return "ambiguous", ["The request is too underspecified for a unique objective subset."]

    return None


class HybridEvidenceGroundedObjectiveSelectorV31:
    """Recall-balanced hybrid objective selector for interactive AwareML.

    V3.1 keeps the exact locked LLaMA runtime but combines it with explicit,
    scenario-local semantic evidence. Unsupported LLM additions are still
    rejected. Strong semantic cues can recover an objective that LLaMA omitted,
    and can transparently recover from malformed LLM JSON in interactive mode.

    This is a development method. It must not replace the frozen Phase-12 v1
    result and requires a fresh independent evaluation before a final claim.
    """

    def __init__(
        self,
        client: Optional[StrictJournalOllamaClient] = None,
        root: Optional[Path] = None,
        prompt_path: Optional[Path] = None,
    ):
        self.client = client or StrictJournalOllamaClient(root=root)
        self.root = self.client.root
        self.prompt_path = prompt_path or (
            self.root / "prompts" / "objective_selection_evidence_grounded_v31.txt"
        )
        if not self.prompt_path.exists():
            raise JournalModelLockError(
                "V3.1 evidence-grounded prompt is missing: {}".format(self.prompt_path)
            )
        self.template = self.prompt_path.read_text(encoding="utf-8")
        self.last_audit: Dict[str, Any] = {}

    def render_prompt(self, scenario: str) -> str:
        if "{{USER_SCENARIO}}" not in self.template:
            raise JournalModelLockError(
                "V3.1 prompt is missing {{USER_SCENARIO}} placeholder."
            )
        return self.template.replace("{{USER_SCENARIO}}", str(scenario).strip())

    def _preflight_result(self, scenario: str) -> Optional[ObjectiveSelectionResult]:
        preflight = _preflight_status(scenario)
        if preflight is None:
            return None
        status, uncertainties = preflight
        self.last_audit = {
            "selector_id": SELECTOR_ID,
            "stage": "deterministic_preflight",
            "status": status,
            "selected_objectives": [],
            "decisions": {
                objective: {
                    "llm_selected": False,
                    "semantic_support": None,
                    "semantic_support_present": False,
                    "accepted": False,
                    "accepted_by": "none",
                    "reason": "preflight_{}".format(status),
                }
                for objective in CANONICAL_OBJECTIVES
            },
        }
        return ObjectiveSelectionResult(
            status=status,
            selected_objectives=[],
            uncertainties=uncertainties,
            source=SELECTOR_ID + ":preflight",
            model=getattr(self.client, "model", None),
            fallback_used=False,
        )

    def _semantic_only_result(
        self,
        scenario: str,
        error: str,
        stage: str,
    ) -> ObjectiveSelectionResult:
        supports = semantic_support_map(scenario)
        selected = [objective for objective in CANONICAL_OBJECTIVES if supports[objective]]
        decisions = {}
        for objective in CANONICAL_OBJECTIVES:
            support = supports[objective]
            decisions[objective] = {
                "llm_selected": False,
                "confidence": "N/A",
                "evidence_returned": None,
                "evidence_phrase_found": bool(support),
                "objective_specific_support": bool(support),
                "semantic_support": support["phrase"] if support else None,
                "support_pattern_id": support["pattern_id"] if support else None,
                "accepted": bool(support),
                "accepted_by": "semantic_recovery_after_llm_failure" if support else "none",
                "reason": "strong scenario-local semantic evidence" if support else "no objective-specific support",
            }

        if selected:
            uncertainty = (
                "The locked LLaMA response could not be used ({}). V3.1 recovered only objectives with explicit "
                "scenario-local semantic evidence; human review is required.".format(stage)
            )
            self.last_audit = {
                "selector_id": SELECTOR_ID,
                "stage": stage,
                "llm_error": error,
                "final_status": "valid",
                "selected_objectives": selected,
                "semantic_recovery_used": True,
                "decisions": decisions,
                "uncertainties": [uncertainty],
            }
            return ObjectiveSelectionResult(
                status="valid",
                selected_objectives=selected,
                uncertainties=[uncertainty],
                source=SELECTOR_ID + ":semantic-recovery",
                model=getattr(self.client, "model", None),
                fallback_used=True,
            )

        self.last_audit = {
            "selector_id": SELECTOR_ID,
            "stage": stage,
            "llm_error": error,
            "final_status": "ambiguous",
            "selected_objectives": [],
            "semantic_recovery_used": False,
            "decisions": decisions,
        }
        return ObjectiveSelectionResult(
            status="ambiguous",
            selected_objectives=[],
            uncertainties=[
                "No sufficiently specific objective evidence was found. Please clarify the deployment priorities."
            ],
            source=SELECTOR_ID + ":abstain",
            model=getattr(self.client, "model", None),
            fallback_used=False,
        )

    def select(self, scenario: str) -> ObjectiveSelectionResult:
        preflight = self._preflight_result(scenario)
        if preflight is not None:
            return preflight

        supports = semantic_support_map(scenario)
        prompt = self.render_prompt(scenario)

        try:
            payload, meta = self.client.generate_json(prompt)
        except JournalModelLockError:
            raise
        except JournalLLMResponseError as exc:
            return self._semantic_only_result(
                scenario, str(exc), "llm_generation_recovery"
            )

        try:
            parsed = EvidenceGroundedPayload.model_validate(
                _normalize_payload(payload)
            )
        except (ValidationError, ValueError) as exc:
            return self._semantic_only_result(
                scenario,
                "{}: {}".format(type(exc).__name__, exc),
                "schema_validation_recovery",
            )

        selected: List[ObjectiveLabel] = []
        uncertainties = list(parsed.uncertainties)
        decisions_audit: Dict[str, Any] = {}
        recovered_objectives: List[str] = []

        for objective in CANONICAL_OBJECTIVES:
            decision = parsed.decisions[objective]
            quote = str(decision.evidence or "").strip() or None
            grounded = bool(quote and _quote_is_grounded(quote, scenario))
            support = supports[objective]
            support_present = bool(support)

            accepted = False
            accepted_by = "none"
            reason = "no objective-specific support"

            if decision.selected and support_present:
                accepted = True
                accepted_by = "llm_plus_semantic_support"
                reason = "LLM proposal confirmed by scenario-local support"
            elif (not decision.selected) and support_present:
                # This is the key V3.1 recall recovery. The rule is transparent
                # and objective-specific; it does not accept broad generic phrases.
                accepted = True
                accepted_by = "semantic_recovery"
                reason = "strong scenario-local support recovered an LLM omission"
                recovered_objectives.append(objective)
            elif decision.selected and not support_present:
                reason = "LLM proposal rejected: no objective-specific scenario cue"
                uncertainties.append(
                    "Rejected {} selection: no objective-specific cue in scenario.".format(
                        objective
                    )
                )

            if accepted:
                selected.append(objective)

            decisions_audit[objective] = {
                "llm_selected": bool(decision.selected),
                "confidence": decision.confidence,
                "evidence_returned": quote,
                "evidence_grounded": grounded,
                "evidence_phrase_found": bool(quote and grounded),
                "objective_specific_support": support_present,
                "semantic_support": support["phrase"] if support else None,
                "support_pattern_id": support["pattern_id"] if support else None,
                "accepted": accepted,
                "accepted_by": accepted_by,
                "reason": reason,
            }

        final_status = parsed.status
        if selected:
            # When supported evidence exists, a model-level ambiguous status is
            # converted into a review-required valid proposal rather than silently
            # discarding the evidence.
            if final_status == "ambiguous":
                uncertainties.append(
                    "The LLaMA response was marked ambiguous, but explicit objective-specific evidence was found. Human review remains required."
                )
            final_status = "valid"
        elif final_status == "valid":
            final_status = "ambiguous"
            uncertainties.append(
                "No objective survived evidence grounding; explicit user clarification is required."
            )

        if final_status in {"contradictory", "out_of_scope"}:
            selected = []
            recovered_objectives = []
            for item in decisions_audit.values():
                item["accepted"] = False
                item["accepted_by"] = "none"
                item["reason"] = "status_{}".format(final_status)

        if recovered_objectives:
            uncertainties.append(
                "V3.1 semantic recovery added: {}. Review these recovered objectives before accepting the proposal.".format(
                    ", ".join(recovered_objectives)
                )
            )

        self.last_audit = {
            "selector_id": SELECTOR_ID,
            "stage": "hybrid_evidence_grounded_llm",
            "model": meta.get("model"),
            "model_digest": meta.get("model_digest"),
            "ollama_version": meta.get("ollama_version"),
            "raw_status": parsed.status,
            "final_status": final_status,
            "selected_objectives": list(selected),
            "semantic_recovery_used": bool(recovered_objectives),
            "semantic_recovered_objectives": recovered_objectives,
            "non_decisive_phrases": list(NON_DECISIVE_PHRASES),
            "decisions": decisions_audit,
            "uncertainties": list(uncertainties),
        }

        return ObjectiveSelectionResult(
            status=final_status,
            selected_objectives=selected,
            uncertainties=uncertainties,
            source=SELECTOR_ID,
            model=meta.get("model"),
            fallback_used=bool(recovered_objectives),
        )
