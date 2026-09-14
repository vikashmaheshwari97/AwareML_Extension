from __future__ import annotations

"""Evidence-grounded objective selector V3.2.

This module is a confirmatory successor to the post-hoc V3.1 development
selector.  It intentionally keeps the frozen LLaMA 3 8B model weights under the separate
Phase-11R confirmatory runtime as the language model and reduces the deterministic layer to a small concept-level
validation guard.

Two operating modes are supported:

* benchmark_mode=True (default): the LLM proposes labels; the deterministic
  guard may reject unsupported additions but NEVER adds a label that the LLM
  omitted. Malformed/model-schema failures remain visible as ``malformed``.
* benchmark_mode=False + allow_semantic_recovery=True: interactive AwareML may
  recover an omitted label from explicit scenario-local concept evidence.  Any
  such recovery is clearly recorded in ``fallback_used`` and ``last_audit``.

The confirmatory Phase-12-v2 benchmark MUST use benchmark_mode=True.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .journal_client import JournalLLMResponseError, JournalModelLockError
from .confirmatory_runtime_v32 import ConfirmatoryOllamaClientV32
from .schemas import ObjectiveLabel, ObjectiveSelectionResult


SELECTOR_ID = "evidence_grounded_objective_selector_v32"
SELECTOR_VERSION = "3.2"
CANONICAL_OBJECTIVES = ("Accuracy", "Runtime", "Energy", "CO2")
ALLOWED_STATUS = {"valid", "ambiguous", "contradictory", "out_of_scope"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}

# Compact concept-level guards.  These are deliberately NOT a catalogue of old
# benchmark phrasings.  Their job is only to verify that an LLM-selected label
# is supported by local evidence of the right semantic kind.
CONCEPT_GUARDS: Dict[str, Tuple[str, ...]] = {
    "Accuracy": (
        r"\bcorrect(?:ness|ly)?\b",
        r"\bincorrect\b",
        r"\bwrong\b",
        r"\berrors?\b",
        r"\bmisclassif(?:y|ied|ication|ications)\b",
        r"\bfalse (?:positive|negative|alarm)s?\b",
        r"\bmiss(?:ed|ing)?\b.{0,24}\b(?:case|event|issue|fault|detection|diagnosis)s?\b",
        r"\b(?:reliable|dependable|trustworthy)\b.{0,40}\b(?:prediction|decision|classification|detection|output|result|identification)s?\b",
        r"\b(?:prediction|predictive|decision) quality\b",
        r"\b(?:consistently )?right\b(?!\s+away)",
    ),
    "Runtime": (
        r"\blatenc(?:y|ies)\b",
        r"\bdelay(?:ed|s)?\b",
        r"\bdeadline(?:s)?\b",
        r"\bresponse time(?:s)?\b",
        r"\btime[- ]critical\b",
        r"\breal[- ]time\b",
        r"\b(?:respond|react|answer|return)(?:s|ed|ing)?\b.{0,24}\b(?:quickly|promptly|immediately|rapidly|fast)\b",
        r"\b(?:quick|rapid|immediate|fast|prompt)\b.{0,24}\b(?:response|reaction|decision|warning|answer)s?\b",
        r"\b(?:arrive|arrives|arrived|arriving)\b.{0,20}\b(?:promptly|quickly|immediately|rapidly)\b",
        r"\bin time\b",
    ),
    "Energy": (
        r"\bbatter(?:y|ies)\b",
        r"\bcharg(?:e|ed|ing|er|ers)\b",
        r"\bpower (?:draw|budget|consumption|use|usage|supply)\b",
        r"\belectrical (?:power|consumption|use|load)\b",
        r"\belectricity (?:consumption|use|usage|demand)\b",
        r"\b(?:drain|draining|preserve|conserve)\b.{0,28}\b(?:battery|power)\b",
        r"\blow[- ]power\b",
        r"\bwatt(?:s|age)?\b",
    ),
    "CO2": (
        r"\bcarbon\b",
        r"\bco2\b",
        r"\bco₂\b",
        r"\bemissions?\b",
        r"\bgreenhouse gas(?:es)?\b",
        r"\bclimate (?:impact|footprint|burden)\b",
        r"\benvironmental (?:impact|footprint|burden)\b",
        r"\b(?:lower|reduce|minimi[sz]e|small|smaller|low)\b.{0,24}\benvironmental footprint\b",
        r"\bgreen(?:er)? (?:computing|operation|deployment|infrastructure)\b",
        r"\blow[- ]impact computing\b",
    ),
}

# Context alone is not positive evidence for any of the four labels.
NON_DECISIVE_CONTEXT = (
    "edge deployment",
    "edge environment",
    "resource constrained",
    "resource-constrained",
    "continuous operation",
    "long-running",
    "long running",
    "sustained deployment",
    "sustainable deployment",
    "sustainability",
    "cloud deployment",
)

GENERIC_AMBIGUOUS_PATTERNS = (
    r"\bmake it good\b",
    r"\bwhatever is best\b",
    r"\bdo whatever is best\b",
    r"\bkeep everything balanced\b",
    r"\bmake the system better\b",
    r"\bexcellent in every possible way\b",
)

OUT_OF_SCOPE_PATTERNS = (
    r"\buser happiness\b",
    r"\bemployee morale\b",
    r"\binterface (?:prettier|appearance|colour|color)\b",
    r"\bmarketing engagement\b",
    r"\bbrand awareness\b",
)

# Explicit statements that an otherwise supported objective is irrelevant.
NEGATIVE_REQUIREMENT_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "Accuracy": (
        r"\b(?:ignore|disregard)\b.{0,28}\b(?:correctness|errors?|prediction quality)\b",
        r"\b(?:correctness|prediction quality)\b.{0,24}\b(?:does not|doesn['’]t|do not|don['’]t) matter\b",
    ),
    "Runtime": (
        r"\b(?:ignore|disregard)\b.{0,28}\b(?:latency|delay|response time|speed)\b",
        r"\b(?:latency|response time|speed)\b.{0,24}\b(?:does not|doesn['’]t|do not|don['’]t) matter\b",
        r"\bdo not care how long it takes\b",
    ),
    "Energy": (
        r"\b(?:ignore|disregard)\b.{0,28}\b(?:battery|power|electricity)\b",
        r"\b(?:battery life|power use|power consumption)\b.{0,24}\b(?:does not|doesn['’]t|do not|don['’]t) matter\b",
    ),
    "CO2": (
        r"\b(?:ignore|disregard)\b.{0,28}\b(?:carbon|emissions?|environmental footprint|climate impact)\b",
        r"\b(?:carbon|emissions?|environmental footprint|climate impact)\b.{0,24}\b(?:does not|doesn['’]t|do not|don['’]t) matter\b",
    ),
}

LOCAL_NEGATION_RE = re.compile(
    r"\b(?:ignore|disregard|irrelevant|not important|not a concern|no concern|"
    r"do not care|don['’]t care|does not matter|doesn['’]t matter|no need)\b",
    re.IGNORECASE,
)


class ObjectiveEvidenceDecisionV32(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected: bool
    evidence: Optional[str] = None
    confidence: str = "medium"

    @model_validator(mode="after")
    def validate_decision(self):
        self.confidence = str(self.confidence).strip().lower()
        if self.confidence not in ALLOWED_CONFIDENCE:
            raise ValueError("confidence must be high, medium, or low")
        if self.selected and not str(self.evidence or "").strip():
            raise ValueError("selected objectives require an evidence quote")
        if not self.selected and not str(self.evidence or "").strip():
            self.evidence = None
        return self


class EvidenceGroundedPayloadV32(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    decisions: Dict[str, ObjectiveEvidenceDecisionV32]
    uncertainties: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_payload(self):
        self.status = str(self.status).strip().lower()
        if self.status not in ALLOWED_STATUS:
            raise ValueError("unsupported status")
        if set(self.decisions) != set(CANONICAL_OBJECTIVES):
            raise ValueError(
                "decisions must contain exactly Accuracy, Runtime, Energy, and CO2"
            )
        return self


def _norm(text: str) -> str:
    text = str(text or "").replace("₂", "2")
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    decisions = dict(out.get("decisions") or {})
    if "CO₂" in decisions and "CO2" not in decisions:
        decisions["CO2"] = decisions.pop("CO₂")
    normalized: Dict[str, Any] = {}
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


def _quote_is_grounded(quote: str, scenario: str) -> bool:
    """Require a scenario-local quote, allowing whitespace normalization only."""
    q = _norm(quote).strip('"\' ')
    s = _norm(scenario)
    return bool(q and q in s)


def _negative_requirement_present(objective: str, text: str) -> bool:
    t = _norm(text)
    return any(re.search(p, t, flags=re.IGNORECASE) for p in NEGATIVE_REQUIREMENT_PATTERNS[objective])


def _locally_negated(text: str, start: int, end: int) -> bool:
    # Restrict negation to the same short clause. This prevents
    # "latency does not matter, but respond immediately" from negating the
    # positive requirement after "but".
    boundary_tokens = (".", ";", ",", "!", "?", " but ", " however ")
    left = max(0, start - 64)
    for token in boundary_tokens:
        pos = text.rfind(token, left, start)
        if pos >= 0:
            left = max(left, pos + len(token))
    right = min(len(text), end + 64)
    for token in boundary_tokens:
        pos = text.find(token, end, right)
        if pos >= 0:
            right = min(right, pos)
    return bool(LOCAL_NEGATION_RE.search(text[left:right]))


def _find_concept_support(objective: str, text: str) -> Optional[Dict[str, str]]:
    t = _norm(text)
    for idx, pattern in enumerate(CONCEPT_GUARDS[objective], 1):
        match = re.search(pattern, t, flags=re.IGNORECASE)
        if match and not _locally_negated(t, match.start(), match.end()):
            return {
                "phrase": match.group(0),
                "guard_id": "{}_concept_{:02d}".format(objective.lower(), idx),
            }
    return None


def semantic_support_map_v32(text: str) -> Dict[str, Optional[Dict[str, str]]]:
    return {
        objective: _find_concept_support(objective, text)
        for objective in CANONICAL_OBJECTIVES
    }


def _preflight_status(scenario: str) -> Optional[Tuple[str, List[str]]]:
    t = _norm(scenario)
    if not t:
        return "ambiguous", ["No scenario was provided."]
    if not re.search(r"[a-z0-9]", t):
        return "out_of_scope", ["No interpretable deployment requirement was provided."]

    supports = semantic_support_map_v32(t)

    # Contradiction is deliberately narrow: a supported positive requirement and
    # an explicit statement that the same requirement is irrelevant.
    contradictory = [
        objective
        for objective in CANONICAL_OBJECTIVES
        if supports[objective] and _negative_requirement_present(objective, t)
    ]
    if contradictory:
        return "contradictory", [
            "The scenario explicitly both requests and dismisses: {}.".format(
                ", ".join(contradictory)
            )
        ]

    if not any(supports.values()):
        if any(re.search(p, t) for p in GENERIC_AMBIGUOUS_PATTERNS):
            return "ambiguous", [
                "The request is too underspecified for a unique objective subset."
            ]
        if any(re.search(p, t) for p in OUT_OF_SCOPE_PATTERNS):
            return "out_of_scope", [
                "The request does not map to Accuracy, Runtime, Energy, or CO2."
            ]

    return None


class EvidenceGroundedObjectiveSelectorV32:
    """Precision-oriented, evidence-grounded objective selector.

    In benchmark mode the deterministic layer is a validator, not a second
    classifier: it can reject an unsupported LLM selection but cannot recover an
    omitted objective. This preserves the evaluated model's errors rather than
    hiding them behind post-processing.
    """

    def __init__(
        self,
        client: Optional[ConfirmatoryOllamaClientV32] = None,
        root: Optional[Path] = None,
        prompt_path: Optional[Path] = None,
        *,
        benchmark_mode: bool = True,
        allow_semantic_recovery: bool = False,
    ):
        self.client = client or ConfirmatoryOllamaClientV32(root=root)
        self.root = Path(root).resolve() if root is not None else Path(self.client.root).resolve()
        self.prompt_path = prompt_path or (
            self.root / "prompts" / "objective_selection_evidence_grounded_v32.txt"
        )
        if not self.prompt_path.exists():
            raise JournalModelLockError(
                "V3.2 evidence-grounded prompt is missing: {}".format(self.prompt_path)
            )
        self.template = self.prompt_path.read_text(encoding="utf-8")
        if "{{USER_SCENARIO}}" not in self.template:
            raise JournalModelLockError(
                "V3.2 prompt is missing {{USER_SCENARIO}} placeholder."
            )
        self.benchmark_mode = bool(benchmark_mode)
        self.allow_semantic_recovery = bool(allow_semantic_recovery) and not self.benchmark_mode
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

    def select(self, scenario: str) -> ObjectiveSelectionResult:
        preflight = _preflight_status(scenario)
        if preflight is not None:
            status, uncertainties = preflight
            self.last_audit = {
                "selector_id": SELECTOR_ID,
                "version": SELECTOR_VERSION,
                "benchmark_mode": self.benchmark_mode,
                "stage": "preflight",
                "status": status,
                "selected_objectives": [],
                "uncertainties": uncertainties,
            }
            return self._result(
                status=status,
                selected=[],
                uncertainties=uncertainties,
                source_suffix="preflight",
                model=getattr(self.client, "model", None),
                fallback_used=False,
            )

        prompt = self.render_prompt(scenario)
        try:
            payload, meta = self.client.generate_json(prompt)
        except JournalModelLockError:
            # Wrong model/digest/runtime is benchmark invalidation, never a parse
            # error and never eligible for semantic recovery.
            raise
        except JournalLLMResponseError as exc:
            uncertainties = [str(exc)]
            self.last_audit = {
                "selector_id": SELECTOR_ID,
                "version": SELECTOR_VERSION,
                "benchmark_mode": self.benchmark_mode,
                "stage": "llm_generation",
                "status": "malformed",
                "error": str(exc),
                "selected_objectives": [],
            }
            return self._result(
                status="malformed",
                selected=[],
                uncertainties=uncertainties,
                source_suffix="malformed",
                model=getattr(self.client, "model", None),
                fallback_used=False,
            )

        try:
            parsed = EvidenceGroundedPayloadV32.model_validate(
                _normalize_payload(payload)
            )
        except (ValidationError, ValueError, TypeError) as exc:
            message = "V3.2 response schema validation failed: {}: {}".format(
                type(exc).__name__, exc
            )
            self.last_audit = {
                "selector_id": SELECTOR_ID,
                "version": SELECTOR_VERSION,
                "benchmark_mode": self.benchmark_mode,
                "stage": "schema_validation",
                "status": "malformed",
                "error": message,
                "selected_objectives": [],
            }
            return self._result(
                status="malformed",
                selected=[],
                uncertainties=[message],
                source_suffix="malformed",
                model=meta.get("model") or getattr(self.client, "model", None),
                fallback_used=False,
            )

        # Non-valid model statuses always abstain.  This avoids converting a model
        # level ambiguity/rejection into a seemingly valid benchmark prediction.
        if parsed.status != "valid":
            self.last_audit = {
                "selector_id": SELECTOR_ID,
                "version": SELECTOR_VERSION,
                "benchmark_mode": self.benchmark_mode,
                "stage": "llm_status",
                "raw_status": parsed.status,
                "final_status": parsed.status,
                "selected_objectives": [],
                "decisions": {
                    obj: {
                        "llm_selected": bool(parsed.decisions[obj].selected),
                        "accepted": False,
                        "reason": "model_status_{}".format(parsed.status),
                    }
                    for obj in CANONICAL_OBJECTIVES
                },
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
        uncertainties = list(parsed.uncertainties)
        decisions_audit: Dict[str, Any] = {}
        semantic_recovered: List[str] = []

        for objective in CANONICAL_OBJECTIVES:
            decision = parsed.decisions[objective]
            quote = str(decision.evidence or "").strip() or None
            grounded = bool(quote and _quote_is_grounded(quote, scenario))
            quote_support = _find_concept_support(objective, quote or "") if grounded else None
            scenario_support = _find_concept_support(objective, scenario)

            accepted = False
            accepted_by = "none"
            reason = "not selected by LLM"

            if decision.selected:
                if not grounded:
                    reason = "rejected: evidence is not an exact scenario-local quote"
                    uncertainties.append(
                        "Rejected {}: evidence was not grounded in the scenario.".format(objective)
                    )
                elif not quote_support:
                    reason = "rejected: quoted evidence does not support this objective concept"
                    uncertainties.append(
                        "Rejected {}: quoted evidence did not contain objective-specific support.".format(objective)
                    )
                else:
                    accepted = True
                    accepted_by = "llm_plus_grounded_concept_guard"
                    reason = "accepted: grounded evidence passed concept guard"
            elif self.allow_semantic_recovery and scenario_support:
                accepted = True
                accepted_by = "interactive_semantic_recovery"
                reason = "interactive-only recovery from explicit scenario evidence"
                semantic_recovered.append(objective)

            if accepted:
                selected.append(objective)  # type: ignore[arg-type]

            decisions_audit[objective] = {
                "llm_selected": bool(decision.selected),
                "confidence": decision.confidence,
                "evidence_returned": quote,
                "evidence_grounded": grounded,
                "quote_concept_support": quote_support,
                "scenario_concept_support": scenario_support,
                "accepted": accepted,
                "accepted_by": accepted_by,
                "reason": reason,
            }

        final_status = "valid" if selected else "ambiguous"
        if not selected:
            uncertainties.append(
                "No LLM-selected objective survived evidence grounding; clarification is required."
            )

        fallback_used = bool(semantic_recovered)
        if semantic_recovered:
            uncertainties.append(
                "Interactive semantic recovery added: {}.".format(
                    ", ".join(semantic_recovered)
                )
            )

        self.last_audit = {
            "selector_id": SELECTOR_ID,
            "version": SELECTOR_VERSION,
            "benchmark_mode": self.benchmark_mode,
            "allow_semantic_recovery": self.allow_semantic_recovery,
            "stage": "evidence_grounded_llm",
            "model": meta.get("model"),
            "model_digest": meta.get("model_digest"),
            "ollama_version": meta.get("ollama_version"),
            "raw_status": parsed.status,
            "final_status": final_status,
            "selected_objectives": list(selected),
            "semantic_recovery_used": fallback_used,
            "semantic_recovered_objectives": semantic_recovered,
            "non_decisive_context": list(NON_DECISIVE_CONTEXT),
            "decisions": decisions_audit,
            "uncertainties": list(uncertainties),
        }

        return self._result(
            status=final_status,
            selected=selected,
            uncertainties=uncertainties,
            source_suffix="benchmark" if self.benchmark_mode else "interactive",
            model=meta.get("model"),
            fallback_used=fallback_used,
        )
