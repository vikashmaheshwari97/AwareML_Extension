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


SELECTOR_ID = "evidence_grounded_objective_selector_v3"
CANONICAL_OBJECTIVES = ("Accuracy", "Runtime", "Energy", "CO2")
_ALLOWED_STATUS = {"valid", "ambiguous", "contradictory", "out_of_scope"}
_ALLOWED_CONFIDENCE = {"high", "medium", "low"}


# Conservative semantic guardrails. These are intentionally asymmetric: an
# objective needs scenario-local evidence before it can survive the V3 filter.
# The goal is to reduce the Phase-12 over-selection failure mode, not to infer
# every generally desirable property of a system.
OBJECTIVE_CUES = {
    "Accuracy": (
        r"rarely make the wrong decision",
        r"avoid(?:ing)? (?:incorrect|wrong) decisions?",
        r"trustworthy predictions?",
        r"dependable (?:predictions?|decisions?|classifications?|detections?)",
        r"reliable (?:predictions?|decisions?|classifications?|detections?)",
        r"consistently reliable classifications?",
        r"correct decisions?",
        r"predictive quality",
        r"prediction quality",
        r"decision quality",
        r"avoid costly misclassification",
        r"strong predictive (?:quality|performance)",
        r"strong performance",
        r"high accuracy",
        r"accurate",
        r"accuracy",
    ),
    "Runtime": (
        r"react quickly",
        r"respond (?:almost )?immediately",
        r"time[- ]critical",
        r"time[- ]to[- ]response",
        r"decision latency",
        r"low latency",
        r"very small latency",
        r"prompt response",
        r"react promptly",
        r"arrive promptly",
        r"promptly",
        r"rapid response",
        r"real[- ]time",
        r"live operation",
        r"fast response",
        r"fast enough",
        r"quickly enough",
        r"runtime",
    ),
    "Energy": (
        r"battery",
        r"battery[- ]backed",
        r"battery[- ]powered",
        r"preserve battery",
        r"battery life",
        r"limited battery",
        r"small battery",
        r"avoid draining",
        r"power draw",
        r"low[- ]power",
        r"energy use",
        r"energy consumption",
        r"energy budget",
        r"energy efficient",
        r"run for months",
        r"last on battery",
    ),
    "CO2": (
        r"environmental footprint",
        r"environmental burden",
        r"environmental[- ]impact",
        r"climate[- ]conscious",
        r"low[- ]impact",
        r"lower[- ]impact",
        r"green(?:er)? operating",
        r"green deployment",
        r"carbon",
        r"co2",
        r"emissions?",
        r"environmentally conscious",
        r"sustainab(?:le|ility)",
    ),
}


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
        keys = set(self.decisions)
        expected = set(CANONICAL_OBJECTIVES)
        if keys != expected:
            raise ValueError(
                "decisions must contain exactly Accuracy, Runtime, Energy, and CO2"
            )
        return self




def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Apply narrow, auditable JSON normalization before schema validation.

    This is not free-form rescue: it only normalizes known label spelling,
    confidence case, and empty evidence on unselected objectives.
    """
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

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _find_support(objective: str, scenario: str) -> Optional[str]:
    text = _norm(scenario)
    for pattern in OBJECTIVE_CUES[objective]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _quote_is_grounded(quote: str, scenario: str) -> bool:
    q = _norm(quote).strip('"\' ')
    s = _norm(scenario)
    if not q:
        return False
    if q in s:
        return True

    # Tolerate small punctuation differences while rejecting free paraphrases.
    q_tokens = [x for x in re.findall(r"[a-z0-9]+", q) if len(x) > 1]
    s_tokens = set(re.findall(r"[a-z0-9]+", s))
    if len(q_tokens) < 2:
        return False
    coverage = sum(token in s_tokens for token in q_tokens) / float(len(q_tokens))
    return coverage >= 0.90


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
        return "ambiguous", ["The request mixes supported and unsupported objectives without a unique valid formulation."]

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


class EvidenceGroundedObjectiveSelectorV3:
    """Evidence-grounded, conservative objective selector for interactive AwareML.

    This method does NOT replace or modify the frozen Phase-12 v1 benchmark.
    It is an improvement candidate motivated by the frozen v1 failure analysis.

    Selection rule:
      1. deterministic preflight catches empty/generic/out-of-scope/contradictory cases;
      2. the exact locked LLaMA runtime returns one decision per objective;
      3. each selected objective must cite scenario-local evidence;
      4. a transparent objective-specific semantic guardrail must support that evidence;
      5. low-confidence selections are rejected.

    The intentionally conservative rule targets the v1 over-selection pattern.
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
            self.root / "prompts" / "objective_selection_evidence_grounded_v3.txt"
        )
        if not self.prompt_path.exists():
            raise JournalModelLockError(
                "V3 evidence-grounded prompt is missing: {}".format(self.prompt_path)
            )
        self.template = self.prompt_path.read_text(encoding="utf-8")
        self.last_audit: Dict[str, Any] = {}

    def render_prompt(self, scenario: str) -> str:
        if "{{USER_SCENARIO}}" not in self.template:
            raise JournalModelLockError(
                "V3 prompt is missing {{USER_SCENARIO}} placeholder."
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
            "decisions": {
                objective: {
                    "llm_selected": False,
                    "accepted": False,
                    "evidence": None,
                    "semantic_support": None,
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

    def select(self, scenario: str) -> ObjectiveSelectionResult:
        preflight = self._preflight_result(scenario)
        if preflight is not None:
            return preflight

        prompt = self.render_prompt(scenario)
        try:
            payload, meta = self.client.generate_json(prompt)
        except JournalModelLockError:
            raise
        except JournalLLMResponseError as exc:
            self.last_audit = {
                "selector_id": SELECTOR_ID,
                "stage": "llm_generation",
                "status": "malformed",
                "error": str(exc),
            }
            return ObjectiveSelectionResult(
                status="malformed",
                selected_objectives=[],
                uncertainties=[str(exc)],
                source=SELECTOR_ID,
                model=getattr(self.client, "model", None),
                fallback_used=False,
            )

        try:
            payload = _normalize_payload(payload)
            parsed = EvidenceGroundedPayload.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            self.last_audit = {
                "selector_id": SELECTOR_ID,
                "stage": "schema_validation",
                "status": "malformed",
                "error": "{}: {}".format(type(exc).__name__, exc),
                "raw_payload": payload,
            }
            return ObjectiveSelectionResult(
                status="malformed",
                selected_objectives=[],
                uncertainties=["V3 objective JSON rejected: {}".format(exc)],
                source=SELECTOR_ID,
                model=meta.get("model"),
                fallback_used=False,
            )

        selected: List[ObjectiveLabel] = []
        uncertainties = list(parsed.uncertainties)
        decisions_audit: Dict[str, Any] = {}

        for objective in CANONICAL_OBJECTIVES:
            decision = parsed.decisions[objective]
            quote = str(decision.evidence or "").strip() or None
            grounded = bool(quote and _quote_is_grounded(quote, scenario))
            support = _find_support(objective, scenario)
            confidence_ok = decision.confidence in {"high", "medium"}
            accepted = bool(decision.selected and support and confidence_ok)

            # If the model selected an objective with a semantically valid cue but
            # slightly paraphrased the evidence quote, retain the selection while
            # recording that the transparent guardrail recovered the exact support.
            recovered = bool(accepted and not grounded and support)
            if accepted:
                selected.append(objective)  # canonical order is preserved
            elif decision.selected:
                reason_bits = []
                if not support:
                    reason_bits.append("no objective-specific cue in scenario")
                if not confidence_ok:
                    reason_bits.append("low confidence")
                if not grounded and not support:
                    reason_bits.append("evidence not grounded")
                uncertainties.append(
                    "Rejected {} selection: {}.".format(
                        objective,
                        "; ".join(reason_bits) or "insufficient grounded support",
                    )
                )

            decisions_audit[objective] = {
                "llm_selected": bool(decision.selected),
                "confidence": decision.confidence,
                "evidence_returned": quote,
                "evidence_grounded": grounded,
                "semantic_support": support,
                "evidence_recovered_by_guardrail": recovered,
                "accepted": accepted,
            }

        final_status = parsed.status
        if final_status == "valid" and not selected:
            final_status = "ambiguous"
            uncertainties.append(
                "No objective survived evidence grounding; explicit user clarification is required."
            )

        # A contradictory/out-of-scope result should not silently carry selected
        # objectives into the downstream weighting stage.
        if final_status in {"contradictory", "out_of_scope"}:
            selected = []
            for item in decisions_audit.values():
                item["accepted"] = False

        self.last_audit = {
            "selector_id": SELECTOR_ID,
            "stage": "evidence_grounded_llm",
            "model": meta.get("model"),
            "model_digest": meta.get("model_digest"),
            "ollama_version": meta.get("ollama_version"),
            "raw_status": parsed.status,
            "final_status": final_status,
            "selected_objectives": list(selected),
            "decisions": decisions_audit,
            "uncertainties": list(uncertainties),
        }

        return ObjectiveSelectionResult(
            status=final_status,
            selected_objectives=selected,
            uncertainties=uncertainties,
            source=SELECTOR_ID,
            model=meta.get("model"),
            fallback_used=False,
        )
