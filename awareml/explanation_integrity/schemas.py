from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceCase:
    case_id: str
    dataset_id: str
    source_stage: str
    source_name: str
    evidence: Dict[str, Any]
    prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Claim:
    claim_id: str
    text: str
    claim_type: str
    metric: Optional[str] = None
    entity: Optional[str] = None
    feature: Optional[str] = None
    numeric_value: Optional[float] = None
    rank_value: Optional[int] = None
    comparator: Optional[str] = None
    cited_keys: List[str] = field(default_factory=list)
    span_start: Optional[int] = None
    span_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimVerification:
    claim: Claim
    status: str
    supported: bool
    contradicted: bool
    unsupported: bool
    numeric_correct: Optional[bool] = None
    citation_valid: Optional[bool] = None
    decision_consistent: Optional[bool] = None
    matched_evidence_key: Optional[str] = None
    expected_value: Any = None
    observed_value: Any = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["claim"] = self.claim.to_dict()
        return payload


@dataclass
class CorrectnessReport:
    case_id: str
    dataset_id: str
    source_stage: str
    source_name: str
    explanation: str
    claims: List[ClaimVerification]
    metrics: Dict[str, Optional[float]]
    evidence_key_count: int
    explanation_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset_id": self.dataset_id,
            "source_stage": self.source_stage,
            "source_name": self.source_name,
            "explanation": self.explanation,
            "claims": [claim.to_dict() for claim in self.claims],
            "metrics": dict(self.metrics),
            "evidence_key_count": int(self.evidence_key_count),
            "explanation_sha256": self.explanation_sha256,
        }


@dataclass
class Intervention:
    intervention_id: str
    kind: str
    evidence_key: str
    original_value: Any
    counterfactual_value: Any
    relevant: bool = True
    expected_decision_change: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FaithfulnessRecord:
    case_id: str
    dataset_id: str
    source_stage: str
    source_name: str
    intervention: Intervention
    original_explanation: str
    counterfactual_explanation: str
    original_correctness: Dict[str, Optional[float]]
    counterfactual_correctness: Dict[str, Optional[float]]
    changed_evidence_acknowledged: float
    numeric_update_accuracy: Optional[float]
    stale_claim_rate: float
    decision_update_consistency: Optional[float]
    irrelevant_invariance: Optional[float]
    explanation_change: float
    faithfulness_score: float
    generator_source: str
    model: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["intervention"] = self.intervention.to_dict()
        return payload


@dataclass
class StimulusRecord:
    stimulus_id: str
    pair_id: str
    dataset_id: str
    source_stage: str
    source_name: str
    prompt: str
    explanation: str
    evidence_summary: Dict[str, Any]
    researcher_label: str
    error_type: Optional[str]
    verifier_metrics: Dict[str, Optional[float]]
    participant_version: Dict[str, Any]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
