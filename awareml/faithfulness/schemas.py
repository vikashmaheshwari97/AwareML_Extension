from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RationaleRecord(BaseModel):
    text: str
    source: str
    model: Optional[str] = None
    evidence_keys: List[str] = Field(
        default_factory=list
    )
    cited_objectives: List[str] = Field(
        default_factory=list
    )
    citation_validity: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
    )
    top_framework_mentioned: bool = False
    top_influence_objective: Optional[str] = None
    top_influence_cited: bool = False
    warnings: List[str] = Field(
        default_factory=list
    )


class CounterfactualRecord(BaseModel):
    scenario: str
    changed_objectives: List[str]
    original_top_framework: str
    counterfactual_top_framework: str
    decision_flipped: bool

    original_rationale: RationaleRecord
    counterfactual_rationale: RationaleRecord

    changed_objective_cited: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
    )
    new_winner_acknowledged: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
    )
    citation_change: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
    )
    counterfactual_sensitivity: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
    )


class DatasetFaithfulnessResult(BaseModel):
    dataset_id: str
    explanation_source: str
    model: Optional[str] = None

    original_top_framework: str
    original_top_influence_objective: Optional[str] = None

    grounding_validity: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
    )
    decision_alignment: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
    )
    attribution_alignment: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
    )
    counterfactual_sensitivity: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
    )
    irrelevant_invariance: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
    )

    evidence_fidelity_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
    )

    counterfactuals: List[
        CounterfactualRecord
    ] = Field(
        default_factory=list
    )
