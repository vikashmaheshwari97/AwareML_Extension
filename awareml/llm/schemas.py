from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


ObjectiveLabel = Literal["Accuracy", "Runtime", "Energy", "CO2"]
ObjectiveSelectionStatus = Literal[
    "valid",
    "ambiguous",
    "malformed",
    "out_of_scope",
    "contradictory",
]


class PrimaryObjectiveWeights(BaseModel):
    """Exactly four empirical objectives used by ML Recommender V2."""

    accuracy: float = Field(0.55, ge=0)
    runtime: float = Field(0.15, ge=0)
    energy: float = Field(0.15, ge=0)
    co2: float = Field(0.15, ge=0)

    @model_validator(mode="after")
    def positive_total(self):
        if self.accuracy + self.runtime + self.energy + self.co2 <= 0:
            raise ValueError("At least one primary objective must be positive.")
        return self

    def normalized_dict(self) -> Dict[str, float]:
        values = {
            "accuracy": max(0.0, float(self.accuracy)),
            "runtime": max(0.0, float(self.runtime)),
            "energy": max(0.0, float(self.energy)),
            "co2": max(0.0, float(self.co2)),
        }
        total = float(sum(values.values()))
        return {key: value / total for key, value in values.items()}


class HCAIRequirements(BaseModel):
    """Human-centred requirements kept outside the four-objective utility."""

    model_config = ConfigDict(extra="forbid")

    drift_sensitivity: Literal["low", "moderate", "high"] = "moderate"
    energy_constraint: Literal["low", "moderate", "strict"] = "low"
    fairness_required: bool = False
    fairness_metric: Optional[str] = None
    explainability_level: Literal["low", "moderate", "high"] = "moderate"
    max_runtime_sec: Optional[float] = Field(None, gt=0)


class ObjectiveSelectionResult(BaseModel):
    """Problem A: scenario -> objective subset.

    This is deliberately separate from downstream objective weighting.
    """

    model_config = ConfigDict(extra="forbid")

    status: ObjectiveSelectionStatus
    selected_objectives: List[ObjectiveLabel] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)

    source: str = "unknown"
    model: Optional[str] = None
    fallback_used: bool = False

    @model_validator(mode="after")
    def validate_selection(self):
        canonical = ["Accuracy", "Runtime", "Energy", "CO2"]
        seen = set()
        ordered = []
        for objective in canonical:
            if objective in self.selected_objectives and objective not in seen:
                ordered.append(objective)
                seen.add(objective)
        if len(ordered) != len(self.selected_objectives):
            raise ValueError("selected_objectives must be unique canonical labels.")
        self.selected_objectives = ordered

        if self.status == "valid" and not self.selected_objectives:
            raise ValueError("A valid selection must contain at least one objective.")
        return self


class GoalInterpretation(BaseModel):
    """Roadmap-aligned Copilot interpretation.

    The primary journal output is selected_objectives. The equal-weight mapping
    is recorded separately so objective selection can be evaluated on its own.
    """

    task: Literal["streaming_classification"] = "streaming_classification"

    selection_status: ObjectiveSelectionStatus = "valid"
    selected_objectives: List[ObjectiveLabel] = Field(default_factory=lambda: ["Accuracy"])
    selection_source: str = "deterministic"
    selection_model: Optional[str] = None
    fallback_used: bool = False

    weighting_policy: Literal["equal_selected_v1"] = "equal_selected_v1"
    primary_weights: PrimaryObjectiveWeights = Field(
        default_factory=lambda: PrimaryObjectiveWeights(
            accuracy=1.0,
            runtime=0.0,
            energy=0.0,
            co2=0.0,
        )
    )

    hcai_requirements: HCAIRequirements = Field(default_factory=HCAIRequirements)

    summary: str = "Streaming classification objective interpretation."
    uncertainties: List[str] = Field(default_factory=list)

    @property
    def drift_sensitivity(self):
        return self.hcai_requirements.drift_sensitivity

    @property
    def energy_constraint(self):
        return self.hcai_requirements.energy_constraint

    @property
    def fairness_required(self):
        return self.hcai_requirements.fairness_required

    @property
    def fairness_metric(self):
        return self.hcai_requirements.fairness_metric

    @property
    def explainability_level(self):
        return self.hcai_requirements.explainability_level

    @property
    def max_runtime_sec(self):
        return self.hcai_requirements.max_runtime_sec


class CopilotConfiguration(BaseModel):
    framework: str
    algorithm: str
    framework_parameters: Dict[str, Any] = Field(default_factory=dict)

    window_size: int = Field(1000, ge=50)
    time_budget_sec: float = Field(60.0, gt=0)

    drift: Dict[str, Any] = Field(default_factory=dict)
    fairness: Dict[str, Any] = Field(default_factory=dict)
    explainability: Dict[str, Any] = Field(default_factory=dict)
    sustainability: Dict[str, Any] = Field(default_factory=dict)

    primary_weights: PrimaryObjectiveWeights = Field(
        default_factory=PrimaryObjectiveWeights
    )


class ConfigDiffItem(BaseModel):
    path: str
    before: Any = None
    after: Any = None


class CopilotProposal(BaseModel):
    proposal_id: str
    goal: str
    interpretation: GoalInterpretation
    proposed_config: CopilotConfiguration

    ml_recommender_rank: int = 1
    ml_recommender_framework: str
    ml_recommender_utility: Optional[float] = None

    rationale: str
    evidence_keys: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    config_diff_from_current: List[ConfigDiffItem] = Field(default_factory=list)

    requires_human_review: bool = True
    status: Literal["proposed"] = "proposed"


class ReviewDecision(BaseModel):
    proposal_id: str
    decision: Literal["approved", "approved_with_edits", "rejected"]
    note: Optional[str] = None

    final_config: Optional[CopilotConfiguration] = None
    config_diff: List[ConfigDiffItem] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    text: str
    source: str
    model: Optional[str] = None
    evidence_keys: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
