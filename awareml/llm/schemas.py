from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class PrimaryObjectiveWeights(BaseModel):
    """The ML recommender keeps exactly four primary objectives."""

    accuracy: float = Field(0.55, ge=0)
    runtime: float = Field(0.15, ge=0)
    energy: float = Field(0.15, ge=0)
    co2: float = Field(0.15, ge=0)

    @model_validator(mode="after")
    def positive_total(self):
        if (
            self.accuracy
            + self.runtime
            + self.energy
            + self.co2
            <= 0
        ):
            raise ValueError(
                "At least one primary objective must be positive."
            )
        return self

    def normalized_dict(self) -> Dict[str, float]:
        values = {
            "accuracy": max(
                0.0,
                float(self.accuracy),
            ),
            "runtime": max(
                0.0,
                float(self.runtime),
            ),
            "energy": max(
                0.0,
                float(self.energy),
            ),
            "co2": max(
                0.0,
                float(self.co2),
            ),
        }
        total = float(sum(values.values()))
        return {
            key: value / total
            for key, value in values.items()
        }


class GoalInterpretation(BaseModel):
    task: Literal[
        "streaming_classification"
    ] = "streaming_classification"
    primary_weights: PrimaryObjectiveWeights = Field(
        default_factory=PrimaryObjectiveWeights
    )

    drift_sensitivity: Literal[
        "low",
        "moderate",
        "high",
    ] = "moderate"

    energy_constraint: Literal[
        "low",
        "moderate",
        "strict",
    ] = "moderate"

    fairness_required: bool = False
    fairness_metric: Optional[str] = None

    explainability_level: Literal[
        "low",
        "moderate",
        "high",
    ] = "moderate"

    max_runtime_sec: Optional[float] = Field(
        None,
        gt=0,
    )

    summary: str = (
        "Streaming classification with balanced "
        "performance and resource preferences."
    )
    uncertainties: List[str] = Field(
        default_factory=list
    )


class CopilotConfiguration(BaseModel):
    framework: str
    algorithm: str
    framework_parameters: Dict[str, Any] = Field(
        default_factory=dict
    )

    window_size: int = Field(
        1000,
        ge=50,
    )
    time_budget_sec: float = Field(
        60.0,
        gt=0,
    )

    drift: Dict[str, Any] = Field(
        default_factory=dict
    )
    fairness: Dict[str, Any] = Field(
        default_factory=dict
    )
    explainability: Dict[str, Any] = Field(
        default_factory=dict
    )
    sustainability: Dict[str, Any] = Field(
        default_factory=dict
    )

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
    evidence_keys: List[str] = Field(
        default_factory=list
    )
    warnings: List[str] = Field(
        default_factory=list
    )

    config_diff_from_current: List[
        ConfigDiffItem
    ] = Field(
        default_factory=list
    )

    requires_human_review: bool = True
    status: Literal[
        "proposed"
    ] = "proposed"


class ReviewDecision(BaseModel):
    proposal_id: str
    decision: Literal[
        "approved",
        "approved_with_edits",
        "rejected",
    ]
    note: Optional[str] = None

    final_config: Optional[
        CopilotConfiguration
    ] = None

    config_diff: List[
        ConfigDiffItem
    ] = Field(
        default_factory=list
    )


class GroundedAnswer(BaseModel):
    text: str
    source: str
    model: Optional[str] = None
    evidence_keys: List[str] = Field(
        default_factory=list
    )
    warnings: List[str] = Field(
        default_factory=list
    )
