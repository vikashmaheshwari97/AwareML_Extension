# Backward-compatible Phase-0/legacy interfaces.
from .objective_parser import parse_objective_text
from .grounded_chat import GroundedChat, ollama_status

# Human-Centric LLM Copilot.
from .client import OllamaClient
from .copilot import CopilotService
from .evidence import (
    EvidenceBundle,
    build_after_evidence,
    build_before_evidence,
    build_during_evidence,
)
from .goal_parser import GoalParser, deterministic_goal_parse
from .grounded_copilot import GroundedCopilotChat
from .journal_client import (
    JournalLLMResponseError,
    JournalModelLockError,
    StrictJournalOllamaClient,
)
from .objective_selection import (
    GoalSelectionError,
    JournalObjectiveSelector,
    deterministic_objective_selection,
    infer_hcai_requirements,
)
from .objective_selection_v3 import (
    EvidenceGroundedObjectiveSelectorV3,
    SELECTOR_ID as OBJECTIVE_SELECTOR_V3_ID,
)
from .objective_selection_v31 import (
    HybridEvidenceGroundedObjectiveSelectorV31,
    SELECTOR_ID as OBJECTIVE_SELECTOR_V31_ID,
)
from .review import ReviewStore, review_proposal
from .schemas import (
    ConfigDiffItem,
    CopilotConfiguration,
    CopilotProposal,
    GoalInterpretation,
    GroundedAnswer,
    HCAIRequirements,
    ObjectiveSelectionResult,
    PrimaryObjectiveWeights,
    ReviewDecision,
)
from .weighting import (
    POLICY_ID as OBJECTIVE_WEIGHTING_POLICY_ID,
    WeightingPolicyError,
    equal_weights_for_selected,
)

__all__ = [
    "parse_objective_text",
    "GroundedChat",
    "ollama_status",
    "OllamaClient",
    "CopilotService",
    "EvidenceBundle",
    "build_after_evidence",
    "build_before_evidence",
    "build_during_evidence",
    "GoalParser",
    "deterministic_goal_parse",
    "GroundedCopilotChat",
    "StrictJournalOllamaClient",
    "JournalModelLockError",
    "JournalLLMResponseError",
    "JournalObjectiveSelector",
    "EvidenceGroundedObjectiveSelectorV3",
    "OBJECTIVE_SELECTOR_V3_ID",
    "HybridEvidenceGroundedObjectiveSelectorV31",
    "OBJECTIVE_SELECTOR_V31_ID",
    "GoalSelectionError",
    "deterministic_objective_selection",
    "infer_hcai_requirements",
    "ReviewStore",
    "review_proposal",
    "ConfigDiffItem",
    "CopilotConfiguration",
    "CopilotProposal",
    "GoalInterpretation",
    "GroundedAnswer",
    "HCAIRequirements",
    "ObjectiveSelectionResult",
    "PrimaryObjectiveWeights",
    "ReviewDecision",
    "OBJECTIVE_WEIGHTING_POLICY_ID",
    "WeightingPolicyError",
    "equal_weights_for_selected",
]
