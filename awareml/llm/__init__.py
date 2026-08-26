# Backward-compatible Phase-0/legacy interfaces.
from .objective_parser import parse_objective_text
from .grounded_chat import GroundedChat, ollama_status

# Phase-7 Human-Centric LLM Copilot.
from .client import OllamaClient
from .copilot import CopilotService
from .evidence import (
    EvidenceBundle,
    build_after_evidence,
    build_before_evidence,
    build_during_evidence,
)
from .goal_parser import (
    GoalParser,
    deterministic_goal_parse,
)
from .grounded_copilot import GroundedCopilotChat
from .review import (
    ReviewStore,
    review_proposal,
)
from .schemas import (
    ConfigDiffItem,
    CopilotConfiguration,
    CopilotProposal,
    GoalInterpretation,
    GroundedAnswer,
    PrimaryObjectiveWeights,
    ReviewDecision,
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
    "ReviewStore",
    "review_proposal",
    "ConfigDiffItem",
    "CopilotConfiguration",
    "CopilotProposal",
    "GoalInterpretation",
    "GroundedAnswer",
    "PrimaryObjectiveWeights",
    "ReviewDecision",
]
