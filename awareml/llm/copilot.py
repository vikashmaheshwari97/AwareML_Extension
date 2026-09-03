from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd

from awareml.recommender.v2_profile import profile_from_dataframe_v2
from awareml.recommender.v2_service import V2Recommender

from .config_diff import diff_configs
from .configuration import synthesize_configuration
from .evidence import (
    EvidenceBundle,
    build_after_evidence,
    build_before_evidence,
    build_during_evidence,
)
from .goal_parser import GoalParser
from .grounded_copilot import GroundedCopilotChat
from .review import ReviewStore, review_proposal
from .schemas import CopilotProposal, ReviewDecision


class CopilotService:
    """Human-centric layer around empirical Recommender V2.

    Phase 11 explicitly separates:
    A) scenario -> selected objective set
    B) selected objective set -> documented weights
    C) weights + dataset profile -> empirical framework ranking

    A and B are dataset-context free. C requires dataset meta-features.
    The LLM never directly chooses a framework.
    """

    def __init__(
        self,
        recommender: Optional[V2Recommender] = None,
        goal_parser: Optional[GoalParser] = None,
        chat: Optional[GroundedCopilotChat] = None,
        review_store: Optional[ReviewStore] = None,
    ):
        self.recommender = recommender
        self.goal_parser = goal_parser or GoalParser()
        self.chat = chat or GroundedCopilotChat()
        self.review_store = review_store or ReviewStore()

    def _get_recommender(self) -> V2Recommender:
        if self.recommender is None:
            self.recommender = V2Recommender()
        return self.recommender

    def interpret_goal(
        self,
        goal: str,
        use_llm: bool = False,
        allow_malformed_fallback: bool = True,
    ):
        """Interpret a scenario without any dataset context."""
        interpretation, parse_meta = self.goal_parser.parse(
            goal,
            use_llm=use_llm,
            allow_malformed_fallback=allow_malformed_fallback,
        )
        return interpretation, {
            "goal_parse": parse_meta,
            "objective_selection": {
                "status": interpretation.selection_status,
                "selected_objectives": list(interpretation.selected_objectives),
                "source": interpretation.selection_source,
                "model": interpretation.selection_model,
                "fallback_used": interpretation.fallback_used,
            },
            "weighting": {
                "policy_id": interpretation.weighting_policy,
                "weights": interpretation.primary_weights.normalized_dict(),
            },
            "dataset_context_used": False,
            "framework_ranking_generated": False,
        }

    def propose_from_profile(
        self,
        goal: str,
        profile: Mapping[str, Any],
        sensitive_attribute: Optional[str] = None,
        current_config: Optional[Mapping[str, Any]] = None,
        use_llm: bool = False,
        coverage: float = 0.90,
    ):
        interpretation, parse_meta = self.goal_parser.parse(
            goal,
            use_llm=use_llm,
        )
        weights = interpretation.primary_weights.normalized_dict()

        ranked, ranking_meta = self._get_recommender().recommend_profile(
            dict(profile),
            weights=weights,
            ranking_mode="point",
            coverage=coverage,
        )

        config, config_warnings = synthesize_configuration(
            interpretation,
            ranked,
            sensitive_attribute=sensitive_attribute,
        )

        evidence = build_before_evidence(
            interpretation,
            dict(profile),
            ranked,
            ranking_meta,
        )

        rationale_answer = self.chat.answer(
            "Why is this framework recommended for the interpreted goal?",
            evidence,
            use_llm=use_llm,
        )

        warnings = []
        warnings.extend(parse_meta.get("warnings", []) or [])
        warnings.extend(config_warnings)
        warnings.extend(ranking_meta.get("warnings", []) or [])
        warnings.extend(rationale_answer.warnings)

        proposal = CopilotProposal(
            proposal_id=str(uuid.uuid4()),
            goal=goal,
            interpretation=interpretation,
            proposed_config=config,
            ml_recommender_rank=int(ranked.iloc[0]["rank"]),
            ml_recommender_framework=str(ranked.iloc[0]["framework"]),
            ml_recommender_utility=(
                float(ranked.iloc[0]["utility"])
                if pd.notna(ranked.iloc[0]["utility"])
                else None
            ),
            rationale=rationale_answer.text,
            evidence_keys=rationale_answer.evidence_keys,
            warnings=warnings,
            config_diff_from_current=(
                diff_configs(current_config or {}, config)
                if current_config
                else []
            ),
        )

        return proposal, ranked, evidence, {
            "goal_parse": parse_meta,
            "objective_selection": {
                "status": interpretation.selection_status,
                "selected_objectives": list(interpretation.selected_objectives),
                "source": interpretation.selection_source,
                "model": interpretation.selection_model,
                "fallback_used": interpretation.fallback_used,
            },
            "weighting": {
                "policy_id": interpretation.weighting_policy,
                "weights": weights,
            },
            "ranking": ranking_meta,
            "rationale_source": rationale_answer.source,
            "rationale_model": rationale_answer.model,
            "dataset_context_used": True,
            "framework_ranking_generated": True,
        }

    def propose_from_dataframe(
        self,
        goal: str,
        df: pd.DataFrame,
        target: str,
        sensitive_attribute: Optional[str] = None,
        current_config: Optional[Mapping[str, Any]] = None,
        use_llm: bool = False,
        window_size: int = 1000,
        time_budget_sec: float = 60.0,
        dataset_family: str = "unknown",
        source_type: str = "unknown",
        drift_type: str = "unknown",
        coverage: float = 0.90,
    ):
        profile = profile_from_dataframe_v2(
            df,
            target=target,
            window_size=window_size,
            time_budget_sec=time_budget_sec,
            dataset_family=dataset_family,
            source_type=source_type,
            drift_type=drift_type,
        )
        return self.propose_from_profile(
            goal,
            profile,
            sensitive_attribute=sensitive_attribute,
            current_config=current_config,
            use_llm=use_llm,
            coverage=coverage,
        )

    def review(
        self,
        proposal: CopilotProposal,
        decision: str,
        edits: Optional[Dict[str, Any]] = None,
        note: Optional[str] = None,
        persist: bool = True,
    ) -> ReviewDecision:
        review = review_proposal(
            proposal,
            decision=decision,
            edits=edits,
            note=note,
        )
        if persist:
            self.review_store.append(proposal, review)
        return review

    def during_evidence(self, result: Any) -> EvidenceBundle:
        return build_during_evidence(result)

    def after_evidence(
        self,
        results: Iterable[Any],
        ranking=None,
    ) -> EvidenceBundle:
        return build_after_evidence(results, ranking=ranking)

    def ask(
        self,
        question: str,
        evidence: EvidenceBundle,
        use_llm: bool = False,
    ):
        return self.chat.answer(question, evidence, use_llm=use_llm)

    def what_if_weights(
        self,
        profile: Mapping[str, Any],
        weights: Mapping[str, float],
        ranking_mode: str = "point",
        coverage: float = 0.90,
    ):
        return self._get_recommender().recommend_profile(
            dict(profile),
            weights=weights,
            ranking_mode=ranking_mode,
            coverage=coverage,
        )
