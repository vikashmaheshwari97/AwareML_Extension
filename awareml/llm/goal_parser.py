from __future__ import annotations

from typing import Optional, Tuple

from .journal_client import JournalModelLockError
from .objective_selection import (
    GoalSelectionError,
    JournalObjectiveSelector,
    deterministic_objective_selection,
    infer_hcai_requirements,
)
from .schemas import GoalInterpretation, ObjectiveSelectionResult
from .weighting import POLICY_ID, equal_weights_for_selected


def _summary(selection: ObjectiveSelectionResult, hcai) -> str:
    objectives = ", ".join(selection.selected_objectives) or "none"
    parts = [
        "streaming classification",
        "selected objectives: {}".format(objectives),
        "{} drift sensitivity".format(hcai.drift_sensitivity),
        "{} explainability".format(hcai.explainability_level),
    ]
    if hcai.fairness_required:
        parts.append("fairness required")
    if hcai.energy_constraint != "low":
        parts.append("{} energy constraint".format(hcai.energy_constraint))
    return "; ".join(parts) + "."


def _interpretation_from_selection(
    text: str,
    selection: ObjectiveSelectionResult,
) -> GoalInterpretation:
    if not selection.selected_objectives:
        raise GoalSelectionError(
            "Objective selection status '{}' produced no usable objectives. "
            "Ask the user to clarify the scenario.".format(selection.status)
        )

    weights, weighting_meta = equal_weights_for_selected(
        selection.selected_objectives
    )
    hcai = infer_hcai_requirements(text)

    uncertainties = list(selection.uncertainties)
    if selection.status != "valid":
        uncertainties.append(
            "Objective selection status is '{}'; human review is required.".format(
                selection.status
            )
        )
    if selection.fallback_used:
        uncertainties.append(
            "A deterministic fallback supplied the objective subset after an LLM parse failure."
        )

    return GoalInterpretation(
        selection_status=selection.status,
        selected_objectives=selection.selected_objectives,
        selection_source=selection.source,
        selection_model=selection.model,
        fallback_used=selection.fallback_used,
        weighting_policy=weighting_meta["policy_id"],
        primary_weights=weights,
        hcai_requirements=hcai,
        summary=_summary(selection, hcai),
        uncertainties=uncertainties,
    )


def deterministic_goal_parse(text: str) -> GoalInterpretation:
    """Backward-compatible deterministic Copilot parse using V2 semantics."""

    selection = deterministic_objective_selection(text)
    if not selection.selected_objectives:
        raise GoalSelectionError(
            "Deterministic objective selector could not infer a usable objective subset: {}.".format(
                selection.status
            )
        )
    return _interpretation_from_selection(text, selection)


class GoalParser:
    """Roadmap-aligned Human goal interpreter.

    Problem A:
        natural-language scenario -> selected objective set

    Problem B:
        selected objective set -> equal_selected_v1 weights

    The empirical ML recommender remains a separate component.
    """

    def __init__(
        self,
        client=None,
        selector: Optional[JournalObjectiveSelector] = None,
    ):
        # `client` is retained for source compatibility with Phase-7 callers,
        # but journal objective selection is deliberately isolated from the
        # generic fallback-capable OllamaClient.
        self.client = client
        self.selector = selector

    def _selector(self) -> JournalObjectiveSelector:
        if self.selector is None:
            self.selector = JournalObjectiveSelector()
        return self.selector

    def parse(
        self,
        text: str,
        use_llm: bool = False,
        allow_malformed_fallback: bool = True,
    ) -> Tuple[GoalInterpretation, dict]:
        if not text or not text.strip():
            raise GoalSelectionError(
                "No natural-language scenario was provided. "
                "Objective selection cannot proceed silently."
            )

        if not use_llm:
            selection = deterministic_objective_selection(text)
            if not selection.selected_objectives:
                raise GoalSelectionError(
                    "Deterministic objective selection returned '{}'. "
                    "Ask the user to clarify the scenario.".format(selection.status)
                )
            interpretation = _interpretation_from_selection(text, selection)
            return interpretation, {
                "source": "deterministic",
                "model": None,
                "selection_status": selection.status,
                "selected_objectives": list(selection.selected_objectives),
                "weighting_policy": POLICY_ID,
                "weights": interpretation.primary_weights.normalized_dict(),
                "fallback_used": False,
                "warnings": list(selection.uncertainties),
            }

        # Wrong model / wrong digest / wrong Ollama version intentionally
        # propagates as JournalModelLockError. Journal evaluation must fail.
        selection = self._selector().select(text)

        if selection.status == "malformed":
            if not allow_malformed_fallback:
                raise GoalSelectionError(
                    "Journal LLM returned malformed objective-selection JSON."
                )

            fallback = deterministic_objective_selection(text)
            if not fallback.selected_objectives:
                raise GoalSelectionError(
                    "Journal LLM output was malformed and deterministic fallback "
                    "could not infer a usable objective subset."
                )

            fallback = ObjectiveSelectionResult(
                status="malformed",
                selected_objectives=fallback.selected_objectives,
                uncertainties=(
                    list(selection.uncertainties)
                    + [
                        "Explicit deterministic fallback used after malformed LLM output."
                    ]
                ),
                source="deterministic-fallback-after-malformed-journal-llm",
                model=selection.model,
                fallback_used=True,
            )
            interpretation = _interpretation_from_selection(text, fallback)
            return interpretation, {
                "source": fallback.source,
                "model": fallback.model,
                "selection_status": "malformed",
                "selected_objectives": list(fallback.selected_objectives),
                "weighting_policy": POLICY_ID,
                "weights": interpretation.primary_weights.normalized_dict(),
                "fallback_used": True,
                "warnings": list(fallback.uncertainties),
            }

        if selection.status in {"out_of_scope", "contradictory"}:
            raise GoalSelectionError(
                "Journal objective selection returned '{}': {}".format(
                    selection.status,
                    "; ".join(selection.uncertainties) or "no additional detail",
                )
            )

        if selection.status == "ambiguous" and not selection.selected_objectives:
            raise GoalSelectionError(
                "Journal objective selection is ambiguous and contains no usable objective subset."
            )

        interpretation = _interpretation_from_selection(text, selection)
        warnings = list(selection.uncertainties)
        if selection.status == "ambiguous":
            warnings.append(
                "LLM objective selection is ambiguous; explicit human review is required."
            )

        return interpretation, {
            "source": selection.source,
            "model": selection.model,
            "selection_status": selection.status,
            "selected_objectives": list(selection.selected_objectives),
            "weighting_policy": POLICY_ID,
            "weights": interpretation.primary_weights.normalized_dict(),
            "fallback_used": False,
            "warnings": warnings,
        }
