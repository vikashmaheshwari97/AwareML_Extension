from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional


CATEGORY_LABELS = {
    "evidence_request": "Evidence request",
    "explanation_probe": "Explanation probe",
    "challenge": "Challenge",
    "counterfactual_or_comparison": "Counterfactual / comparison",
    "clarification": "Clarification",
    "other_follow_up": "Other follow-up",
    "empty": "Empty",
}

CATEGORY_DESCRIPTIONS = {
    "evidence_request": "Requests data, measurements, provenance, source evidence, or proof.",
    "explanation_probe": "Asks why or how the recommendation/explanation was produced.",
    "challenge": "Questions, disputes, or expresses skepticism about the recommendation.",
    "counterfactual_or_comparison": "Compares alternatives or asks what would happen under another choice.",
    "clarification": "Asks for a definition, meaning, or simpler explanation.",
    "other_follow_up": "A follow-up that does not match the five pre-defined categories.",
}

THINK_ALOUD_PROMPTS = [
    "Tell me what you are looking at before deciding whether to trust the recommendation.",
    "What information would you need before acting on this recommendation?",
    "What makes you ask a follow-up question - or decide not to ask one?",
    "Which visual or explanation detail changed your decision most?",
    "Was there a point where you felt you had enough information? Why?",
]


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def classify_follow_up(text: str) -> str:
    """Lightweight behavior classifier used as an analysis aid.

    The classifier is intentionally transparent and deterministic. Its output is
    never treated as a replacement for manual qualitative coding.
    """
    t = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if not t:
        return "empty"

    # Put explicit challenge phrases before generic why/how phrases so questions
    # such as "why are you sure?" are not automatically reduced to an explanation probe.
    if _contains_any(
        t,
        [
            "wrong",
            "disagree",
            "are you sure",
            "really",
            "challenge",
            "i doubt",
            "not convinced",
            "but why",
            "why should i trust",
        ],
    ):
        return "challenge"

    if _contains_any(
        t,
        [
            "what if",
            "instead",
            "compare",
            "comparison",
            "versus",
            " vs ",
            "alternative",
            "another framework",
            "second best",
        ],
    ):
        return "counterfactual_or_comparison"

    # Clarification takes precedence over generic metric/data keywords.
    if _contains_any(t, ["what does", "mean", "clarify", "define", "what is", "simpler terms"]):
        return "clarification"

    if _contains_any(
        t,
        [
            "source",
            "evidence",
            "show me",
            "where did",
            "prove",
            "data",
            "measurement",
            "metric",
            "numbers",
        ],
    ):
        return "evidence_request"

    if _contains_any(t, ["why", "how", "explain", "reason"]):
        return "explanation_probe"

    return "other_follow_up"


def derive_representative_pattern(
    follow_up_count: int,
    first_category: Optional[str],
    evidence_viewed: Optional[Iterable[str]],
    recommendation_decision: Optional[str],
) -> str:
    """Return a compact descriptive behavior pattern.

    This is a structural summary of logged actions, not a qualitative theme.
    """
    count = int(follow_up_count or 0)
    evidence = list(evidence_viewed or [])
    decision = str(recommendation_decision or "").strip().lower()
    first = str(first_category or "").strip()

    if count == 0:
        if decision == "accept":
            return "Accept without follow-up"
        if decision in {"override", "reject"}:
            return "Decline without follow-up"
        return "No follow-up recorded"

    if first == "challenge" and decision in {"override", "reject"}:
        return "Challenge then decline"
    if first == "counterfactual_or_comparison":
        return "Compare before deciding"
    if first == "evidence_request" or evidence:
        if decision == "accept":
            return "Verify evidence then accept"
        if decision in {"override", "reject"}:
            return "Verify evidence then decline"
        return "Evidence-led investigation"
    if count >= 3:
        return "Multi-step investigation"
    if first == "clarification":
        return "Clarify before deciding"
    if first == "explanation_probe":
        return "Probe explanation before deciding"
    return "Other follow-up pattern"
