from __future__ import annotations

import re


def classify_follow_up(text: str) -> str:
    t = text.lower().strip()
    if not t:
        return "empty"
    if any(p in t for p in ["source", "evidence", "show me", "where", "prove", "data"]):
        return "evidence_request"
    if any(p in t for p in ["why", "how", "explain", "reason"]):
        return "explanation_probe"
    if any(p in t for p in ["wrong", "disagree", "sure", "really", "challenge", "but"]):
        return "challenge"
    if any(p in t for p in ["what if", "instead", "compare", "versus", "vs"]):
        return "counterfactual_or_comparison"
    if any(p in t for p in ["mean", "clarify", "define", "what is"]):
        return "clarification"
    return "other_follow_up"


THINK_ALOUD_PROMPTS = [
    "Tell me what you are looking at before deciding whether to trust the recommendation.",
    "What information would you need before acting on this recommendation?",
    "What makes you ask a follow-up question - or decide not to ask one?",
    "Which visual or explanation detail changed your decision most?",
    "Was there a point where you felt you had enough information? Why?",
]
