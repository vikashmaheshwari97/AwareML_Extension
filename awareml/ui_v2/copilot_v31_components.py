from __future__ import annotations

from typing import Any, Dict

import streamlit as st


COPILOT_RESULT_KEYS = (
    "copilot_proposal",
    "copilot_ranked",
    "copilot_evidence",
    "copilot_meta",
    "copilot_review",
    "copilot_context_free_interpretation",
    "copilot_context_free_meta",
    "copilot_objective_review",
    "copilot_objective_review_note",
)


def clear_previous_copilot_result(state: Dict[str, Any]) -> None:
    """Clear scenario-dependent result state before evaluating a new scenario.

    This prevents a failed/abstained new request from displaying a previous
    scenario's objective set, weights, or framework recommendation.
    """
    for key in COPILOT_RESULT_KEYS:
        state.pop(key, None)
    state.pop("copilot_clarification", None)


def set_copilot_clarification(
    state: Dict[str, Any],
    *,
    message: str,
    detail: str = "",
    error_type: str = "selection_abstention",
) -> None:
    state["copilot_clarification"] = {
        "message": message,
        "detail": detail,
        "error_type": error_type,
    }


def render_copilot_clarification(state: Dict[str, Any]) -> bool:
    item = state.get("copilot_clarification")
    if not isinstance(item, dict):
        return False

    st.markdown(
        """
        <div class="r9-callout" style="border-left:4px solid #d97706; padding:14px 16px; margin:10px 0 16px 0;">
          <b>Objective clarification required</b><br>
          AwareML did not generate a new objective weighting or framework recommendation for this scenario.
          The previous scenario result has been cleared so it cannot be mistaken for the current request.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(str(item.get("message") or "Please clarify the deployment priorities."))
    if item.get("detail"):
        with st.expander("Why AwareML abstained", expanded=False):
            st.write(str(item.get("detail")))

    st.markdown("**Clarification cues you can add to the scenario**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.caption("🎯 **Prediction quality**\n\nDependable, correct, reliable decisions")
    with c2:
        st.caption("⚡ **Fast response**\n\nPrompt, immediate, low-delay operation")
    with c3:
        st.caption("🔋 **Battery / power**\n\nBattery endurance, charging or power budget")
    with c4:
        st.caption("🌱 **Environmental impact**\n\nFootprint, emissions or low-impact computing")
    st.caption(
        "These cues are examples for clarification, not automatic objective labels. "
        "Human review remains required."
    )
    return True


def v31_audit_rows(evidence_audit: Dict[str, Any]):
    decisions = dict((evidence_audit or {}).get("decisions") or {})
    rows = []
    for objective in ["Accuracy", "Runtime", "Energy", "CO2"]:
        row = dict(decisions.get(objective) or {})
        phrase = row.get("evidence_returned") or row.get("semantic_support")
        rows.append(
            {
                "Objective": objective,
                "LLM proposed": bool(row.get("llm_selected")),
                "Evidence phrase found": bool(row.get("evidence_phrase_found")),
                "Objective-specific support": bool(row.get("objective_specific_support")),
                "Accepted by V3.1": bool(row.get("accepted")),
                "Accepted by": row.get("accepted_by") or "none",
                "Confidence": row.get("confidence") or "N/A",
                "Scenario evidence": phrase or "N/A",
                "Decision reason": row.get("reason") or "N/A",
            }
        )
    return rows
