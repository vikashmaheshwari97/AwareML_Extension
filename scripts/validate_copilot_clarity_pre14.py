from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
helper = (ROOT / "awareml/ui_v2/pre14_usability.py").read_text(encoding="utf-8")
page = (ROOT / "awareml/ui_v2/pages_copilot.py").read_text(encoding="utf-8")

checks = {
    "no_nested_research_expander": 'with st.expander("Research evidence & diagnostics"' not in helper,
    "research_view_direct_renderer": "research_renderer(interpretation, parse_meta, state)" in helper,
    "simple_flow_numbered": "## 3 · Copilot AutoML Plan" in helper and "### 5 · Final plan decision" in helper,
    "plain_framework_explanation": "Why this framework fits your priorities" in helper,
    "llm_not_framework_chooser": "The LLM did not choose" in helper,
    "technical_ids_removed_from_simple_prose": "re.sub(r\"\\\\s*\\\\[evidence" in helper or "Evidence IDs are useful for Research View" in helper,
    "fairness_constraint_semantics": "Fairness constraint" in helper and "Not requested by this goal" in helper,
    "fairness_audit_semantics": "This does not disable post-run fairness analysis" in helper,
    "utility_not_confidence": "not a probability, correctness score or model confidence" in helper,
    "simple_final_review_persistent": "ReviewStore(ROOT / \"artifacts\" / \"copilot\" / \"reviews.jsonl\")" in helper,
    "simple_view_hides_advanced_detail": 'if state.get("copilot_view_mode") == "Simple View":\n        return' in page,
    "research_view_preserved": "_PRE14_RESEARCH_OBJECTIVE_RENDERER" in page,
}

print("=" * 88)
print("AwareML Copilot clarity pre-Phase-14 validation")
print("=" * 88)
failed = []
for name, ok in checks.items():
    print("{:<52} {}".format(name, "PASS" if ok else "FAIL"))
    if not ok:
        failed.append(name)

if failed:
    print("\nFAILED checks:", ", ".join(failed))
    raise SystemExit(1)
print("\nCopilot clarity pre-Phase-14 validation: PASS")
print("=" * 88)
