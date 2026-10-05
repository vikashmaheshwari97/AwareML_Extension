from pathlib import Path
import ast
import re

ROOT = Path(__file__).resolve().parents[1]
COPILOT = ROOT / "awareml" / "ui_v2" / "copilot_three_path.py"
VALIDATION = ROOT / "awareml" / "ui_v2" / "phase18_validation_evidence.py"


def test_ui_files_parse():
    ast.parse(COPILOT.read_text(encoding="utf-8"))
    ast.parse(VALIDATION.read_text(encoding="utf-8"))


def test_no_stale_embedded_true_phase18_calls():
    source = COPILOT.read_text(encoding="utf-8")
    calls = re.findall(r"render_phase18_heldout_validation\(([^)]*)\)", source, flags=re.S)
    assert calls
    assert all(re.search(r"\bembedded\s*=\s*True\b", args) is None for args in calls)


def test_research_panel_is_collapsed_and_heldout_is_collapsible():
    source = COPILOT.read_text(encoding="utf-8")
    assert 'with st.expander("Research validation details", expanded=False):' in source
    assert "collapsible=True" in source


def test_heldout_renderer_contains_complete_evidence_sections():
    source = VALIDATION.read_text(encoding="utf-8")
    for token in [
        "collapsible: bool = False",
        "31-dataset held-out validation · supplied evidence",
        "Held-out datasets",
        "Preference cases",
        "Top-1 oracle agreement",
        "Top-3 oracle coverage",
        "Mean normalized regret",
        "Mean Spearman",
        "Objective-level winner agreement",
        "Preference-case results by evaluation cohort",
        "Per-dataset summary (31 datasets)",
        "Evidence provenance",
        "render_body(show_heading=False)",
    ]:
        assert token in source


def test_no_phase18_call_is_nested_inside_research_expander():
    source = COPILOT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    def is_research_with(node):
        if not isinstance(node, ast.With):
            return False
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "expander"
                and isinstance(func.value, ast.Name)
                and func.value.id == "st"
            ):
                continue
            if call.args and isinstance(call.args[0], ast.Constant):
                if call.args[0].value == "Research validation details":
                    return True
        return False

    for node in ast.walk(tree):
        if is_research_with(node):
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Name)
                        and sub.func.id == "render_phase18_heldout_validation"
                    ):
                        raise AssertionError(
                            "Phase-18 held-out renderer is nested inside Research validation details"
                        )
