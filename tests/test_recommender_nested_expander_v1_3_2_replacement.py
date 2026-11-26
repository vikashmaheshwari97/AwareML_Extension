from pathlib import Path
import ast
import re

ROOT = Path(__file__).resolve().parents[1]
COPILOT = ROOT / "awareml" / "ui_v2" / "copilot_three_path.py"
VALIDATION = ROOT / "awareml" / "ui_v2" / "phase18_validation_evidence.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_phase18_renderer_supports_collapsible_sibling_mode():
    source = _source(VALIDATION)
    assert "collapsible: bool = False" in source
    assert "31-dataset held-out validation · supplied evidence" in source
    assert "render_body(show_heading=False)" in source


def test_dataset_aware_calls_phase18_renderer_in_collapsible_mode():
    source = _source(COPILOT)
    assert "collapsible=True" in source
    calls = re.findall(r"render_phase18_heldout_validation\(([^)]*)\)", source, flags=re.S)
    assert calls, "No Phase-18 renderer call found"
    assert all(re.search(r"\bembedded\s*=\s*True\b", args) is None for args in calls)


def test_research_validation_and_heldout_are_sibling_panels():
    source = _source(COPILOT)
    assert 'with st.expander("Research validation details", expanded=False):' in source

    tree = ast.parse(source)

    def is_research_expander(node):
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
        if is_research_expander(node):
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Name)
                        and sub.func.id == "render_phase18_heldout_validation"
                    ):
                        raise AssertionError(
                            "Phase-18 held-out renderer must be a sibling, not nested inside Research validation details"
                        )


def test_hotfixed_files_parse():
    ast.parse(_source(COPILOT))
    ast.parse(_source(VALIDATION))
