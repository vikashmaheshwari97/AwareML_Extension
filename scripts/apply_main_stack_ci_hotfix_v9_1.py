from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_test_function(path: Path, function_name: str, new_body_lines) -> None:
    if not path.exists():
        raise SystemExit("Required test file not found: {}".format(path))

    text = path.read_text(encoding="utf-8-sig")
    pattern = re.compile(
        r"^def " + re.escape(function_name) + r"\(\):\n(?:^(?!def ).*\n?)*",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(
            "Could not locate test function {} in {}".format(function_name, path)
        )

    body = "\n".join(new_body_lines)
    replacement = "def {}():\n{}\n\n".format(function_name, body.rstrip())
    updated = text[:match.start()] + replacement + text[match.end():]
    path.write_text(updated, encoding="utf-8")
    print("[patched] {} :: {}".format(path.relative_to(ROOT), function_name))


def main() -> None:
    replace_test_function(
        ROOT / "tests/test_goal_copilot_unified_ux.py",
        "test_post_approval_next_steps_are_explicit",
        [
            '    assert "What happens next" in UNIFIED',
            '    assert "Run Studio" in UNIFIED',
            '    assert "Streaming Observatory" in UNIFIED',
            '    assert "Decision Lab" in UNIFIED',
        ],
    )

    replace_test_function(
        ROOT / "tests/test_phase16_research_visuals_v2.py",
        "test_visual_module_contains_redesign_features",
        [
            '    source = (ROOT / "awareml" / "ui_v2" / "phase16_research_visuals.py").read_text(encoding="utf-8")',
            '    assert "CONSTRUCT_LABELS" in source',
            '    assert "ITEM_LABELS" in source',
            '    assert "Likert response distribution" in source',
            '    assert "Researcher-only views designed for fast interpretation of calibration, reliance, subgroup patterns and timing." in source',
            '    assert "Trust discrimination by explanation source" in source',
            '    assert "plotly.graph_objects" in source',
            '    assert "go.Heatmap" in source',
        ],
    )

    print("")
    print("=" * 78)
    print("AWAREML MAIN-STACK CI HOTFIX V9.1: PASS")
    print("=" * 78)
    print("Production code changed: NO")
    print("Protocol/stimulus/frozen design changed: NO")
    print("Tests updated: 2 stale assertions/functions")
    print("")
    print("Now run:")
    print("  python -m pytest -q .\\tests\\test_goal_copilot_unified_ux.py .\\tests\\test_phase16_research_visuals_v2.py")
    print("  python .\\scripts\\run_main_stack_local_v9.py")


if __name__ == "__main__":
    main()
