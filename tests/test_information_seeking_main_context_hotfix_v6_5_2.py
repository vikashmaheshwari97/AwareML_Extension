from pathlib import Path


UI = Path("awareml/ui_v2/phase17_information_seeking.py")
CONTEXT = Path("awareml/studies/information_seeking_context.py")


def test_main_context_hotfix_markers_present():
    text = UI.read_text(encoding="utf-8")
    assert "AWAREML_PHASE17_MAIN_CONTEXT_HOTFIX_V6_5_2" in text
    assert "Initialize standardized Main Study context from current run" in text
    assert (
        "Main Study participant access is waiting for one standardized measured context"
        in text
    )
    assert "Open Researcher Workspace to initialize Main Study context" in text
    assert "five_framework_ready" in text


def test_stale_pilot_instruction_removed_from_main_context_error():
    text = UI.read_text(encoding="utf-8")
    assert (
        "save the measured run as a standardized Pilot context so later participants "
        "do not need to rerun AwareML"
    ) not in text


def test_v651_standardized_context_resolver_is_present():
    text = CONTEXT.read_text(encoding="utf-8")
    assert "Main Study deliberately prefers the saved standardized context" in text
    assert '"missing_final_context"' in text


def test_patched_ui_compiles():
    text = UI.read_text(encoding="utf-8")
    compile(text, str(UI), "exec")
