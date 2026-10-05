from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_heldout_renderer_has_full_collapsible_mode():
    source = (
        ROOT / "awareml" / "ui_v2" / "phase18_validation_evidence.py"
    ).read_text(encoding="utf-8")
    assert "collapsible: bool = False" in source
    assert '"31-dataset held-out validation · supplied evidence"' in source
    assert "render_body(show_heading=False)" in source


def test_dataset_aware_heldout_panel_is_sibling_not_embedded():
    source = (
        ROOT / "awareml" / "ui_v2" / "copilot_three_path.py"
    ).read_text(encoding="utf-8")
    assert "collapsible=True" in source
    assert "render_phase18_heldout_validation(expanded=False, embedded=True)" not in source


def test_all_heldout_details_remain_available():
    source = (
        ROOT / "awareml" / "ui_v2" / "phase18_validation_evidence.py"
    ).read_text(encoding="utf-8")
    for label in [
        "Held-out datasets",
        "Preference cases",
        "Top-1 oracle agreement",
        "Top-3 oracle coverage",
        "Objective-level winner agreement",
        "Preference-case results by evaluation cohort",
        "Per-dataset summary (31 datasets)",
        "Evidence provenance",
    ]:
        assert label in source


def test_hotfixed_files_parse():
    for rel in [
        "awareml/ui_v2/phase18_validation_evidence.py",
        "awareml/ui_v2/copilot_three_path.py",
    ]:
        ast.parse((ROOT / rel).read_text(encoding="utf-8"))
