from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_visual_module_is_plotly_and_keeps_compatibility_exports():
    path = ROOT / "awareml/ui_v2/phase16_research_visuals.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert "plotly.graph_objects" in source
    assert "def render_trust_calibration_visuals" in source
    assert "def render_secondary_visuals" in source
    assert "render_secondary_questionnaire_visuals = render_secondary_visuals" in source
    assert "Trust discrimination by explanation source" in source
    assert "Diverging 100% Likert profile" in source


def test_command_centre_branding_has_no_visible_phase9_research_os_header():
    core = (ROOT / "awareml/ui_v2/pages_core.py").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    pages = (ROOT / "awareml/ui_v2/pages.py").read_text(encoding="utf-8")
    command = (ROOT / "awareml/ui_v2/command_center_upgrade.py").read_text(encoding="utf-8")
    ast.parse(core)
    ast.parse(command)

    assert "return render_command_center_upgrade()" in core
    assert "AwareML Dashboard" in command
    assert "COMMAND CENTRE" in command
    assert "PHASE 9 · RESEARCH UI V2" not in command
    assert "AwareML Research OS" not in command
    assert 'page_title="AwareML Dashboard"' in app
    assert '<div class="r9-brand-sub">Dashboard</div>' in app
    assert '"Command Centre": command_center_page' in pages


def test_participant_scale_wording_not_rewritten_and_terminology_is_clarified():
    source = (ROOT / "awareml/ui_v2/phase16_trust_calibration.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "validated trust statements below" in source
    assert "does not refer to the separate AwareML Copilot workspace" in source
    assert 'str(item["text"])' in source


def test_upgrade_does_not_modify_protocol_or_stimulus_files():
    # Contract test: installer contains no write target for protocol/stimulus-bank paths.
    installer = (ROOT / "scripts/apply_phase16_visual_commandcentre_v8.py").read_text(encoding="utf-8")
    assert "protocol.json" not in installer
    assert "trust_stimulus_bank_v1" not in installer
