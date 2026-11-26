"""A source checkout must not require local demonstration records."""
import pytest

from awareml.ui_v2 import illustrative_study_results as demo


@pytest.mark.parametrize("renderer", ["render_trust_demo", "render_information_demo"])
def test_missing_local_demo_keeps_study_workspace_usable(tmp_path, monkeypatch, renderer):
    monkeypatch.setattr(demo, "DATA_PATH", tmp_path / "not-installed" / "awareml_34_cases.json")
    monkeypatch.setattr(demo, "_researcher_access", lambda: True)
    messages = []
    monkeypatch.setattr(demo.st, "info", messages.append)

    def unexpected_error(message):
        pytest.fail("Missing optional data must not produce an application error: " + str(message))

    monkeypatch.setattr(demo.st, "error", unexpected_error)
    getattr(demo, renderer)()
    assert len(messages) == 1
    assert "study workspace remains available" in messages[0]
