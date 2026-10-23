from pathlib import Path
import json
import hashlib

import pytest
import yaml

from awareml.resource_paths import resolve_text_resource

ROOT = Path(__file__).resolve().parents[1]


def test_original_explicit_resource_has_priority(tmp_path):
    original = tmp_path / "instructions.txt"
    original.write_text("explicit local resource")
    original.with_suffix(".prompt").write_text("tracked alternative")
    assert resolve_text_resource(original) == original


@pytest.mark.parametrize("extension", [".prompt", ".ref"])
def test_relocated_resource_and_missing_resource(tmp_path, extension):
    original = tmp_path / "resource.txt"
    assert resolve_text_resource(original) == original
    with pytest.raises(FileNotFoundError):
        resolve_text_resource(original).read_text()
    tracked = original.with_suffix(extension)
    tracked.write_text("preserved content")
    assert resolve_text_resource(original).read_text() == "preserved content"


def test_original_phase10_prompt_hash_is_preserved():
    protocol = json.loads((ROOT / "data/journal/protocol_v1/journal_experimental_protocol_v1.json").read_text())
    llm = protocol["journal_llm"]
    prompt = resolve_text_resource(ROOT / llm["prompt_file"])
    assert hashlib.sha256(prompt.read_bytes()).hexdigest() == llm["prompt_sha256"]


def test_ci_installs_v3_requirements_and_keeps_full_suite():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["main-stack"]["steps"]
    install = next(s["run"] for s in steps if s.get("name") == "Install main stack")
    assert "-r requirements-protocol-v3.in" in install
    test_step = next(s for s in steps if s.get("name") == "Tests")
    assert test_step["run"] == "pytest -q"
    assert not test_step.get("continue-on-error", False)


def test_legacy_installer_rejects_path_outside_project(tmp_path):
    from scripts.apply_multilevel_xai_upgrade import apply
    package, project = tmp_path / "package", tmp_path / "project"
    (package / "payload").mkdir(parents=True)
    (project / "awareml").mkdir(parents=True)
    (project / "app.py").write_text("original")
    (package / "UPGRADE_MANIFEST.json").write_text(json.dumps({"files": [
        {"path": "../escape.py", "new_sha256": "invalid"}
    ]}))
    with pytest.raises(ValueError, match="Unsafe upgrade path"):
        apply(package, project, allow_local_changes=True)
    assert (project / "app.py").read_text() == "original"
    assert not (project / ".multilevel_xai_backups").exists()
