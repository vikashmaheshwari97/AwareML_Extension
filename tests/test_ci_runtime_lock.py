from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _job_commands(job):
    return "\n".join(
        str(step.get("run") or "")
        for step in (job.get("steps") or [])
    )


def test_ci_jobs_use_exact_python_3810_container():
    jobs = _workflow()["jobs"]
    for name in ("main-stack", "oaml-river-compatibility"):
        assert jobs[name]["container"]["image"] == "python:3.8.10-buster"


def test_ci_does_not_use_setup_python():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions/setup-python" not in text


def test_ci_explicitly_verifies_patch_version():
    jobs = _workflow()["jobs"]
    for name in ("main-stack", "oaml-river-compatibility"):
        commands = _job_commands(jobs[name])
        assert "(3, 8, 10)" in commands


def test_main_and_oaml_dependency_contracts_remain_isolated():
    jobs = _workflow()["jobs"]
    main_commands = _job_commands(jobs["main-stack"])
    oaml_commands = _job_commands(jobs["oaml-river-compatibility"])

    # Avoid brittle string matching on Python quote style after YAML parsing.
    assert "river.__version__" in main_commands
    assert "0.10.1" in main_commands
    assert "xgboost.__version__" in main_commands
    assert "2.1.4" in main_commands

    assert "river==0.8.0" in oaml_commands
    assert "numpy==1.23.5" in oaml_commands
    assert "pandas==1.3.5" in oaml_commands
    assert "scipy==1.10.1" in oaml_commands
    assert "scikit-learn==1.1.3" in oaml_commands
