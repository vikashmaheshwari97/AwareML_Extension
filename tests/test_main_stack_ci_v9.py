from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_hotfix_scripts_parse():
    for rel in [
        "scripts/run_main_stack_local_v9.py",
    ]:
        ast.parse((ROOT / rel).read_text(encoding="utf-8"))


def test_local_ci_runner_does_not_compile_dot_venv():
    source = (ROOT / "scripts/run_main_stack_local_v9.py").read_text(encoding="utf-8")
    assert 'ROOT / ".venv"' not in source
    assert '"awareml"' in source
    assert '"tests"' in source
