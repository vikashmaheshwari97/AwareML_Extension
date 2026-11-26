from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "tests/test_copilot_three_path_pre14.py",
    "tests/test_copilot_workspace_v5.py",
    "tests/test_goal_copilot_unified_ux.py",
    "tests/test_phase12_v31_final_pre14.py",
    "tests/test_phase15_allstages_live_runner.py",
    "tests/test_phase15_simplebatch_final_fix.py",
    "tests/test_phase15_stageb_generation_prompt.py",
    "tests/test_phase15_ui.py",
    "tests/test_pre_phase14.py",
}

proc = subprocess.run(
    ["git", "diff", "--name-only"],
    cwd=str(ROOT),
    text=True,
    capture_output=True,
    check=True,
)
changed = {
    line.strip().replace("\\", "/")
    for line in proc.stdout.splitlines()
    if line.strip()
}
unexpected = sorted(path for path in changed if path not in EXPECTED)
if unexpected:
    raise SystemExit(
        "FAIL: non-test tracked files are modified: {}".format(", ".join(unexpected))
    )
print("PASS: tracked diff is test-only")

checks = {
    "tests/test_copilot_three_path_pre14.py": ["That path uses learned models"],
    "tests/test_copilot_workspace_v5.py": ["Dataset-aware ML Recommender"],
    "tests/test_goal_copilot_unified_ux.py": [
        "active_path = render_copilot_workspace_header(state)",
        "Human-centred AI oversight",
    ],
    "tests/test_phase12_v31_final_pre14.py": [
        "EvidenceGroundedObjectiveSelectorV33",
        "copilot_v33_components.py",
    ],
    "tests/test_phase15_allstages_live_runner.py": [
        "Run complete Explanation integrity live check",
    ],
    "tests/test_phase15_simplebatch_final_fix.py": [
        "phase15_live_guided_prompt_v4",
        "Run complete Explanation integrity live check",
    ],
    "tests/test_phase15_stageb_generation_prompt.py": [
        "phase15_live_guided_prompt_v4",
    ],
    "tests/test_phase15_ui.py": [
        "Explanation Integrity Lab",
        "pages_explanation_integrity.py",
    ],
    "tests/test_pre_phase14.py": [
        "Objective-selection evidence status",
        "Hybrid evidence-grounded V3.3 selection audit",
    ],
}
for rel, markers in checks.items():
    text = (ROOT / rel).read_text(encoding="utf-8")
    for marker in markers:
        if marker not in text:
            raise SystemExit("FAIL: {} missing {}".format(rel, marker))
print("PASS: current-contract test markers are present")

cmd = [sys.executable, "-m", "pytest", "-q"] + sorted(EXPECTED)
print("RUN:", " ".join(cmd))
result = subprocess.run(cmd, cwd=str(ROOT))
if result.returncode != 0:
    raise SystemExit(result.returncode)

print()
print("PASS: all updated CI-contract test modules pass")
print("NEXT: run full suite with: pytest -q")
