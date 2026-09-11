from __future__ import annotations

import argparse
import hashlib
import py_compile
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PAYLOAD_DIR = PACKAGE_DIR / "trust_access_payload"
BACKUP_DIR_NAME = ".trust_calibration_simple_access_backup"

CORE_GUARDS = [
    Path("awareml/studies/trust.py"),
    Path("awareml/studies/trust_analysis.py"),
    Path("awareml/studies/trust_freeze.py"),
    Path("data/journal/trust_calibration_phase16_v1/design/protocol.json"),
]

PAYLOAD_FILES = [
    Path("awareml/ui_v2/phase16_trust_calibration.py"),
    Path("phase16_participant_app.py"),
    Path("START_AWAREML.ps1"),
    Path("START_PARTICIPANT_ONLY.ps1"),
    Path("docs/TRUST_CALIBRATION_ACCESS.md"),
    Path("tests/test_trust_calibration_access_ui.py"),
]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def find_repo(explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(PACKAGE_DIR.resolve())
    candidates.extend(list(PACKAGE_DIR.resolve().parents)[:4])

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        if (resolved / "app.py").exists() and (resolved / "awareml").is_dir():
            return resolved
    raise SystemExit("Could not locate AwareML_Extension. Run the installer from the repository root.")


def backup_file(repo, backup_root, rel):
    src = repo / rel
    if not src.exists():
        return
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def patch_gitignore(repo, backup_root):
    path = repo / ".gitignore"
    if not path.exists():
        return False
    marker = "# Trust Calibration simple-access local artifacts"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    backup_file(repo, backup_root, Path(".gitignore"))
    addition = """

# Trust Calibration simple-access local artifacts
/.trust_calibration_simple_access_backup/
/trust_access_payload/
/APPLY_TRUST_CALIBRATION_SIMPLE_ACCESS.py
/README_TRUST_CALIBRATION_SIMPLE_ACCESS.txt
/TRUST_CALIBRATION_SIMPLE_ACCESS_MANIFEST.json
"""
    path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description="Simplify Trust Calibration participant/researcher access without changing study logic.")
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    repo = find_repo(args.repo)
    print("Repository:", repo)

    required = [
        repo / "awareml/studies/trust.py",
        repo / "awareml/studies/trust_analysis.py",
        repo / "awareml/ui_v2/phase16_trust_calibration.py",
        repo / "phase16_participant_app.py",
        repo / "scripts/validate_phase16_trust.py",
    ]
    missing = [str(path.relative_to(repo)) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Trust Calibration must already be installed. Missing: {}".format(", ".join(missing)))

    core_before = {}
    for rel in CORE_GUARDS:
        path = repo / rel
        if path.exists():
            core_before[str(rel)] = sha256_file(path)

    backup_root = repo / BACKUP_DIR_NAME / utc_stamp()
    backup_root.mkdir(parents=True, exist_ok=True)

    replaced = 0
    created = 0
    for rel in PAYLOAD_FILES:
        src = PAYLOAD_DIR / rel
        if not src.exists():
            raise SystemExit("Missing payload file: {}".format(src))
        dst = repo / rel
        if dst.exists():
            backup_file(repo, backup_root, rel)
            replaced += 1
        else:
            created += 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))

    gitignore_changed = patch_gitignore(repo, backup_root)

    compile_targets = [
        repo / "awareml/ui_v2/phase16_trust_calibration.py",
        repo / "phase16_participant_app.py",
    ]
    for path in compile_targets:
        py_compile.compile(str(path), doraise=True)
    print("Python compilation: PASS")

    core_after = {}
    for rel in CORE_GUARDS:
        path = repo / rel
        if path.exists():
            core_after[str(rel)] = sha256_file(path)

    if core_before != core_after:
        raise SystemExit("CORE LOGIC GUARD: FAIL — a protected study-logic/protocol file changed unexpectedly.")
    print("Scientific core hash guard: PASS")

    ui = (repo / "awareml/ui_v2/phase16_trust_calibration.py").read_text(encoding="utf-8")
    participant = (repo / "phase16_participant_app.py").read_text(encoding="utf-8")
    checks = {
        "single_dashboard_access_choice": "Choose your study access" in ui,
        "participant_button_present": "Enter participant study" in ui,
        "researcher_button_present": "Open researcher workspace" in ui,
        "dashboard_hidden_during_participant_study": 'section[data-testid="stSidebar"]' in ui,
        "pilot_main_labels_are_human_readable": "Pilot Study" in ui and "Main Study" in ui,
        "participant_only_entrypoint_preserved": "render_phase16_participant_study(isolated=True)" in participant,
        "researcher_key_still_protected": "compare_digest" in ui and "Researcher key" in ui,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("UI/access validation failed: {}".format(", ".join(failed)))

    print("Access/UI validation:")
    for name in sorted(checks):
        print("  {}: PASS".format(name))

    result = subprocess.run(
        [sys.executable, "-m", "scripts.validate_phase16_trust"],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout.rstrip())
    if result.returncode != 0:
        raise SystemExit("Trust Calibration validator failed after access simplification.")

    print("Files replaced:", replaced)
    print("Files created:", created)
    print(".gitignore updated:", gitignore_changed)
    print("Backup root:", backup_root)
    print("\nTRUST CALIBRATION SIMPLE ACCESS: PASS")
    print("Next checks:")
    print("  pytest -q tests/test_phase16_trust_calibration.py tests/test_trust.py tests/test_trust_calibration_access_ui.py")
    print("  pytest -q")
    print("  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
