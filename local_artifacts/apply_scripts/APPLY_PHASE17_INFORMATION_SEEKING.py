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
PAYLOAD_DIR = PACKAGE_DIR / "phase17_information_seeking_payload"
BACKUP_NAME = ".phase17_information_seeking_backup"

PAYLOAD_FILES = [
    Path("awareml/studies/information_seeking.py"),
    Path("awareml/studies/information_seeking_analysis.py"),
    Path("awareml/ui_v2/phase17_information_seeking.py"),
    Path("data/journal/information_seeking_v1/design/protocol.json"),
    Path("data/journal/information_seeking_v1/design/coding_scheme.json"),
    Path("docs/PHASE17_INFORMATION_SEEKING.md"),
    Path("scripts/validate_phase17_information_seeking.py"),
    Path("scripts/analyze_phase17_information_seeking.py"),
    Path("scripts/freeze_phase17_design.py"),
    Path("scripts/freeze_phase17_results.py"),
    Path("tests/test_phase17_information_seeking.py"),
]

PROTECTED_PHASE16 = [
    Path("awareml/studies/trust.py"),
    Path("awareml/studies/trust_analysis.py"),
    Path("data/journal/trust_calibration_phase16_v1/design/protocol.json"),
    Path("data/journal/trust_stimulus_bank_v1/frozen/manifest.json"),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp():
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
    raise SystemExit("Could not locate AwareML_Extension. Run from the repository root.")


def backup(repo, backup_root, rel):
    src = repo / rel
    if not src.exists():
        return
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def patch_study_labs(repo, backup_root):
    path = repo / "awareml" / "ui_v2" / "study_labs_v3.py"
    if not path.exists():
        raise SystemExit("Missing awareml/ui_v2/study_labs_v3.py")
    text = path.read_text(encoding="utf-8")
    marker = "# PHASE17_INFORMATION_SEEKING_OVERRIDE_V1"
    if marker in text:
        return False
    backup(repo, backup_root, Path("awareml/ui_v2/study_labs_v3.py"))
    addition = """

# PHASE17_INFORMATION_SEEKING_OVERRIDE_V1
# Keep the historical implementation above for provenance, but route the
# Advanced-Labs Information-Seeking workspace to the journal study implementation.
from .phase17_information_seeking import (
    information_seeking_research_page as _phase17_information_seeking_research_page,
)
information_seeking_research_page = _phase17_information_seeking_research_page
"""
    path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
    return True


def patch_advanced_label(repo, backup_root):
    path = repo / "awareml" / "ui_v2" / "pages_advanced.py"
    if not path.exists():
        raise SystemExit("Missing awareml/ui_v2/pages_advanced.py")
    text = path.read_text(encoding="utf-8")
    original = text
    replacements = [
        ('"Information-Seeking Lab": information_seeking_v2_page,',
         '"Information-Seeking Study": information_seeking_v2_page,'),
        ('"Information-Seeking · Phase 17": information_seeking_v2_page,',
         '"Information-Seeking Study": information_seeking_v2_page,'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    if text != original:
        backup(repo, backup_root, Path("awareml/ui_v2/pages_advanced.py"))
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_gitignore(repo, backup_root):
    path = repo / ".gitignore"
    if not path.exists():
        return False
    marker = "# Information-Seeking local installer artifacts"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    backup(repo, backup_root, Path(".gitignore"))
    addition = """

# Information-Seeking local installer artifacts
/.phase17_information_seeking_backup/
/phase17_information_seeking_payload/
/APPLY_PHASE17_INFORMATION_SEEKING.py
/README_PHASE17_INFORMATION_SEEKING.txt
/PHASE17_INFORMATION_SEEKING_MANIFEST.json
"""
    path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    repo = find_repo(args.repo)
    print("Repository:", repo)

    required = [
        repo / "awareml/ui_v2/study_labs_v3.py",
        repo / "awareml/ui_v2/pages_specialist.py",
        repo / "awareml/ui_v2/pages_advanced.py",
        repo / "awareml/studies/store.py",
    ]
    missing = [str(p.relative_to(repo)) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing prerequisites: {}".format(", ".join(missing)))

    phase16_before = {}
    for rel in PROTECTED_PHASE16:
        path = repo / rel
        if path.exists():
            phase16_before[str(rel)] = sha(path)

    backup_root = repo / BACKUP_NAME / stamp()
    backup_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for rel in PAYLOAD_FILES:
        src = PAYLOAD_DIR / rel
        if not src.exists():
            raise SystemExit("Missing payload: {}".format(src))
        dst = repo / rel
        if dst.exists():
            backup(repo, backup_root, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        copied += 1

    labs_patched = patch_study_labs(repo, backup_root)
    label_patched = patch_advanced_label(repo, backup_root)
    gitignore_patched = patch_gitignore(repo, backup_root)

    compile_targets = [
        repo / "awareml/studies/information_seeking.py",
        repo / "awareml/studies/information_seeking_analysis.py",
        repo / "awareml/ui_v2/phase17_information_seeking.py",
        repo / "scripts/validate_phase17_information_seeking.py",
        repo / "scripts/analyze_phase17_information_seeking.py",
        repo / "scripts/freeze_phase17_design.py",
        repo / "scripts/freeze_phase17_results.py",
    ]
    for target in compile_targets:
        py_compile.compile(str(target), doraise=True)
    print("Python compilation: PASS")

    phase16_after = {}
    for rel in PROTECTED_PHASE16:
        path = repo / rel
        if path.exists():
            phase16_after[str(rel)] = sha(path)
    if phase16_before != phase16_after:
        raise SystemExit("PHASE-16 PROTECTION: FAIL — protected Trust Calibration artifacts changed.")
    print("Phase-16 protection hash guard: PASS")

    result = subprocess.run(
        [sys.executable, "-m", "scripts.validate_phase17_information_seeking"],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout.rstrip())
    if result.returncode != 0:
        raise SystemExit("Information-Seeking validator failed.")

    print("Files copied/replaced:", copied)
    print("study_labs_v3 integration patched:", labs_patched)
    print("Advanced Labs label patched:", label_patched)
    print(".gitignore patched:", gitignore_patched)
    print("Backup root:", backup_root)
    print("")
    print("INFORMATION-SEEKING APPLY: PASS")
    print("Next checks:")
    print("  pytest -q tests/test_phase17_information_seeking.py")
    print("  pytest -q")
    print("  python -m scripts.validate_phase17_information_seeking")
    print("  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
