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
PAYLOAD_DIR = PACKAGE_DIR / "phase17_grounding_ux_payload"
BACKUP_NAME = ".information_seeking_grounding_ux_backup"

PAYLOAD_FILES = [
    Path("awareml/studies/information_seeking.py"),
    Path("awareml/studies/information_seeking_analysis.py"),
    Path("awareml/studies/information_seeking_grounding.py"),
    Path("awareml/studies/information_seeking_context.py"),
    Path("awareml/ui_v2/phase17_information_seeking.py"),
    Path("data/journal/information_seeking_v1/design/protocol.json"),
    Path("docs/PHASE17_INFORMATION_SEEKING.md"),
    Path("scripts/validate_phase17_information_seeking.py"),
    Path("tests/test_phase17_information_seeking.py"),
]

PHASE16_GUARDS = [
    Path("awareml/studies/trust.py"),
    Path("awareml/studies/trust_analysis.py"),
    Path("awareml/studies/trust_freeze.py"),
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


def patch_gitignore(repo, backup_root):
    path = repo / ".gitignore"
    if not path.exists():
        return False
    marker = "# Information-Seeking grounding/UX local artifacts"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    backup(repo, backup_root, Path(".gitignore"))
    addition = """

# Information-Seeking grounding/UX local artifacts
/.information_seeking_grounding_ux_backup/
/phase17_grounding_ux_payload/
/APPLY_INFORMATION_SEEKING_GROUNDING_UX_FIX.py
/README_INFORMATION_SEEKING_GROUNDING_UX_FIX.txt
/INFORMATION_SEEKING_GROUNDING_UX_MANIFEST.json
/artifacts/information_seeking_context_pilot.json
/artifacts/information_seeking_context_final.json
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
        repo / "awareml/ui_v2/phase17_information_seeking.py",
        repo / "awareml/studies/information_seeking.py",
        repo / "awareml/studies/information_seeking_analysis.py",
        repo / "scripts/validate_phase17_information_seeking.py",
    ]
    missing = [str(p.relative_to(repo)) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "The Information-Seeking Study must already be installed. Missing: {}".format(
                ", ".join(missing)
            )
        )

    phase16_before = {}
    for rel in PHASE16_GUARDS:
        path = repo / rel
        if path.exists():
            phase16_before[str(rel)] = sha(path)

    backup_root = repo / BACKUP_NAME / stamp()
    backup_root.mkdir(parents=True, exist_ok=True)

    replaced = 0
    created = 0
    for rel in PAYLOAD_FILES:
        src = PAYLOAD_DIR / rel
        if not src.exists():
            raise SystemExit("Missing payload file: {}".format(src))
        dst = repo / rel
        if dst.exists():
            backup(repo, backup_root, rel)
            replaced += 1
        else:
            created += 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))

    gitignore_changed = patch_gitignore(repo, backup_root)

    compile_targets = [
        repo / "awareml/studies/information_seeking.py",
        repo / "awareml/studies/information_seeking_analysis.py",
        repo / "awareml/studies/information_seeking_grounding.py",
        repo / "awareml/studies/information_seeking_context.py",
        repo / "awareml/ui_v2/phase17_information_seeking.py",
        repo / "scripts/validate_phase17_information_seeking.py",
        repo / "tests/test_phase17_information_seeking.py",
    ]
    for target in compile_targets:
        py_compile.compile(str(target), doraise=True)
    print("Python compilation: PASS")

    phase16_after = {}
    for rel in PHASE16_GUARDS:
        path = repo / rel
        if path.exists():
            phase16_after[str(rel)] = sha(path)
    if phase16_before != phase16_after:
        raise SystemExit(
            "PHASE-16 PROTECTION: FAIL — protected Trust Calibration files changed."
        )
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
        raise SystemExit("Information-Seeking validator failed after grounding/UX fix.")

    print("Files replaced:", replaced)
    print("Files created:", created)
    print(".gitignore updated:", gitignore_changed)
    print("Backup root:", backup_root)
    print("")
    print("INFORMATION-SEEKING GROUNDING/UX FIX: PASS")
    print("Next checks:")
    print("  pytest -q tests/test_phase17_information_seeking.py")
    print("  python -m scripts.validate_phase17_information_seeking")
    print("  pytest -q")
    print("  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
