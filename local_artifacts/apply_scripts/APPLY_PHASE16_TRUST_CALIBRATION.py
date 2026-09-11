from __future__ import annotations

import argparse
import hashlib
import os
import py_compile
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List


PACKAGE_DIR = Path(__file__).resolve().parent
PAYLOAD_DIR = PACKAGE_DIR / "phase16_trust_payload"
OVERRIDE_MARKER = "# PHASE16_TRUST_CALIBRATION_OVERRIDE_V1"
IGNORE_MARKER = "# Phase 16 — Trust Calibration local installer/runtime artifacts"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_repo(explicit: str = None) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(PACKAGE_DIR.resolve())
    candidates.extend(list(PACKAGE_DIR.resolve().parents)[:3])
    for candidate in candidates:
        if (candidate / "app.py").exists() and (candidate / "awareml").is_dir():
            return candidate
    raise SystemExit(
        "Could not locate AwareML_Extension repository. Run from the repo root or pass --repo <path>."
    )


def backup_file(repo: Path, backup_root: Path, rel: Path) -> None:
    src = repo / rel
    if not src.exists():
        return
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def copy_payload(repo: Path, backup_root: Path, force: bool) -> List[Path]:
    written: List[Path] = []
    for src in sorted(PAYLOAD_DIR.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(PAYLOAD_DIR)
        dst = repo / rel
        if dst.exists():
            # Existing Phase-16 design/frozen artifacts are never silently overwritten.
            protected_new = (
                str(rel).replace("\\", "/").startswith("data/journal/trust_calibration_phase16_v1/")
                or rel.name == "phase16_participant_app.py"
            )
            if protected_new and not force:
                # Replacement is allowed only when content is already identical.
                if sha256(dst) == sha256(src):
                    continue
                raise SystemExit(
                    "Refusing to overwrite existing Phase-16 file: {}. Re-run with --force only if you intend "
                    "to replace local draft Phase-16 work. Frozen result folders should never be overwritten.".format(rel)
                )
            backup_file(repo, backup_root, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        written.append(rel)
    return written


def patch_study_labs(repo: Path, backup_root: Path) -> bool:
    rel = Path("awareml/ui_v2/study_labs_v3.py")
    path = repo / rel
    if not path.exists():
        raise SystemExit("Missing required integration file: {}".format(rel))
    text = path.read_text(encoding="utf-8")
    if OVERRIDE_MARKER in text:
        return False
    backup_file(repo, backup_root, rel)
    addition = (
        "\n\n{}\n"
        "# Phase 16 replaces the earlier utility-based trust-calibration UI. Keep the old\n"
        "# function body only as historical source; all callers resolve to the Phase-16 page.\n"
        "from .phase16_trust_calibration import (\n"
        "    phase16_trust_calibration_page as trust_calibration_research_page,\n"
        ")\n"
    ).format(OVERRIDE_MARKER)
    path.write_text(text.rstrip() + addition, encoding="utf-8")
    return True


def patch_advanced_label(repo: Path, backup_root: Path) -> bool:
    rel = Path("awareml/ui_v2/pages_advanced.py")
    path = repo / rel
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if '"Trust Calibration · Phase 16"' in text:
        return False
    old = '"Trust Calibration": trust_calibration_v2_page,'
    new = '"Trust Calibration · Phase 16": trust_calibration_v2_page,'
    if old not in text:
        print("WARNING: Could not find the expected Advanced Labs label; integration still works through the function override.")
        return False
    backup_file(repo, backup_root, rel)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_gitignore(repo: Path, backup_root: Path) -> bool:
    path = repo / ".gitignore"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if IGNORE_MARKER in text:
        return False
    backup_file(repo, backup_root, Path(".gitignore"))
    addition = """

# Phase 16 — Trust Calibration local installer/runtime artifacts
/.phase16_trust_calibration_backup/
/phase16_trust_payload/
/APPLY_PHASE16_TRUST_CALIBRATION.py
/README_PHASE16_TRUST_CALIBRATION.txt
/PHASE16_PACKAGE_MANIFEST.json
"""
    path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
    return True


def verify_phase15_prerequisite(repo: Path) -> None:
    root = repo / "data/journal/trust_stimulus_bank_v1/frozen"
    required = [
        root / "manifest.json",
        root / "stimuli_participant.json",
        root / "stimuli_researcher.json",
        root / "summary.json",
    ]
    missing = [str(path.relative_to(repo)) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Phase 16 requires the frozen Phase-15 trust stimulus bank. Missing: {}".format(
                ", ".join(missing)
            )
        )


def compile_phase16(repo: Path) -> None:
    files = [
        repo / "awareml/studies/trust.py",
        repo / "awareml/studies/trust_analysis.py",
        repo / "awareml/studies/trust_freeze.py",
        repo / "awareml/ui_v2/phase16_trust_calibration.py",
        repo / "phase16_participant_app.py",
        repo / "scripts/analyze_phase16_trust.py",
        repo / "scripts/freeze_phase16_design.py",
        repo / "scripts/freeze_phase16_results.py",
        repo / "scripts/validate_phase16_trust.py",
    ]
    for path in files:
        py_compile.compile(str(path), doraise=True)


def run_validator(repo: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.validate_phase16_trust"],
        cwd=str(repo),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply AwareML Phase-16 Trust Calibration implementation.")
    parser.add_argument("--repo", default=None, help="Path to AwareML_Extension repo root")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of an existing Phase-16 draft file. Does not bypass final freeze safeguards.",
    )
    parser.add_argument("--skip-validator", action="store_true")
    args = parser.parse_args()

    if not PAYLOAD_DIR.exists():
        raise SystemExit("Missing package payload: {}".format(PAYLOAD_DIR))
    repo = find_repo(args.repo)
    print("Repository:", repo)
    verify_phase15_prerequisite(repo)

    backup_root = repo / ".phase16_trust_calibration_backup" / utc_stamp()
    backup_root.mkdir(parents=True, exist_ok=True)

    written = copy_payload(repo, backup_root, args.force)
    patched_study = patch_study_labs(repo, backup_root)
    patched_label = patch_advanced_label(repo, backup_root)
    patched_ignore = patch_gitignore(repo, backup_root)

    compile_phase16(repo)
    print("Phase-16 Python compilation: PASS")
    print("Files copied/replaced:", len(written))
    print("study_labs_v3 integration patched:", patched_study)
    print("Advanced Labs label patched:", patched_label)
    print(".gitignore patched:", patched_ignore)
    print("Backup root:", backup_root)

    if not args.skip_validator:
        code = run_validator(repo)
        if code != 0:
            print("Phase-16 package applied, but validator did not pass. Inspect the output above before committing.")
            return code

    print("\nPHASE-16 APPLY: PASS")
    print("Next checks:")
    print("  pytest -q tests/test_phase16_trust_calibration.py tests/test_trust.py")
    print("  streamlit run app.py")
    print("  streamlit run phase16_participant_app.py --server.port 8502")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
