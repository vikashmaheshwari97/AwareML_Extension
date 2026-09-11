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
PAYLOAD_DIR = PACKAGE_DIR / "trust_final_usability_payload"
BACKUP_DIR_NAME = ".trust_calibration_final_usability_backup"

CORE_GUARDS = [
    Path("awareml/studies/trust.py"),
    Path("awareml/studies/trust_analysis.py"),
    Path("awareml/studies/trust_freeze.py"),
    Path("data/journal/trust_calibration_phase16_v1/design/protocol.json"),
    Path("data/journal/trust_stimulus_bank_v1/frozen/manifest.json"),
]

PAYLOAD_FILES = [
    Path("awareml/ui_v2/phase16_trust_calibration.py"),
    Path("docs/TRUST_CALIBRATION_STUDY_GUIDE.md"),
    Path("tests/test_trust_calibration_final_usability.py"),
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
        marker = str(resolved)
        if marker in seen:
            continue
        seen.add(marker)
        if (resolved / "app.py").exists() and (resolved / "awareml").is_dir():
            return resolved
    raise SystemExit(
        "Could not locate AwareML_Extension. Run this installer from the repository root."
    )


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
    marker = "# Trust Calibration final-usability local artifacts"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    backup_file(repo, backup_root, Path(".gitignore"))
    addition = """

# Trust Calibration final-usability local artifacts
/.trust_calibration_final_usability_backup/
/trust_final_usability_payload/
/APPLY_TRUST_CALIBRATION_FINAL_USABILITY.py
/README_TRUST_CALIBRATION_FINAL_USABILITY.txt
/TRUST_CALIBRATION_FINAL_USABILITY_MANIFEST.json
"""
    path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Apply Trust Calibration usability/readiness fixes without changing scientific core logic."
    )
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
        repo / "data/journal/trust_stimulus_bank_v1/frozen/stimuli_researcher.json",
    ]
    missing = [str(path.relative_to(repo)) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Trust Calibration must already be installed. Missing: {}".format(
                ", ".join(missing)
            )
        )

    current_ui = (
        repo / "awareml/ui_v2/phase16_trust_calibration.py"
    ).read_text(encoding="utf-8")
    if "Choose your study access" not in current_ui:
        raise SystemExit(
            "This patch expects the Simple Access upgrade to be installed first. "
            "No files were changed."
        )

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

    ui_path = repo / "awareml/ui_v2/phase16_trust_calibration.py"
    py_compile.compile(str(ui_path), doraise=True)
    py_compile.compile(
        str(repo / "tests/test_trust_calibration_final_usability.py"),
        doraise=True,
    )
    print("Python compilation: PASS")

    core_after = {}
    for rel in CORE_GUARDS:
        path = repo / rel
        if path.exists():
            core_after[str(rel)] = sha256_file(path)

    if core_before != core_after:
        raise SystemExit(
            "SCIENTIFIC CORE HASH GUARD: FAIL — a protected core/protocol/stimulus file changed unexpectedly."
        )
    print("Scientific core hash guard: PASS")

    ui = ui_path.read_text(encoding="utf-8")
    checks = {
        "analysis_isolated_by_collection_mode":
            'analysis_key = "p16_analysis_result_{}".format(mode)' in ui,
        "legacy_shared_analysis_cache_removed":
            'st.session_state.pop("p16_analysis_result", None)' in ui,
        "beginner_self_contained_guidance":
            "No prior AwareML run is required." in ui,
        "reference_evidence_added":
            "Reference evidence." in ui and "_reference_evidence_for_stimulus" in ui,
        "raw_provenance_hidden":
            "_EVIDENCE_TAG_RE" in ui,
        "readiness_wording_specific":
            "Pending instrument" in ui and "Pending calculation" in ui,
        "main_zero_state_explained":
            "Main Study currently has 0 participants because final collection is still locked." in ui,
        "module_form_commands":
            "python -m scripts.freeze_phase16_design" in ui,
        "existing_simple_access_preserved":
            "Choose your study access" in ui and "Open researcher workspace" in ui,
    }

    print("Usability validation:")
    for name in sorted(checks):
        print("  {}: {}".format(name, "PASS" if checks[name] else "FAIL"))

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(
            "TRUST CALIBRATION FINAL USABILITY: FAIL — checks failed: {}".format(
                ", ".join(failed)
            )
        )

    result = subprocess.run(
        [sys.executable, "-m", "scripts.validate_phase16_trust"],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout.rstrip())
    if result.returncode != 0:
        raise SystemExit(
            "Trust Calibration validator failed after the usability patch."
        )

    print("Files replaced:", replaced)
    print("Files created:", created)
    print(".gitignore updated:", gitignore_changed)
    print("Backup root:", backup_root)
    print("")
    print("TRUST CALIBRATION FINAL USABILITY: PASS")
    print("Next checks:")
    print(
        "  pytest -q tests/test_phase16_trust_calibration.py tests/test_trust.py "
        "tests/test_trust_calibration_access_ui.py tests/test_trust_calibration_final_usability.py"
    )
    print("  pytest -q")
    print("  python -m scripts.validate_phase16_trust")
    print("  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
