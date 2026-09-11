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
PAYLOAD_DIR = PACKAGE_DIR / "trust_ui_finish_payload"
BACKUP_DIR_NAME = ".trust_calibration_ui_finish_backup"

CORE_GUARDS = [
    Path("awareml/studies/trust.py"),
    Path("awareml/studies/trust_analysis.py"),
    Path("awareml/studies/trust_freeze.py"),
    Path("data/journal/trust_calibration_phase16_v1/design/protocol.json"),
    Path("data/journal/trust_stimulus_bank_v1/frozen/manifest.json"),
    Path("data/journal/trust_stimulus_bank_v1/frozen/stimuli_participant.json"),
    Path("data/journal/trust_stimulus_bank_v1/frozen/stimuli_researcher.json"),
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
    raise SystemExit("Could not locate AwareML_Extension. Run from the repository root.")


def backup_file(repo, backup_root, rel):
    src = repo / rel
    if not src.exists():
        return
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def patch_app(repo, backup_root):
    path = repo / "app.py"
    text = path.read_text(encoding="utf-8")
    original = text

    import_line = "from awareml.ui_v2.layout_polish import inject_layout_polish\n"
    if import_line not in text:
        anchor = "from awareml.ui_v2.theme import inject_research_theme\n"
        if anchor not in text:
            raise SystemExit("Could not find research-theme import anchor in app.py.")
        text = text.replace(anchor, anchor + import_line, 1)

    call_line = "inject_layout_polish()\n"
    if call_line not in text:
        anchor = 'inject_research_theme(state.get("theme_mode", "System"))\n'
        if anchor not in text:
            raise SystemExit("Could not find research-theme call anchor in app.py.")
        text = text.replace(anchor, anchor + call_line, 1)

    if text != original:
        backup_file(repo, backup_root, Path("app.py"))
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_participant_app(repo, backup_root):
    path = repo / "phase16_participant_app.py"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    original = text

    import_line = "from awareml.ui_v2.layout_polish import inject_layout_polish\n"
    if import_line not in text:
        anchor = "from awareml.ui_v2.theme import inject_research_theme\n"
        if anchor in text:
            text = text.replace(anchor, anchor + import_line, 1)

    if "inject_layout_polish()" not in text:
        anchor = 'inject_research_theme("System")\n'
        if anchor in text:
            text = text.replace(anchor, anchor + "inject_layout_polish()\n", 1)

    if text != original:
        backup_file(repo, backup_root, Path("phase16_participant_app.py"))
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_gitignore(repo, backup_root):
    path = repo / ".gitignore"
    if not path.exists():
        return False
    marker = "# Trust Calibration UI-finish local artifacts"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    backup_file(repo, backup_root, Path(".gitignore"))
    addition = """

# Trust Calibration UI-finish local artifacts
/.trust_calibration_ui_finish_backup/
/trust_ui_finish_payload/
/APPLY_TRUST_CALIBRATION_UI_FINISH.py
/README_TRUST_CALIBRATION_UI_FINISH.txt
/TRUST_CALIBRATION_UI_FINISH_MANIFEST.json
"""
    path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Apply final Trust Calibration spacing and Pilot researcher-access polish."
    )
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    repo = find_repo(args.repo)
    print("Repository:", repo)

    current_ui = repo / "awareml/ui_v2/phase16_trust_calibration.py"
    if not current_ui.exists():
        raise SystemExit("Trust Calibration UI is not installed.")
    current_text = current_ui.read_text(encoding="utf-8")
    expected = [
        "No prior AwareML run is required.",
        "Reference evidence.",
        "Choose your study access",
        'analysis_key = "p16_analysis_result_{}".format(mode)',
    ]
    missing = [item for item in expected if item not in current_text]
    if missing:
        raise SystemExit(
            "This patch expects the latest Trust Calibration usability build. "
            "Missing markers: {}".format(", ".join(missing))
        )

    core_before = {}
    for rel in CORE_GUARDS:
        path = repo / rel
        if path.exists():
            core_before[str(rel)] = sha256_file(path)

    backup_root = repo / BACKUP_DIR_NAME / utc_stamp()
    backup_root.mkdir(parents=True, exist_ok=True)

    replacements = [
        Path("awareml/ui_v2/phase16_trust_calibration.py"),
        Path("awareml/ui_v2/layout_polish.py"),
        Path("docs/TRUST_CALIBRATION_UI_FINISH.md"),
        Path("tests/test_trust_calibration_ui_finish.py"),
    ]

    replaced = 0
    created = 0
    for rel in replacements:
        src = PAYLOAD_DIR / rel
        dst = repo / rel
        if not src.exists():
            raise SystemExit("Missing payload file: {}".format(src))
        if dst.exists():
            backup_file(repo, backup_root, rel)
            replaced += 1
        else:
            created += 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))

    app_changed = patch_app(repo, backup_root)
    participant_app_changed = patch_participant_app(repo, backup_root)
    gitignore_changed = patch_gitignore(repo, backup_root)

    compile_targets = [
        repo / "awareml/ui_v2/phase16_trust_calibration.py",
        repo / "awareml/ui_v2/layout_polish.py",
        repo / "phase16_participant_app.py",
        repo / "tests/test_trust_calibration_ui_finish.py",
    ]
    for path in compile_targets:
        if path.exists():
            py_compile.compile(str(path), doraise=True)
    print("Python compilation: PASS")

    core_after = {}
    for rel in CORE_GUARDS:
        path = repo / rel
        if path.exists():
            core_after[str(rel)] = sha256_file(path)

    if core_before != core_after:
        raise SystemExit(
            "SCIENTIFIC CORE HASH GUARD: FAIL — protected study/stimulus files changed."
        )
    print("Scientific core hash guard: PASS")

    ui = current_ui.read_text(encoding="utf-8")
    app = (repo / "app.py").read_text(encoding="utf-8")
    checks = {
        "simple_pilot_key_restored":
            'LOCAL_PILOT_RESEARCHER_KEY = "phase16-local-test-key"' in ui,
        "streamlit_missing_secrets_probe_removed":
            "st.secrets.get" not in ui,
        "pilot_key_disabled_after_design_freeze":
            "not FINAL_DESIGN_MANIFEST.exists()" in ui,
        "trust_spacing_polish_present":
            'div[data-testid="stHorizontalBlock"]' in ui,
        "stimulus_diversity_note_present":
            "Stimulus diversity review before Main Study." in ui,
        "global_spacing_loaded":
            "inject_layout_polish()" in app,
        "scientific_access_and_analysis_logic_preserved":
            "TrustCalibrationStudy()" in ui
            and 'analysis_key = "p16_analysis_result_{}".format(mode)' in ui,
    }
    print("UI/access validation:")
    for name in sorted(checks):
        print("  {}: {}".format(name, "PASS" if checks[name] else "FAIL"))
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("TRUST CALIBRATION UI FINISH: FAIL — {}".format(", ".join(failed)))

    result = subprocess.run(
        [sys.executable, "-m", "scripts.validate_phase16_trust"],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout.rstrip())
    if result.returncode != 0:
        raise SystemExit("Trust Calibration validator failed after UI finish.")

    print("Files replaced:", replaced)
    print("Files created:", created)
    print("app.py spacing integration patched:", app_changed)
    print("participant-only spacing integration patched:", participant_app_changed)
    print(".gitignore updated:", gitignore_changed)
    print("Backup root:", backup_root)
    print("")
    print("TRUST CALIBRATION UI FINISH: PASS")
    print("Next checks:")
    print(
        "  pytest -q tests/test_phase16_trust_calibration.py tests/test_trust.py "
        "tests/test_trust_calibration_access_ui.py tests/test_trust_calibration_final_usability.py "
        "tests/test_trust_calibration_ui_finish.py"
    )
    print("  pytest -q")
    print("  python -m scripts.validate_phase16_trust")
    print("  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
