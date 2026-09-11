from __future__ import annotations

import argparse
import os
import py_compile
import shutil
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PAYLOAD_DIR = PACKAGE_DIR / "trust_ui_polish_payload"
BACKUP_DIR_NAME = ".trust_calibration_ui_polish_backup"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def find_repo(explicit=None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(Path.cwd())
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
        if (resolved / "awareml").is_dir() and (resolved / "app.py").exists():
            return resolved
    raise SystemExit("Could not locate the AwareML_Extension repository. Run this installer from the repo root or pass --repo.")


def backup_file(repo: Path, backup_root: Path, rel: Path) -> None:
    src = repo / rel
    if not src.exists():
        return
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def copy_payload(repo: Path, backup_root: Path) -> list:
    written = []
    files = [
        Path("awareml/ui_v2/phase16_trust_calibration.py"),
        Path("phase16_participant_app.py"),
    ]
    for rel in files:
        src = PAYLOAD_DIR / rel
        if not src.exists():
            raise SystemExit("Missing payload file: {}".format(src))
        dst = repo / rel
        if dst.exists():
            backup_file(repo, backup_root, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        written.append(rel)
    return written


def patch_text_file(repo: Path, backup_root: Path, rel: Path, replacements) -> int:
    path = repo / rel
    if not path.exists():
        raise SystemExit("Missing required UI file: {}".format(rel))
    text = path.read_text(encoding="utf-8")
    updated = text
    count = 0
    for old, new in replacements:
        if old in updated:
            updated = updated.replace(old, new)
            count += 1
    if updated != text:
        backup_file(repo, backup_root, rel)
        path.write_text(updated, encoding="utf-8")
    return count


def patch_gitignore(repo: Path, backup_root: Path) -> bool:
    rel = Path(".gitignore")
    path = repo / rel
    if not path.exists():
        return False
    marker = "# Trust Calibration UI polish local artifacts"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    backup_file(repo, backup_root, rel)
    addition = """

# Trust Calibration UI polish local artifacts
/.trust_calibration_ui_polish_backup/
/trust_ui_polish_payload/
/APPLY_TRUST_CALIBRATION_UI_POLISH.py
/README_TRUST_CALIBRATION_UI_POLISH.txt
/TRUST_CALIBRATION_UI_POLISH_MANIFEST.json
"""
    path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
    return True


def compile_targets(repo: Path) -> None:
    targets = [
        repo / "awareml/ui_v2/phase16_trust_calibration.py",
        repo / "phase16_participant_app.py",
        repo / "awareml/ui_v2/pages_advanced.py",
        repo / "awareml/ui_v2/page_utils.py",
        repo / "app.py",
    ]
    for path in targets:
        py_compile.compile(str(path), doraise=True)


def validate_ui(repo: Path) -> dict:
    trust_ui = (repo / "awareml/ui_v2/phase16_trust_calibration.py").read_text(encoding="utf-8")
    participant = (repo / "phase16_participant_app.py").read_text(encoding="utf-8")
    advanced = (repo / "awareml/ui_v2/pages_advanced.py").read_text(encoding="utf-8")
    page_utils = (repo / "awareml/ui_v2/page_utils.py").read_text(encoding="utf-8")
    app = (repo / "app.py").read_text(encoding="utf-8")

    checks = {
        "trust_label_clean": '"Trust Calibration": trust_calibration_v2_page,' in advanced,
        "explanation_integrity_label_clean": '"Explanation Integrity": phase15_explanation_integrity_page,' in advanced,
        "trust_page_no_visible_phase_wording": all(x not in trust_ui for x in ["Phase 16", "Phase-16", "Phase 15", "Phase-15"]),
        "participant_page_no_visible_phase_wording": all(x not in participant for x in ["Phase 16", "Phase-16", "Phase 15", "Phase-15"]),
        "recommender_pill_no_v2": '"ML Recommender",' in page_utils and '"ML Recommender V2"' not in page_utils,
        "sidebar_brand_no_ui_v2": "Research OS · UI V2" not in app,
        "sidebar_recommender_no_v2": "Recommender V2" not in app,
        "isolated_hero_pills_use_valid_pairs": '("Blinded", "good")' in trust_ui,
        "participant_entrypoint_preserved": "render_phase16_participant_study(isolated=True)" in participant,
        "study_logic_imports_preserved": "TrustCalibrationStudy" in trust_ui and "analyze_phase16_store" in trust_ui,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("UI validation failed: {}".format(", ".join(failed)))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Trust Calibration professional UI polish without changing study logic.")
    parser.add_argument("--repo", default=None, help="Path to AwareML_Extension repo root")
    args = parser.parse_args()

    if not PAYLOAD_DIR.exists():
        raise SystemExit("Missing UI polish payload: {}".format(PAYLOAD_DIR))

    repo = find_repo(args.repo)
    print("Repository:", repo)

    required = [
        repo / "awareml/studies/trust.py",
        repo / "awareml/studies/trust_analysis.py",
        repo / "awareml/ui_v2/phase16_trust_calibration.py",
        repo / "phase16_participant_app.py",
    ]
    missing = [str(path.relative_to(repo)) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Trust Calibration implementation must already be installed. Missing: {}".format(", ".join(missing)))

    backup_root = repo / BACKUP_DIR_NAME / utc_stamp()
    backup_root.mkdir(parents=True, exist_ok=True)

    written = copy_payload(repo, backup_root)

    advanced_changes = patch_text_file(
        repo,
        backup_root,
        Path("awareml/ui_v2/pages_advanced.py"),
        [
            ('"Explanation Integrity · Phase 15": phase15_explanation_integrity_page,', '"Explanation Integrity": phase15_explanation_integrity_page,'),
            ('"Trust Calibration · Phase 16": trust_calibration_v2_page,', '"Trust Calibration": trust_calibration_v2_page,'),
        ],
    )
    pill_changes = patch_text_file(
        repo,
        backup_root,
        Path("awareml/ui_v2/page_utils.py"),
        [
            ('("ML Recommender V2",', '("ML Recommender",'),
        ],
    )
    app_changes = patch_text_file(
        repo,
        backup_root,
        Path("app.py"),
        [
            ("Research OS · UI V2", "Research OS"),
            ("Recommender V2", "Recommender"),
        ],
    )
    ignore_changed = patch_gitignore(repo, backup_root)

    compile_targets(repo)
    checks = validate_ui(repo)

    print("Python compilation: PASS")
    print("UI files replaced:", len(written))
    print("Advanced Labs labels updated:", advanced_changes)
    print("Global recommender pill updated:", pill_changes)
    print("Sidebar labels updated:", app_changes)
    print(".gitignore updated:", ignore_changed)
    print("Backup root:", backup_root)
    print("UI validation:")
    for key in sorted(checks):
        print("  {}: PASS".format(key))

    print("\nTRUST CALIBRATION UI POLISH: PASS")
    print("Next checks:")
    print("  pytest -q tests/test_phase16_trust_calibration.py tests/test_trust.py")
    print("  pytest -q")
    print("  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
