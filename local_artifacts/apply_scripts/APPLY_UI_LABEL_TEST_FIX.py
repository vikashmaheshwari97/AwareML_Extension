from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


OLD = '    assert "Explanation Integrity · Phase 15" in advanced\n'
NEW = (
    '    assert \'"Explanation Integrity": phase15_explanation_integrity_page\' in advanced\n'
    '    assert "Explanation Integrity · Phase 15" not in advanced\n'
)


def find_repo(explicit=None):
    if explicit:
        repo = Path(explicit).resolve()
        if (repo / "tests" / "test_phase15_ui.py").exists():
            return repo
        raise SystemExit("ERROR: --repo does not look like AwareML_Extension.")

    cwd = Path.cwd().resolve()
    candidates = [cwd] + list(cwd.parents)
    for repo in candidates:
        if (repo / "tests" / "test_phase15_ui.py").exists() and (repo / "awareml").exists():
            return repo
    raise SystemExit("ERROR: Could not locate the AwareML_Extension repository.")


def main():
    parser = argparse.ArgumentParser(description="Update stale UI-label test after professional label cleanup.")
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    repo = find_repo(args.repo)
    test_file = repo / "tests" / "test_phase15_ui.py"
    text = test_file.read_text(encoding="utf-8")

    if NEW.strip() in text:
        print("Test expectation already updated: PASS")
        return 0

    if OLD not in text:
        raise SystemExit(
            "ERROR: Expected old assertion was not found. "
            "No file was changed so an unrelated test version is not overwritten."
        )

    backup_root = repo / ".ui_label_test_fix_backup" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_root / "tests" / "test_phase15_ui.py"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(test_file, backup_path)

    updated = text.replace(OLD, NEW, 1)
    test_file.write_text(updated, encoding="utf-8")

    check = test_file.read_text(encoding="utf-8")
    validations = {
        "clean_label_expected": '"Explanation Integrity": phase15_explanation_integrity_page' in check,
        "old_visible_label_rejected": 'assert "Explanation Integrity · Phase 15" not in advanced' in check,
        "integration_function_preserved": 'phase15_explanation_integrity_page' in check,
        "live_probe_checks_preserved": 'Exploratory Live Dataset Probe' in check,
    }

    print(f"Repository: {repo}")
    print(f"Updated: {test_file}")
    print(f"Backup: {backup_path}")
    print("Validation:")
    for name, ok in validations.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    if not all(validations.values()):
        shutil.copy2(backup_path, test_file)
        raise SystemExit("UI LABEL TEST FIX: FAIL — original test restored.")

    print("\nUI LABEL TEST FIX: PASS")
    print("Next checks:")
    print("  pytest -q tests/test_phase15_ui.py")
    print("  pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
