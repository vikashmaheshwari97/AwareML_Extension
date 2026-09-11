from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil


ROOT = Path.cwd()
BACKUP = (
    ROOT
    / ".phase14_validator_cleanup_backup"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


def backup(path: Path) -> None:
    if not path.exists():
        return
    dest = BACKUP / path.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


def patch_robustness_test() -> None:
    path = ROOT / "tests" / "test_phase14_robustness_hotfix.py"
    if not path.exists():
        raise RuntimeError("Missing tests/test_phase14_robustness_hotfix.py")

    text = path.read_text(encoding="utf-8")
    original = text

    old = '    assert result["equal_opportunity_diff"] == 0.0\n'
    new = (
        "    # Group B has no positive ground-truth cases in this synthetic\n"
        "    # fixture, so its TPR is undefined. Equal Opportunity therefore\n"
        "    # remains None/N/A rather than being fabricated as zero.\n"
        '    assert result["equal_opportunity_diff"] is None\n'
    )

    if old in text:
        text = text.replace(old, new, 1)

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print("Patched robustness test: undefined Equal Opportunity -> None/N/A.")
    else:
        print("Robustness test already aligned.")


def patch_hardening_validator() -> None:
    path = ROOT / "scripts" / "validate_phase14_hardening.py"
    if not path.exists():
        raise RuntimeError("Missing scripts/validate_phase14_hardening.py")

    text = path.read_text(encoding="utf-8")
    original = text

    text = text.replace(
        '"Per-group calibration details" in integrated',
        '"Per-group calibration and hard-label details" in integrated',
    )
    text = text.replace(
        '"repeatability_ui": "Repeatability · Phase 14" in integrated',
        '"repeatability_ui": "Dataset-specific repeatability · Phase 14" in integrated',
    )

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print("Patched legacy hardening validator for final-freeze UI wording.")
    else:
        print("Hardening validator already aligned.")


def patch_robustness_validator() -> None:
    path = ROOT / "scripts" / "validate_phase14_robustness_hotfix.py"
    if not path.exists():
        raise RuntimeError(
            "Missing scripts/validate_phase14_robustness_hotfix.py"
        )

    text = path.read_text(encoding="utf-8")
    original = text

    old_block = (
        '        "artifact_repeatability_loader": (\n'
        '            "phase14_repeated_results.json" in integrated\n'
        '            and "repeatability_manifest.json" in integrated\n'
        '        ),\n'
    )
    new_block = (
        '        "artifact_repeatability_loader": (\n'
        '            "find_latest_matching_run" in integrated\n'
        '            and "list_dataset_runs" in integrated\n'
        '            and "dataset_content_sha256" in integrated\n'
        '        ),\n'
    )

    if old_block in text:
        text = text.replace(old_block, new_block, 1)
    elif '"artifact_repeatability_loader"' in text:
        start = text.find('        "artifact_repeatability_loader": (')
        if start != -1:
            end = text.find("        ),\n", start)
            if end != -1:
                end += len("        ),\n")
                text = text[:start] + new_block + text[end:]

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print("Patched robustness validator for registry-based artifact loading.")
    else:
        print("Robustness validator already aligned.")


def main() -> None:
    if not (ROOT / "app.py").exists() or not (ROOT / "awareml").exists():
        raise RuntimeError(
            "Run this installer from the AwareML_Extension project root."
        )

    BACKUP.mkdir(parents=True, exist_ok=True)

    patch_robustness_test()
    patch_hardening_validator()
    patch_robustness_validator()

    print("=" * 96)
    print("AwareML Phase-14 legacy validator cleanup: APPLIED")
    print("=" * 96)
    print("Backup:", BACKUP)
    print()
    print("This cleanup changes only stale tests/validators.")
    print("It does NOT change fairness metrics, repeatability results,")
    print("the 5-repetition protocol, or any frozen research/model artifact.")
    print()
    print("Run next:")
    print(
        r"  pytest -q .\tests\test_phase14_hardening.py "
        r".\tests\test_phase14_integrated_labs.py "
        r".\tests\test_phase14_robustness_hotfix.py "
        r".\tests\test_phase14_final_freeze.py"
    )
    print(r"  python -m scripts.validate_phase14_hardening")
    print(r"  python -m scripts.validate_phase14_robustness_hotfix")
    print(r"  python -m scripts.validate_phase14_final_freeze")
    print("=" * 96)


if __name__ == "__main__":
    main()
