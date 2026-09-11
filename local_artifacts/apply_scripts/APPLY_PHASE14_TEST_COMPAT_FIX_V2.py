from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil


ROOT = Path.cwd()
BACKUP = (
    ROOT
    / ".phase14_test_compat_backup"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


DUTCH_CONSTANTS = (
    'DUTCH_FILENAME = "dutch_census_stream_awareml.csv"\n'
    'DUTCH_TARGET = "occupation_binary"\n'
    'DUTCH_SENSITIVE = "sex"\n'
    'DUTCH_POSITIVE_LABEL = 1\n\n'
)


def backup(path: Path) -> None:
    if not path.exists():
        return
    dest = BACKUP / path.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


def patch_runner() -> None:
    path = ROOT / "scripts" / "run_phase14_repeatability.py"
    if not path.exists():
        raise RuntimeError(
            "Missing scripts/run_phase14_repeatability.py"
        )

    text = path.read_text(encoding="utf-8")
    original = text

    if 'DUTCH_TARGET = "occupation_binary"' not in text:
        needle = "from awareml.types import RunConfig"
        pos = text.find(needle)
        if pos == -1:
            raise RuntimeError(
                "Could not locate 'from awareml.types import RunConfig'."
            )

        line_end = text.find("\n", pos)
        if line_end == -1:
            line_end = len(text) - 1

        insert_at = line_end + 1
        text = (
            text[:insert_at]
            + "\n"
            + DUTCH_CONSTANTS
            + text[insert_at:]
        )

    text = text.replace(
        'csv_path.name.lower()\n        == "dutch_census_stream_awareml.csv"',
        'csv_path.name.lower()\n        == DUTCH_FILENAME',
    )
    text = text.replace(
        'csv_path.name.lower() == "dutch_census_stream_awareml.csv"',
        'csv_path.name.lower() == DUTCH_FILENAME',
    )

    fallback_marker = (
        "if csv_path.name.lower() == DUTCH_FILENAME and not profile:"
    )
    if fallback_marker not in text:
        needle = "    profile = _load_demo_profile(csv_path)\n"
        pos = text.find(needle)
        if pos != -1:
            insert_at = pos + len(needle)
            fallback = (
                "\n"
                "    if csv_path.name.lower() == DUTCH_FILENAME and not profile:\n"
                "        profile = {\n"
                '            "target": DUTCH_TARGET,\n'
                '            "sensitive_attribute": DUTCH_SENSITIVE,\n'
                '            "positive_label": DUTCH_POSITIVE_LABEL,\n'
                "        }\n"
            )
            text = text[:insert_at] + fallback + text[insert_at:]

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(
            "Patched repeatability runner: explicit Dutch defaults restored "
            "without making the runner Dutch-only."
        )
    else:
        print("Repeatability runner already compatible; no change needed.")


def patch_test() -> None:
    path = ROOT / "tests" / "test_phase14_robustness_hotfix.py"
    if not path.exists():
        raise RuntimeError(
            "Missing tests/test_phase14_robustness_hotfix.py"
        )

    text = path.read_text(encoding="utf-8")
    original = text

    old = '    assert result["equal_opportunity_diff"] == 0.0\n'
    new = (
        "    # Group B has no positive ground-truth cases in this synthetic\n"
        "    # fixture, so its true-positive rate is undefined. Equal Opportunity must\n"
        "    # remain None/N/A rather than being fabricated as zero.\n"
        '    assert result["equal_opportunity_diff"] is None\n'
    )

    if old in text:
        text = text.replace(old, new, 1)

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(
            "Patched robustness test: undefined Equal Opportunity remains N/A."
        )
    else:
        print("Robustness test already compatible; no change needed.")


def main() -> None:
    if not (ROOT / "app.py").exists() or not (ROOT / "awareml").exists():
        raise RuntimeError(
            "Run this installer from the AwareML_Extension project root."
        )

    BACKUP.mkdir(parents=True, exist_ok=True)

    patch_runner()
    patch_test()

    print("=" * 96)
    print("AwareML Phase-14 Test Compatibility Fix v2: APPLIED")
    print("=" * 96)
    print("Backup root:", BACKUP)
    print()
    print("Important:")
    print("  - No production fairness metric was changed to force a test pass.")
    print("  - Undefined Equal Opportunity still remains None/N/A.")
    print("  - Dutch Census values are convenience defaults only.")
    print("  - The Phase-14 runner still supports arbitrary CSV datasets.")
    print()
    print("Run next:")
    print(
        r"  pytest -q .\tests\test_phase14_hardening.py "
        r".\tests\test_phase14_integrated_labs.py "
        r".\tests\test_phase14_robustness_hotfix.py"
    )
    print(r"  python -m scripts.validate_phase14_hardening")
    print(r"  python -m scripts.validate_phase14_robustness_hotfix")
    print("=" * 96)


if __name__ == "__main__":
    main()
