from __future__ import annotations

from pathlib import Path
import shutil
from datetime import datetime


ROOT = Path.cwd()
BACKUP = (
    ROOT
    / ".phase14_test_compat_backup"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


def backup(path: Path):
    if not path.exists():
        return
    dest = BACKUP / path.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


def patch_runner():
    path = ROOT / "scripts" / "run_phase14_repeatability.py"
    text = path.read_text(encoding="utf-8")
    original = text

    constants = '''
DUTCH_FILENAME = "dutch_census_stream_awareml.csv"
DUTCH_TARGET = "occupation_binary"
DUTCH_SENSITIVE = "sex"
DUTCH_POSITIVE_LABEL = 1

'''

    if 'DUTCH_TARGET = "occupation_binary"' not in text:
        anchor = 'from awareml.types import RunConfig\\n\\n\\n'
        if anchor not in text:
            raise RuntimeError("Could not locate runner import anchor.")
        text = text.replace(
            anchor,
            'from awareml.types import RunConfig\\n\\n\\n' + constants,
            1,
        )

    text = text.replace(
        'csv_path.name.lower()\\n        == "dutch_census_stream_awareml.csv"',
        'csv_path.name.lower()\\n        == DUTCH_FILENAME',
    )

    marker = '    profile = _load_demo_profile(csv_path)\\n\\n'
    fallback = '''    profile = _load_demo_profile(csv_path)

    if csv_path.name.lower() == DUTCH_FILENAME and not profile:
        profile = {
            "target": DUTCH_TARGET,
            "sensitive_attribute": DUTCH_SENSITIVE,
            "positive_label": DUTCH_POSITIVE_LABEL,
        }

'''
    if marker in text and 'and not profile:' not in text:
        text = text.replace(marker, fallback, 1)

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(
            "Patched runner: restored explicit Dutch Census schema constants."
        )


def patch_test():
    path = ROOT / "tests" / "test_phase14_robustness_hotfix.py"
    text = path.read_text(encoding="utf-8")
    original = text

    old = '    assert result["equal_opportunity_diff"] == 0.0\\n'
    new = '''    # Equal opportunity is undefined for group B here because that
    # synthetic group contains no positive ground-truth cases. The correct
    # research-safe representation is None/N/A, never a fabricated zero.
    assert result["equal_opportunity_diff"] is None
'''

    if old in text:
        text = text.replace(old, new, 1)

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(
            "Patched test: undefined Equal Opportunity must remain None/N/A."
        )


def main():
    if not (ROOT / "app.py").exists():
        raise RuntimeError(
            "Run this from the AwareML_Extension project root."
        )

    BACKUP.mkdir(parents=True, exist_ok=True)

    patch_runner()
    patch_test()

    print("=" * 88)
    print("Phase-14 test compatibility fix applied.")
    print("Backup:", BACKUP)
    print()
    print(
        "No fairness metric was changed to manufacture a passing test."
    )
    print(
        "Undefined Equal Opportunity remains None/N/A by design."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
