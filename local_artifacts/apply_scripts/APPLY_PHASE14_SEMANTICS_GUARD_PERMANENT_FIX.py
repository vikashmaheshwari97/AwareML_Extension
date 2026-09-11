from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
import shutil


ROOT = Path.cwd()
PAYLOAD = ROOT / "phase14_semantics_guard_payload"
BACKUP = (
    ROOT
    / ".phase14_semantics_guard_backup"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


def sha256(path: Path):
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def backup(path: Path):
    if not path.exists():
        return
    dest = BACKUP / path.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(path, dest)


def patch_phase14_test():
    path = ROOT / "tests" / "test_phase14_robustness_hotfix.py"
    if not path.exists():
        raise RuntimeError("Missing tests/test_phase14_robustness_hotfix.py")

    text = path.read_text(encoding="utf-8")
    original = text

    old = '    assert result["equal_opportunity_diff"] == 0.0\n'
    replacement = (
        "    # Group B has no positive ground-truth cases, therefore its TPR is\n"
        "    # undefined. Equal Opportunity must remain None/N/A; zero would\n"
        "    # fabricate fairness evidence that does not exist.\n"
        '    assert result["equal_opportunity_diff"] is None\n'
    )

    if old in text:
        text = text.replace(old, replacement, 1)

    # Defensive cleanup if an older installer already inserted a duplicate.
    text = text.replace(
        'assert result["equal_opportunity_diff"] == 0.0',
        'assert result["equal_opportunity_diff"] is None',
    )

    backup(path)
    path.write_text(text, encoding="utf-8")

    final = path.read_text(encoding="utf-8")
    if 'assert result["equal_opportunity_diff"] == 0.0' in final:
        raise RuntimeError("Stale fake-zero assertion still present after patch.")
    if 'assert result["equal_opportunity_diff"] is None' not in final:
        raise RuntimeError("Correct None/N/A assertion was not installed.")

    print("Patched Phase-14 robustness test permanently to None/N/A semantics.")


def copy_guard_files():
    for source in PAYLOAD.rglob("*"):
        if not source.is_file():
            continue
        destination = ROOT / source.relative_to(PAYLOAD)
        backup(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def remove_stale_installer_payloads():
    # These are extraction leftovers from older installers. They are NOT AwareML
    # runtime/source packages. Keeping them allowed a later installer to copy an
    # old test back over the corrected test.
    names = (
        "payload",
        "phase14_payload",
        "three_path_payload",
    )
    removed = []
    for name in names:
        path = ROOT / name
        if path.exists() and path.is_dir():
            backup(path)
            shutil.rmtree(path)
            removed.append(name)
    print("Removed stale installer payload directories:", removed or "none")


def update_gitignore():
    path = ROOT / ".gitignore"
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    rules = [
        "/payload/",
        "/phase14_payload/",
        "/three_path_payload/",
        "/phase14_semantics_guard_payload/",
        "/.phase14_semantics_guard_backup/",
    ]

    missing = [rule for rule in rules if rule not in text]
    if missing:
        backup(path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n# Installer extraction / safety artifacts\n")
            for rule in missing:
                handle.write(rule + "\n")
        print("Updated .gitignore for installer payload directories.")


def main():
    if not (ROOT / "app.py").exists() or not (ROOT / "awareml").exists():
        raise RuntimeError(
            "Run this installer from the AwareML_Extension project root."
        )
    if not PAYLOAD.exists():
        raise RuntimeError(
            "phase14_semantics_guard_payload is missing. Extract the full ZIP first."
        )

    BACKUP.mkdir(parents=True, exist_ok=True)

    protected = [
        ROOT / "awareml" / "analysis" / "fairness.py",
        ROOT / "data" / "journal" / "explanation_correctness_v1" / "frozen" / "manifest.json",
        ROOT / "data" / "journal" / "faithfulness_v2" / "frozen" / "manifest.json",
        ROOT / "data" / "journal" / "trust_stimulus_bank_v1" / "frozen" / "manifest.json",
    ]
    before = {str(path): sha256(path) for path in protected if path.exists()}

    patch_phase14_test()
    copy_guard_files()
    remove_stale_installer_payloads()
    update_gitignore()

    # Remove this installer's own payload after successful application so it
    # cannot become another stale source of files later.
    if PAYLOAD.exists():
        shutil.rmtree(PAYLOAD)

    after = {str(path): sha256(path) for path in protected if path.exists()}
    changed = [key for key in before if before.get(key) != after.get(key)]
    if changed:
        raise RuntimeError(
            "Protected production/frozen artifact changed unexpectedly: {}"
            .format(changed)
        )

    print("=" * 100)
    print("AwareML Phase-14 Fairness Semantics Permanent Guard: APPLIED")
    print("=" * 100)
    print("Backup:", BACKUP)
    print()
    print("Permanent fix:")
    print("  1. Equal Opportunity undefined case is asserted as None/N/A")
    print("  2. Production fairness.py is unchanged")
    print("  3. A guard test prevents the fake-zero assertion from returning unnoticed")
    print("  4. Stale generic installer payload folders were removed")
    print("  5. This installer's payload deletes itself after application")
    print("  6. Ollama preflight script installed for Phase-15 runs")
    print()
    print("Run next:")
    print(
        r"  pytest -q .\tests\test_phase14_robustness_hotfix.py "
        r".\tests\test_phase14_semantics_guard.py"
    )
    print(r"  python -m scripts.validate_phase14_semantics_guard")
    print(r"  pytest -q")
    print(r"  python -m scripts.preflight_phase15_ollama")
    print("=" * 100)


if __name__ == "__main__":
    main()
