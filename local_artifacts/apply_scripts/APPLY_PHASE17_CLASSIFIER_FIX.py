from __future__ import annotations

import argparse
import py_compile
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


OLD_BLOCK = r'''    if _contains_any(
        t,
        [
            "source",
            "evidence",
            "show me",
            "where did",
            "prove",
            "data",
            "measurement",
            "metric",
            "numbers",
        ],
    ):
        return "evidence_request"

    if _contains_any(t, ["what does", "mean", "clarify", "define", "what is", "simpler terms"]):
        return "clarification"
'''

NEW_BLOCK = r'''    # Clarification takes precedence over generic metric/data keywords.
    # Example: "What does this metric mean?" is a clarification, not an
    # evidence request merely because it contains the word "metric".
    if _contains_any(t, ["what does", "mean", "clarify", "define", "what is", "simpler terms"]):
        return "clarification"

    if _contains_any(
        t,
        [
            "source",
            "evidence",
            "show me",
            "where did",
            "prove",
            "data",
            "measurement",
            "metric",
            "numbers",
        ],
    ):
        return "evidence_request"
'''


def find_repo(explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    candidates.append(Path.cwd().resolve())
    candidates.extend(list(Path.cwd().resolve().parents)[:4])
    for candidate in candidates:
        if (candidate / "app.py").exists() and (candidate / "awareml").is_dir():
            return candidate
    raise SystemExit("Could not locate AwareML_Extension repository.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    repo = find_repo(args.repo)
    classifier = repo / "awareml" / "studies" / "information_seeking.py"
    test_file = repo / "tests" / "test_phase17_information_seeking.py"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = repo / ".phase17_classifier_fix_backup" / stamp

    text = classifier.read_text(encoding="utf-8")
    if NEW_BLOCK in text:
        print("Classifier precedence fix already applied: PASS")
    elif OLD_BLOCK in text:
        backup_path = backup_root / "awareml" / "studies" / "information_seeking.py"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(classifier), str(backup_path))
        classifier.write_text(text.replace(OLD_BLOCK, NEW_BLOCK, 1), encoding="utf-8")
        print("Classifier precedence fix applied: PASS")
        print("Backup:", backup_path)
    else:
        raise SystemExit(
            "Expected classifier block was not found. No file changed so an unrelated version is not overwritten."
        )

    test_text = test_file.read_text(encoding="utf-8")
    exact_assert = '    assert classify_follow_up("What does this metric mean?") == "clarification"\n'
    if exact_assert not in test_text:
        anchor = '    assert classify_follow_up("What does equalized odds mean?") == "clarification"\n'
        if anchor not in test_text:
            raise SystemExit("Could not locate Phase-17 classifier test anchor.")
        backup_path = backup_root / "tests" / "test_phase17_information_seeking.py"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(test_file), str(backup_path))
        test_file.write_text(test_text.replace(anchor, anchor + exact_assert, 1), encoding="utf-8")
        print("Exact validator regression test added: PASS")
    else:
        print("Exact validator regression test already present: PASS")

    py_compile.compile(str(classifier), doraise=True)

    sys.path.insert(0, str(repo))
    from awareml.studies.information_seeking import classify_follow_up

    checks = {
        "evidence_request": classify_follow_up("Show me the evidence") == "evidence_request",
        "explanation_probe": classify_follow_up("Why is this recommended?") == "explanation_probe",
        "challenge": classify_follow_up("Are you sure this is right?") == "challenge",
        "comparison": classify_follow_up("Compare it with another framework") == "counterfactual_or_comparison",
        "clarification_exact_validator_phrase": classify_follow_up("What does this metric mean?") == "clarification",
    }
    print("Classifier smoke validation:")
    for name, ok in checks.items():
        print("  {}: {}".format(name, "PASS" if ok else "FAIL"))
    if not all(checks.values()):
        raise SystemExit("PHASE-17 CLASSIFIER FIX: FAIL")

    print("")
    print("PHASE-17 CLASSIFIER FIX: PASS")
    print("Next checks:")
    print("  pytest -q tests/test_phase17_information_seeking.py")
    print("  python -m scripts.validate_phase17_information_seeking")
    print("  pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
