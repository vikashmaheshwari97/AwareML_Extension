from pathlib import Path

from awareml.explanation_integrity.evidence import (
    evidence_numeric_candidates,
    flatten_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence_source = (
        ROOT
        / "awareml"
        / "explanation_integrity"
        / "evidence.py"
    ).read_text(encoding="utf-8")

    sample = {
        "candidates": {
            "EvoAutoML": {
                "accuracy": 0.779,
                "accuracy_lower": 0.58,
                "accuracy_upper": 1.0,
                "runtime": 60.01,
                "runtime_lower": 59.98,
                "runtime_upper": 60.04,
            }
        }
    }
    flat = flatten_evidence(sample)

    accuracy = evidence_numeric_candidates(
        flat,
        "accuracy",
        entity="EvoAutoML",
    )
    runtime = evidence_numeric_candidates(
        flat,
        "runtime",
        entity="EvoAutoML",
    )

    checks = {
        "point_priority_code_installed": (
            "exact_point_candidates" in evidence_source
            and "confidence/uncertainty interval fields" in evidence_source
        ),
        "accuracy_point_only": (
            accuracy == [
                (
                    "evidence.candidates.EvoAutoML.accuracy",
                    0.779,
                )
            ]
        ),
        "runtime_point_only": (
            runtime == [
                (
                    "evidence.candidates.EvoAutoML.runtime",
                    60.01,
                )
            ]
        ),
    }

    print("=" * 98)
    print("AwareML Phase-15 point-estimate resolver validation")
    print("=" * 98)

    failed = []
    for name, ok in checks.items():
        print("{:<66} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)

    print("=" * 98)
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))

    print("Phase-15 point-estimate resolver: PASS")


if __name__ == "__main__":
    main()
