from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    test_path = ROOT / "tests" / "test_phase14_robustness_hotfix.py"
    guard_path = ROOT / "tests" / "test_phase14_semantics_guard.py"

    text = test_path.read_text(encoding="utf-8")
    guard = guard_path.read_text(encoding="utf-8")

    stale_payloads = [
        name
        for name in (
            "payload",
            "phase14_payload",
            "three_path_payload",
        )
        if (ROOT / name).exists()
    ]

    checks = {
        "phase14_test_uses_na_semantics": (
            'assert result["equal_opportunity_diff"] is None' in text
        ),
        "fake_zero_assertion_removed": (
            'assert result["equal_opportunity_diff"] == 0.0' not in text
        ),
        "permanent_guard_test_installed": (
            "test_legacy_phase14_test_cannot_reintroduce_fake_zero" in guard
        ),
        "stale_generic_payloads_removed": (len(stale_payloads) == 0),
    }

    print("=" * 92)
    print("AwareML Phase-14 fairness semantics guard validation")
    print("=" * 92)

    failed = []
    for name, ok in checks.items():
        print("{:<62} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)

    if stale_payloads:
        print("Stale payload directories still present:", stale_payloads)

    print("=" * 92)
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))

    print("Phase-14 fairness semantics guard: PASS")


if __name__ == "__main__":
    main()
