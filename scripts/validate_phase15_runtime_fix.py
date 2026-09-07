from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    path = ROOT / relative
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main():
    benchmark = read("awareml/explanation_integrity/benchmark.py")
    runner = read("scripts/run_phase15_llm_benchmark.py")
    phase14_test = read("tests/test_phase14_robustness_hotfix.py")

    checks = {
        "phase14_undefined_equal_opportunity_test_fixed": (
            'assert result["equal_opportunity_diff"] is None' in phase14_test
        ),
        "llm_timeout_is_configurable": (
            "--timeout-sec" in runner
            and "default=300.0" in runner
        ),
        "llm_retries_are_configurable": (
            "--retries" in runner
            and "retries: int = 1" in benchmark
        ),
        "partial_failures_are_preserved": (
            '"failures": failures' in benchmark
            and '"failure_count": len(failures)' in benchmark
        ),
        "benchmark_does_not_abort_by_default": (
            "fail_fast: bool = False" in benchmark
        ),
        "completion_status_is_explicit": (
            '"complete": bool(complete)' in benchmark
            and '"completed_faithfulness_count"' in benchmark
        ),
        "partial_results_written_to_disk": (
            '"failures.json"' in runner
        ),
    }

    print("=" * 96)
    print("AwareML Phase-15 runtime resilience fix validation")
    print("=" * 96)

    failed = []
    for name, ok in checks.items():
        print("{:<66} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)

    print("=" * 96)
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))

    print("Phase-15 runtime resilience fix validation: PASS")


if __name__ == "__main__":
    main()
