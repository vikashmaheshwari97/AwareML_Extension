from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    path = ROOT / relative
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main():
    claims = read("awareml/explanation_integrity/claims.py")
    verifier = read("awareml/explanation_integrity/verifier.py")
    benchmark = read("awareml/explanation_integrity/benchmark.py")
    runner = read("scripts/run_phase15_llm_benchmark.py")

    checks = {
        "natural_parenthesized_numeric_claims": (
            r"\(?\s*(" in claims
        ),
        "utility_claims_supported": (
            '"utility"' in claims
        ),
        "natural_recommendation_rank_claims": (
            "top[-\\s]*ranked" in claims
            and "recommended" in claims
        ),
        "ranking_mode_categorical_claim": (
            'claim_type="categorical"' in claims
            and "ranking_mode" in verifier
        ),
        "faithfulness_uses_matched_changed_key": (
            "matched_evidence_key" in verifier
        ),
        "case_progress_callback": (
            "progress_callback" in benchmark
            and "case_start" in benchmark
            and "case_complete" in benchmark
        ),
        "runner_prints_case_progress": (
            "generating original explanation" in runner
            and "generating counterfactual" in runner
        ),
    }

    print("=" * 96)
    print("AwareML Phase-15 verifier grammar + progress validation")
    print("=" * 96)

    failed = []
    for name, ok in checks.items():
        print("{:<66} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)

    print("=" * 96)
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))

    print("Phase-15 verifier grammar + progress validation: PASS")


if __name__ == "__main__":
    main()
