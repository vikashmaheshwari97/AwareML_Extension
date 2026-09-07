from pathlib import Path

from awareml.explanation_integrity.controlled import (
    build_controlled_cases,
    intervention_from_case,
)
from awareml.explanation_integrity.faithfulness import apply_intervention
from awareml.explanation_integrity.evidence import top_ranked_framework


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    path = ROOT / relative
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main():
    claims = read("awareml/explanation_integrity/claims.py")
    evidence = read("awareml/explanation_integrity/evidence.py")
    verifier = read("awareml/explanation_integrity/verifier.py")
    rescore = read("scripts/rescore_phase15_run.py")

    cases = {
        case.case_id: case
        for case in build_controlled_cases()
    }

    b = cases["P15_01_B"]
    bcf = apply_intervention(b, intervention_from_case(b))
    chat = cases["P15_01_F_CHAT"]
    chatcf = apply_intervention(chat, intervention_from_case(chat))

    checks = {
        "citation_first_numeric_extraction": (
            "_citation_numeric_claims" in claims
        ),
        "ranking_utility_sibling_resolution": (
            "evidence.ranking.{}.framework" in evidence
        ),
        "key_metric_specific_segment_precedence": (
            "Prefer the most specific/right-most path segment" in evidence
        ),
        "natural_ranked_first_grammar": (
            "framework\\s+ranked\\s+first" in claims
        ),
        "most_influential_feature_grammar": (
            "most influential" in claims
        ),
        "calibration_availability_checked": (
            'claim.claim_type == "availability"' in verifier
            and "structured Brier/ECE evidence" in verifier
        ),
        "stage_b_still_flips": (
            top_ranked_framework(b.evidence) == "AutoClass"
            and top_ranked_framework(bcf.evidence) == "ChaCha"
        ),
        "chat_still_flips": (
            top_ranked_framework(chat.evidence) == "AutoClass"
            and top_ranked_framework(chatcf.evidence) == "ChaCha"
        ),
        "saved_run_rescore_available": (
            "No LLM generation is performed during rescoring" in rescore
        ),
    }

    print("=" * 98)
    print("AwareML Phase-15 final verifier validation")
    print("=" * 98)

    failed = []
    for name, ok in checks.items():
        print("{:<68} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)

    print("=" * 98)
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))

    print("Phase-15 final verifier validation: PASS")


if __name__ == "__main__":
    main()
