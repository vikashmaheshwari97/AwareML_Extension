from pathlib import Path

from awareml.explanation_integrity.controlled import (
    build_controlled_cases,
    intervention_from_case,
)
from awareml.explanation_integrity.evidence import top_ranked_framework
from awareml.explanation_integrity.faithfulness import apply_intervention


ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = {
        case.case_id: case
        for case in build_controlled_cases()
    }

    stage_b = cases["P15_01_B"]
    stage_b_cf = apply_intervention(
        stage_b,
        intervention_from_case(stage_b),
    )

    stage_chat = cases["P15_01_F_CHAT"]
    stage_chat_cf = apply_intervention(
        stage_chat,
        intervention_from_case(stage_chat),
    )

    claims = (
        ROOT
        / "awareml"
        / "explanation_integrity"
        / "claims.py"
    ).read_text(encoding="utf-8")

    checks = {
        "stage_b_decision_flip": (
            top_ranked_framework(stage_b.evidence) == "AutoClass"
            and top_ranked_framework(stage_b_cf.evidence) == "ChaCha"
        ),
        "stage_b_derived_utility_recomputed": (
            stage_b_cf.evidence["candidates"]["AutoClass"]["utility"]
            < stage_b_cf.evidence["candidates"]["ChaCha"]["utility"]
        ),
        "stage_chat_decision_flip": (
            top_ranked_framework(stage_chat.evidence) == "AutoClass"
            and top_ranked_framework(stage_chat_cf.evidence) == "ChaCha"
        ),
        "line_aware_markdown_parser": (
            "splitlines(True)" in claims
        ),
        "dp_diff_literal_supported": (
            "dp(?:_diff)?" in claims
        ),
        "brier_key_literal_supported": (
            "group[_\\s]+brier[_\\s]+score[_\\s]+gap" in claims
        ),
        "ece_key_literal_supported": (
            "group[_\\s]+ece[_\\s]+gap" in claims
        ),
    }

    print("=" * 98)
    print("AwareML Phase-15 benchmark validity validation")
    print("=" * 98)

    failed = []
    for name, ok in checks.items():
        print("{:<66} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)

    print("=" * 98)
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))

    print("Phase-15 benchmark validity: PASS")
    print(
        "Previous 4-case empirical run should remain diagnostic only; "
        "rerun after this fix before HPC scaling."
    )


if __name__ == "__main__":
    main()
