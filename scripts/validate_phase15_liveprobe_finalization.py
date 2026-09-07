from pathlib import Path

from awareml.explanation_integrity.faithfulness import (
    OllamaEvidenceExplanationGenerator,
)
from awareml.explanation_integrity.live_guided import (
    LiveGuidedOllamaGenerator,
)


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    path = ROOT / relative
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main():
    claims = read("awareml/explanation_integrity/claims.py")
    verifier = read("awareml/explanation_integrity/verifier.py")
    ui = read("awareml/ui_v2/phase15_explanation_integrity.py")
    helper = read("awareml/ui_v2/phase15_live_batch_ui.py")

    checks = {
        "empirical_prompt_v4_unchanged": (
            OllamaEvidenceExplanationGenerator.prompt_version
            == "phase15_explanation_prompt_v4"
        ),
        "live_prompt_v3_preserved": (
            LiveGuidedOllamaGenerator.prompt_version
            == "phase15_live_guided_prompt_v3"
        ),
        "rank_parser_requires_ranked_word": (
            r"(?:is|was|remains)\s+ranked" in claims
            and r"(?:ranked\s*)?" not in claims
        ),
        "fairness_direct_path_resolution": (
            "Prefer the direct canonical fairness path" in verifier
        ),
        "advanced_manual_not_expander": (
            "Show advanced manual verification" in ui
            and "with st.expander(\n        \"Advanced manual verification" not in ui
        ),
        "ui_v9": (
            "p15_live_v9_" in ui
            and "p15_live_v9_" in helper
        ),
        "review_semantics_explained": (
            "What REVIEW means" in helper
            and "You do not manually approve a REVIEW result" in helper
        ),
        "review_claim_diagnostics": (
            "Why this source is marked REVIEW" in helper
        ),
        "batch_first_preserved": (
            "Run complete Phase-15 live check" in helper
            and "You do not need to run the stages again" in helper
        ),
        "three_tabs_preserved": (
            '"Controlled benchmark"' in ui
            and '"Live Dataset Probe · exploratory"' in ui
            and '"Track 2 stimulus bank"' in ui
        ),
    }

    print("=" * 108)
    print("AwareML Phase-15 Live Probe finalization validation")
    print("=" * 108)

    failed = []
    for name, ok in checks.items():
        print("{:<82} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)

    print("=" * 108)
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))

    print("Phase-15 Live Probe finalization: PASS")


if __name__ == "__main__":
    main()
