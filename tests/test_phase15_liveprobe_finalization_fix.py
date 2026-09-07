from pathlib import Path

from awareml.explanation_integrity.claims import ClaimExtractor
from awareml.explanation_integrity.schemas import EvidenceCase
from awareml.explanation_integrity.verifier import GeneralEvidenceVerifier


ROOT = Path(__file__).resolve().parents[1]


def _stage_b_evidence():
    return {
        "recommendation": {"top_framework": "EvoAutoML"},
        "candidates": {
            "EvoAutoML": {
                "rank": 1,
                "utility": 0.7,
                "accuracy": 0.7790153725,
                "runtime": 60.01348877,
                "energy": 0.0009980459423,
                "co2": 0.0004155477977,
            }
        },
        "ranking": [
            {"rank": 1, "framework": "EvoAutoML", "utility": 0.7}
        ],
    }


def test_metric_sentence_is_not_misread_as_framework_rank():
    evidence = _stage_b_evidence()
    text = (
        "Predicted utility for EvoAutoML is 0.7.\n"
        "Predicted accuracy for EvoAutoML is 0.7790153725.\n"
        "Predicted runtime for EvoAutoML is 60.01348877 seconds."
    )
    claims = ClaimExtractor().extract(text, evidence)

    rank_claims = [
        claim for claim in claims
        if claim.claim_type == "framework_rank"
    ]
    numeric_metrics = {
        claim.metric for claim in claims
        if claim.claim_type == "numeric"
    }

    assert rank_claims == []
    assert {"utility", "accuracy", "runtime"}.issubset(numeric_metrics)


def test_explicit_rank_sentence_still_extracts_rank():
    evidence = _stage_b_evidence()
    claims = ClaimExtractor().extract(
        "EvoAutoML is ranked 1.",
        evidence,
    )
    rank_claims = [
        claim for claim in claims
        if claim.claim_type == "framework_rank"
    ]
    assert len(rank_claims) == 1
    assert rank_claims[0].rank_value == 1


def test_stage_e_fairness_sentence_is_not_fake_rank():
    evidence = {
        "frameworks": {
            "AutoClass": {
                "fairness": {
                    "dp_diff": 0.04247290245,
                    "equal_opportunity_diff": 0.0755899705,
                    "equalized_odds_gap": 0.0755899705,
                    "group_brier_score_gap": 0.007494084065,
                    "group_ece_gap": 0.01593189955,
                }
            }
        }
    }
    claims = ClaimExtractor().extract(
        "The DP/SPD gap in AutoClass is 0.04247290245.",
        evidence,
    )
    assert not any(
        claim.claim_type == "framework_rank"
        for claim in claims
    )
    assert any(
        claim.claim_type == "numeric"
        and claim.metric == "dp"
        for claim in claims
    )


def test_stage_e_direct_ece_path_wins_over_nested_duplicate():
    evidence = {
        "frameworks": {
            "AutoClass": {
                "fairness": {
                    "group_ece_gap": 0.01593189955,
                },
                "temporal_summary": {
                    "fairness": {
                        "group_ece_gap": 0.222,
                    }
                },
            }
        }
    }
    case = EvidenceCase(
        case_id="E",
        dataset_id="dutch.csv",
        source_stage="E",
        source_name="Stage E · fairness",
        prompt="Explain fairness.",
        evidence=evidence,
        metadata={"focus_framework": "AutoClass"},
    )
    report = GeneralEvidenceVerifier().verify(
        case,
        "The Group ECE gap in AutoClass is 0.01593189955.",
    )

    numeric = [
        row for row in report.claims
        if row.claim.claim_type == "numeric"
    ]
    assert len(numeric) == 1
    assert numeric[0].supported is True
    assert (
        numeric[0].matched_evidence_key
        == "evidence.frameworks.AutoClass.fairness.group_ece_gap"
    )


def test_dutch_style_stage_b_now_has_no_false_contradictions():
    case = EvidenceCase(
        case_id="B",
        dataset_id="dutch.csv",
        source_stage="B",
        source_name="Stage B",
        prompt="Explain.",
        evidence=_stage_b_evidence(),
        metadata={},
    )
    text = (
        "The EvoAutoML framework is the top recommendation.\n"
        "Predicted utility for EvoAutoML is 0.7.\n"
        "Predicted accuracy for EvoAutoML is 0.7790153725.\n"
        "Predicted runtime for EvoAutoML is 60.01348877 seconds.\n"
        "Predicted energy consumption for EvoAutoML is 0.0009980459423 units.\n"
        "Predicted CO2 emissions for EvoAutoML are 0.0004155477977 units."
    )
    report = GeneralEvidenceVerifier().verify(case, text)

    assert report.metrics["numeric_correctness"] == 1.0
    assert report.metrics["contradiction_rate"] == 0.0
    assert report.metrics["unsupported_claim_rate"] == 0.0
    assert report.metrics["claim_precision"] == 1.0


def test_ui_has_no_nested_advanced_expander():
    source = (
        ROOT
        / "awareml"
        / "ui_v2"
        / "phase15_explanation_integrity.py"
    ).read_text(encoding="utf-8")

    assert 'with st.expander(\n        "Advanced manual verification · optional"' not in source
    assert "Show advanced manual verification" in source
    assert "p15_live_v9_" in source


def test_review_guidance_and_diagnostics_are_present():
    source = (
        ROOT
        / "awareml"
        / "ui_v2"
        / "phase15_live_batch_ui.py"
    ).read_text(encoding="utf-8")

    assert "What REVIEW means" in source
    assert "You do not manually approve a REVIEW result" in source
    assert "Why this source is marked REVIEW" in source
    assert "p15_live_v9_" in source
