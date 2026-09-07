from awareml.explanation_integrity.controlled import (
    build_controlled_cases,
    intervention_from_case,
)
from awareml.explanation_integrity.faithfulness import (
    FaithfulnessV2Evaluator,
)
from awareml.explanation_integrity.verifier import GeneralEvidenceVerifier


CASES = {
    case.case_id: case
    for case in build_controlled_cases()
}


class Fixed:
    source = "test-replay"
    model = "llama3:8b"

    def __init__(self, text):
        self.text = text

    def generate(self, case):
        return self.text, {
            "source": self.source,
            "model": self.model,
        }


def test_stage_b_real_wording_is_fully_resolved():
    text = (
        'The current pre-run framework is recommended because it has the '
        'highest ranking, specifically "AutoClass" with a rank of 1. '
        'This ranking is based on the predicted evidence, which shows that '
        '"AutoClass" has the highest utility (0.367) and accuracy (0.86) '
        'among all the candidates.'
    )
    report = GeneralEvidenceVerifier().verify(CASES["P15_01_B"], text)

    assert report.metrics["claim_precision"] == 1.0
    assert report.metrics["numeric_correctness"] == 1.0
    assert report.metrics["decision_consistency"] == 1.0


def test_stage_e_citation_first_metrics_are_resolved():
    text = (
        "DP difference: 0.035 "
        "[evidence.frameworks.AutoClass.fairness.dp_diff]\n"
        "Equal opportunity difference: 0.028 "
        "[evidence.frameworks.AutoClass.fairness.equal_opportunity_diff]\n"
        "Equalized odds gap: 0.044 "
        "[evidence.frameworks.AutoClass.fairness.equalized_odds_gap]\n"
        "Group Brier score gap: 0.012 "
        "[evidence.frameworks.AutoClass.fairness.group_brier_score_gap]\n"
        "Group ECE gap: 0.018 "
        "[evidence.frameworks.AutoClass.fairness.group_ece_gap]"
    )
    report = GeneralEvidenceVerifier().verify(CASES["P15_01_E"], text)

    supported = {
        row.claim.metric
        for row in report.claims
        if row.supported
    }
    assert {"dp", "eo", "eodds", "brier_gap", "ece_gap"}.issubset(
        supported
    )
    assert report.metrics["numeric_correctness"] == 1.0


def test_calibration_unavailable_is_contradicted_when_brier_ece_exist():
    text = (
        "Calibration fairness is not explicitly mentioned in the evidence, "
        "so it is unavailable."
    )
    report = GeneralEvidenceVerifier().verify(CASES["P15_01_E"], text)

    assert report.metrics["contradiction_rate"] == 1.0
    assert report.metrics["unsupported_claim_rate"] == 0.0


def test_xai_most_influential_feature_is_decision_consistent():
    text = (
        'Based on the SHAP explanation, the most influential feature is '
        '"hours_per_week" with a SHAP value of -0.19. '
        '"age" has a SHAP value of 0.05.'
    )
    case = CASES["P15_01_F_XAI"]
    record = FaithfulnessV2Evaluator().evaluate(
        case,
        intervention_from_case(case),
        Fixed(text),
        original_explanation=(
            'The most influential feature is "age" with a SHAP value of 0.31.'
        ),
    )

    assert record.changed_evidence_acknowledged == 1.0
    assert record.decision_update_consistency == 1.0
    assert record.stale_claim_rate == 0.0


def test_chat_rank_and_ranking_utility_are_resolved():
    text = (
        'The framework ranked first is "AutoClass", with a rank of 1 and '
        'a utility value of 0.367. This indicates that "AutoClass" has the '
        'highest ranking.'
    )
    report = GeneralEvidenceVerifier().verify(
        CASES["P15_01_F_CHAT"],
        text,
    )

    assert report.metrics["claim_precision"] == 1.0
    assert report.metrics["numeric_correctness"] == 1.0
    assert report.metrics["decision_consistency"] == 1.0


def test_chat_decision_flip_is_recognized():
    text = (
        "Based on the provided evidence, the framework ranked first is ChaCha, "
        "with a ranking of 1 and a utility of 0.271. "
        "ChaCha has the highest accuracy (0.78)."
    )
    case = CASES["P15_01_F_CHAT"]
    record = FaithfulnessV2Evaluator().evaluate(
        case,
        intervention_from_case(case),
        Fixed(text),
        original_explanation=(
            'The framework ranked first is "AutoClass", with a rank of 1 '
            'and a utility value of 0.367.'
        ),
    )

    assert record.changed_evidence_acknowledged == 1.0
    assert record.decision_update_consistency == 1.0
    assert record.stale_claim_rate == 0.0
    assert record.faithfulness_score >= 0.9
