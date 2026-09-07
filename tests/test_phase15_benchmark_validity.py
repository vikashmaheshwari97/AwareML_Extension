from awareml.explanation_integrity.controlled import (
    build_controlled_cases,
    intervention_from_case,
)
from awareml.explanation_integrity.evidence import top_ranked_framework
from awareml.explanation_integrity.faithfulness import (
    FaithfulnessV2Evaluator,
    apply_intervention,
)


def _case(case_id):
    return next(
        case for case in build_controlled_cases()
        if case.case_id == case_id
    )


def test_stage_b_counterfactual_is_a_coherent_decision_flip():
    case = _case("P15_01_B")
    intervention = intervention_from_case(case)
    counterfactual = apply_intervention(case, intervention)

    assert top_ranked_framework(case.evidence) == "AutoClass"
    assert top_ranked_framework(counterfactual.evidence) == "ChaCha"
    assert intervention.metadata["expected_new_decision"] == "ChaCha"

    assert (
        counterfactual.evidence["candidates"]["AutoClass"]["accuracy"]
        == 0.69
    )
    assert (
        counterfactual.evidence["candidates"]["AutoClass"]["utility"]
        < counterfactual.evidence["candidates"]["ChaCha"]["utility"]
    )


def test_stage_chat_counterfactual_is_a_coherent_decision_flip():
    case = _case("P15_01_F_CHAT")
    intervention = intervention_from_case(case)
    counterfactual = apply_intervention(case, intervention)

    assert top_ranked_framework(case.evidence) == "AutoClass"
    assert top_ranked_framework(counterfactual.evidence) == "ChaCha"
    assert intervention.metadata["expected_new_decision"] == "ChaCha"


def test_real_stage_e_list_output_acknowledges_dp_counterfactual():
    original = """Here's the explanation of fairness disparities for the focus framework, including calibration fairness where available:

According to the evidence, the fairness disparities for the focus framework are as follows:

* For AutoClass, the disparity metrics are: dp_diff = 0.035, equal_opportunity_diff = 0.028, equalized_odds_gap = 0.044, group_brier_score_gap = 0.012, and group_ece_gap = 0.018.
* For ChaCha, the disparity metrics are: dp_diff = 0.043, equal_opportunity_diff = 0.0344, equalized_odds_gap = 0.0496, group_brier_score_gap = 0.0136, and group_ece_gap = 0.02.
* For OAML, the disparity metrics are: dp_diff = 0.051, equal_opportunity_diff = 0.0408, equalized_odds_gap = 0.0552, group_brier_score_gap = 0.0152, and group_ece_gap = 0.022.
* For EvoAutoML, the disparity metrics are: dp_diff = 0.059, equal_opportunity_diff = 0.0472, equalized_odds_gap = 0.0608, group_brier_score_gap = 0.0168, and group_ece_gap = 0.024.
* For AutoStreamML, the disparity metrics are: dp_diff = 0.067, equal_opportunity_diff = 0.0536, equalized_odds_gap = 0.0664, group_brier_score_gap = 0.0184, and group_ece_gap = 0.026.
"""

    counterfactual_text = """Based on the provided evidence:

* Fairness metrics:
        + AutoClass: dp_diff = 0.155, equal_opportunity_diff = 0.028, equalized_odds_gap = 0.044, group_brier_score_gap = 0.012, group_ece_gap = 0.018
        + ChaCha: dp_diff = 0.043, equal_opportunity_diff = 0.0344, equalized_odds_gap = 0.0496, group_brier_score_gap = 0.0136, group_ece_gap = 0.02
        + OAML: dp_diff = 0.051, equal_opportunity_diff = 0.0408, equalized_odds_gap = 0.0552, group_brier_score_gap = 0.0152, group_ece_gap = 0.022
        + EvoAutoML: dp_diff = 0.059, equal_opportunity_diff = 0.0472, equalized_odds_gap = 0.0608, group_brier_score_gap = 0.0168, group_ece_gap = 0.024
        + AutoStreamML: dp_diff = 0.067, equal_opportunity_diff = 0.0536, equalized_odds_gap = 0.0664, group_brier_score_gap = 0.0184, group_ece_gap = 0.026
"""

    class Fixed:
        source = "test-fixed"
        model = "llama3:8b"

        def generate(self, case):
            return counterfactual_text, {
                "source": self.source,
                "model": self.model,
            }

    case = _case("P15_01_E")
    record = FaithfulnessV2Evaluator().evaluate(
        case,
        intervention_from_case(case),
        Fixed(),
        original_explanation=original,
    )

    assert record.changed_evidence_acknowledged == 1.0
    assert record.numeric_update_accuracy == 1.0
    assert record.stale_claim_rate == 0.0
    assert record.faithfulness_score >= 0.90


def test_fairness_key_names_brier_and_ece_are_parsed():
    from awareml.explanation_integrity.verifier import GeneralEvidenceVerifier

    case = _case("P15_01_E")
    text = (
        "AutoClass: dp_diff = 0.035, equal_opportunity_diff = 0.028, "
        "equalized_odds_gap = 0.044, group_brier_score_gap = 0.012, "
        "group_ece_gap = 0.018"
    )

    report = GeneralEvidenceVerifier().verify(case, text)
    metrics = {
        item.claim.metric
        for item in report.claims
        if item.supported
    }

    assert {"dp", "eo", "eodds", "brier_gap", "ece_gap"}.issubset(metrics)
