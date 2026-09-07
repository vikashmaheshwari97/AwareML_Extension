from awareml.explanation_integrity.controlled import (
    build_controlled_cases,
    intervention_from_case,
)
from awareml.explanation_integrity.faithfulness import (
    FaithfulnessV2Evaluator,
)
from awareml.explanation_integrity.verifier import GeneralEvidenceVerifier


ORIGINAL = """The current pre-run framework recommended is AutoClass. This recommendation is based on the ranking mode being "point" and the top-ranked framework being AutoClass. According to the evidence, AutoClass has the highest utility value of 0.367, which suggests that it is the most suitable framework for the given task. Additionally, AutoClass has the highest accuracy of 0.86 and the lowest runtime of 8.4 seconds, indicating its efficiency and effectiveness."""

COUNTERFACTUAL = """The current pre-run framework, AutoClass, is recommended due to its high accuracy (0.69) and ranking mode, which is set to "point". This ranking mode prioritizes frameworks with the highest accuracy, and AutoClass has the highest accuracy among the five frameworks. Additionally, AutoClass has a relatively low runtime (8.4 seconds) and energy consumption (0.00092 kWh) compared to other frameworks, making it a good balance between performance and efficiency."""


class _FixedCounterfactualGenerator:
    source = "test-fixed"
    model = "llama3:8b"

    def generate(self, case):
        return COUNTERFACTUAL, {
            "source": self.source,
            "model": self.model,
        }


def test_real_llama_original_is_parsed_as_supported():
    case = build_controlled_cases()[0]
    report = GeneralEvidenceVerifier().verify(case, ORIGINAL)

    assert report.metrics["claim_precision"] == 1.0
    assert report.metrics["numeric_correctness"] == 1.0
    assert report.metrics["unsupported_claim_rate"] == 0.0
    assert report.metrics["decision_consistency"] == 1.0


def test_real_llama_counterfactual_acknowledges_changed_accuracy():
    case = build_controlled_cases()[0]
    record = FaithfulnessV2Evaluator().evaluate(
        case,
        intervention_from_case(case),
        _FixedCounterfactualGenerator(),
        original_explanation=ORIGINAL,
    )

    assert record.changed_evidence_acknowledged == 1.0
    assert record.numeric_update_accuracy == 1.0
    assert record.stale_claim_rate == 0.0
    assert record.faithfulness_score >= 0.9

    # One separate unsupported causal/ranking rationale may remain, but the
    # factual numeric update itself must be recognized.
    assert (
        record.counterfactual_correctness["unsupported_claim_rate"]
        <= 0.25
    )
