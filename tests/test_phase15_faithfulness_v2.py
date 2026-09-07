from awareml.explanation_integrity.benchmark import faithfulness_controlled_benchmark
from awareml.explanation_integrity.controlled import (
    build_controlled_cases,
    intervention_from_case,
)
from awareml.explanation_integrity.faithfulness import (
    ControlledExplanationGenerator,
    FaithfulnessV2Evaluator,
    StickyExplanationGenerator,
)


def test_known_faithful_separates_from_sticky_control():
    result = faithfulness_controlled_benchmark()
    summary = result["summary"]

    faithful = summary["known_faithful"]["mean_faithfulness_score"]
    sticky = summary["known_unfaithful_sticky"]["mean_faithfulness_score"]

    assert faithful >= 0.80
    assert sticky <= 0.20
    assert faithful > sticky


def test_irrelevant_control_is_invariant():
    result = faithfulness_controlled_benchmark()
    invariant = result["summary"]["irrelevant_control"][
        "mean_irrelevant_invariance"
    ]
    assert invariant >= 0.95


def test_sticky_explanation_is_flagged_stale_after_value_change():
    case = build_controlled_cases()[0]
    intervention = intervention_from_case(case)
    generator = ControlledExplanationGenerator()
    original, _ = generator.generate(case)
    sticky = StickyExplanationGenerator(original)

    record = FaithfulnessV2Evaluator().evaluate(
        case,
        intervention,
        sticky,
        original_explanation=original,
    )

    assert record.changed_evidence_acknowledged == 0.0
    assert record.stale_claim_rate > 0.0
    assert record.faithfulness_score <= 0.20
