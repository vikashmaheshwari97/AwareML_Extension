from awareml.explanation_integrity.benchmark import correctness_controlled_benchmark
from awareml.explanation_integrity.controlled import build_controlled_cases
from awareml.explanation_integrity.verifier import GeneralEvidenceVerifier


def test_known_correct_control_is_perfect():
    result = correctness_controlled_benchmark()
    metrics = result["summary"]["by_label"]["known_correct"]

    assert metrics["claim_precision"] == 1.0
    assert metrics["supported_claim_rate"] == 1.0
    assert metrics["numeric_correctness"] == 1.0
    assert metrics["citation_validity"] == 1.0
    assert metrics["unsupported_claim_rate"] == 0.0
    assert metrics["contradiction_rate"] == 0.0


def test_every_known_incorrect_stimulus_is_detected():
    result = correctness_controlled_benchmark()
    rows = [
        row for row in result["reports"]
        if row["researcher_label"] == "known_incorrect"
    ]
    assert len(rows) >= 20

    for row in rows:
        metrics = row["metrics"]
        detected = (
            float(metrics.get("contradiction_rate") or 0.0) > 0.0
            or float(metrics.get("unsupported_claim_rate") or 0.0) > 0.0
            or (
                metrics.get("citation_validity") is not None
                and float(metrics["citation_validity"]) < 1.0
            )
        )
        assert detected, row["case_id"]


def test_correctness_and_faithfulness_are_not_same_metric_family():
    result = correctness_controlled_benchmark()
    metrics = result["summary"]["overall"]

    assert "claim_precision" in metrics
    assert "numeric_correctness" in metrics
    assert "contradiction_rate" in metrics
    assert "faithfulness_score" not in metrics


def test_general_verifier_catches_wrong_framework_rank():
    case = [
        case for case in build_controlled_cases()
        if case.source_stage == "B"
    ][0]
    top = case.evidence["recommendation"]["top_framework"]
    other = next(
        name for name in case.evidence["candidates"]
        if name != top
    )
    text = "{} is ranked #1 [evidence.recommendation.top_framework].".format(
        other
    )
    report = GeneralEvidenceVerifier().verify(case, text)

    assert report.metrics["decision_consistency"] == 0.0
    assert report.metrics["contradiction_rate"] > 0.0
