from awareml.explanation_integrity.evidence import (
    evidence_numeric_candidates,
    flatten_evidence,
)
from awareml.explanation_integrity.schemas import EvidenceCase
from awareml.explanation_integrity.verifier import GeneralEvidenceVerifier


def _evidence():
    return {
        "recommendation": {
            "top_framework": "EvoAutoML",
            "ranking_mode": "point",
        },
        "candidates": {
            "EvoAutoML": {
                "rank": 1,
                "utility": 0.70,
                "accuracy": 0.7790153724972056,
                "accuracy_lower": 0.5834859059564592,
                "accuracy_upper": 1.0,
                "runtime": 60.01348876953125,
                "runtime_lower": 59.98831052278789,
                "runtime_upper": 60.03866701627461,
                "energy": 0.0009980459422634748,
                "energy_lower": 0.0008514646350670207,
                "energy_upper": 0.001144627249459929,
                "co2": 0.00041554779766015917,
                "co2_lower": 0.0003538233329783938,
                "co2_upper": 0.00047727226234192454,
            }
        },
        "ranking": [
            {
                "rank": 1,
                "framework": "EvoAutoML",
                "utility": 0.70,
            }
        ],
    }


def test_point_accuracy_excludes_interval_bounds():
    flat = flatten_evidence(_evidence())
    rows = evidence_numeric_candidates(
        flat,
        "accuracy",
        entity="EvoAutoML",
    )

    assert rows == [
        (
            "evidence.candidates.EvoAutoML.accuracy",
            0.7790153724972056,
        )
    ]


def test_point_runtime_energy_co2_exclude_interval_bounds():
    flat = flatten_evidence(_evidence())

    expected = {
        "runtime": "evidence.candidates.EvoAutoML.runtime",
        "energy": "evidence.candidates.EvoAutoML.energy",
        "co2": "evidence.candidates.EvoAutoML.co2",
    }

    for metric, key in expected.items():
        rows = evidence_numeric_candidates(
            flat,
            metric,
            entity="EvoAutoML",
        )
        assert len(rows) == 1
        assert rows[0][0] == key


def test_real_stage_b_style_sentence_resolves_point_metrics():
    case = EvidenceCase(
        case_id="LIVE_STAGE_B_POINT_TEST",
        dataset_id="dutch_census_stream_awareml.csv",
        source_stage="B",
        source_name="Stage B · setup/recommendation explanation",
        prompt="Explain the recommendation.",
        evidence=_evidence(),
        metadata={},
    )

    explanation = (
        "EvoAutoML is the current top recommendation, with a utility of 0.70. "
        "Its predicted accuracy is 0.7790153724972056, runtime is "
        "60.01348876953125 seconds, energy is 0.0009980459422634748 kWh, "
        "and CO2 is 0.00041554779766015917 kg."
    )

    report = GeneralEvidenceVerifier().verify(case, explanation)

    assert report.metrics["contradiction_rate"] == 0.0
    assert report.metrics["numeric_correctness"] == 1.0

    numeric = [
        row for row in report.claims
        if row.claim.claim_type == "numeric"
    ]
    assert numeric
    assert all(row.supported for row in numeric)

    supported_metrics = {
        row.claim.metric
        for row in numeric
        if row.supported
    }
    assert {
        "accuracy",
        "runtime",
        "energy",
        "co2",
        "utility",
    }.issubset(supported_metrics)
