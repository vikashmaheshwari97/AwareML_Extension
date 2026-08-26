from awareml.analysis.sustainability import SustainabilitySession


def test_disabled_sustainability_is_missing_not_zero():
    record = SustainabilitySession(enabled=False).start().stop().to_dict()
    assert record["status"] == "not_measured"
    assert record["energy_kwh"] is None
    assert record["co2_kg"] is None
