import pytest

from awareml.studies.trust import Phase16Error, TrustCalibrationStudy


def test_utility_based_trust_scaffold_is_retired():
    study = TrustCalibrationStudy.__new__(TrustCalibrationStudy)
    with pytest.raises(Phase16Error, match="removed the utility-based"):
        study.build_case(
            [
                {"framework": "A", "utility": 0.9},
                {"framework": "B", "utility": 0.7},
                {"framework": "C", "utility": 0.2},
            ],
            condition="wrong",
        )
