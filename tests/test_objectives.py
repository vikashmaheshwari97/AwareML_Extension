from awareml.recommender.schema import ObjectiveRequest
from awareml.types import ObjectiveWeights


def test_objective_normalization():
    w = ObjectiveWeights(accuracy=2, runtime=1, energy=0, co2=0, fairness=0, interpretability=0).as_dict()
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert abs(w["accuracy"] - 2/3) < 1e-9


def test_schema_rejects_all_zero():
    import pytest
    with pytest.raises(ValueError):
        ObjectiveRequest(accuracy=0, runtime=0, energy=0, co2=0, fairness=0, interpretability=0)
