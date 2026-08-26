import pandas as pd
from awareml.engine.pareto import epsilon_pareto_mask


def test_pareto_keeps_tradeoff_points():
    df = pd.DataFrame({
        "framework": ["A", "B", "C"],
        "accuracy": [0.9, 0.8, 0.7],
        "runtime_sec": [10.0, 5.0, 12.0],
    })
    mask = epsilon_pareto_mask(df, epsilon=0.0)
    assert bool(mask.iloc[0])
    assert bool(mask.iloc[1])
    assert not bool(mask.iloc[2])
