import numpy as np

from awareml.recommender import HistoricalMLRecommender, load_meta_logs, meta_log_coverage


def test_historical_meta_logs_and_recommender():
    df = load_meta_logs("data/meta/meta_logs.json")
    coverage = meta_log_coverage(df)
    assert coverage["rows"] > 0
    assert coverage["datasets"] >= 10
    assert "AutoStreamML" in coverage["frameworks"]

    model = HistoricalMLRecommender(meta_df=df, seed=7).train()
    candidates, meta = model.predict_candidates(
        n_features=14,
        n_classes=2,
        n_samples=48000,
        weights={"accuracy": 0.55, "runtime": 0.15, "energy": 0.15, "co2": 0.15},
    )
    assert set(candidates["framework"]) == set(model.frameworks)
    assert int(candidates.iloc[0]["rank"]) == 1
    assert np.isfinite(float(candidates.iloc[0]["utility"]))
    assert "coverage" in meta


def test_unmeasured_framework_carbon_is_not_invented():
    df = load_meta_logs("data/meta/meta_logs.json")
    model = HistoricalMLRecommender(meta_df=df, seed=7).train()
    candidates, _ = model.predict_candidates(14, 2, 48000)
    if "ChaCha" in set(candidates["framework"]):
        row = candidates[candidates["framework"] == "ChaCha"].iloc[0]
        # Historical ChaCha rows contain no positive CO2 measurements in the
        # bundled meta logs. Research-grade behavior is N/A, not a fabricated 0.
        assert int(row["co2_ug_support"]) == 0
        assert np.isnan(float(row["co2_ug"]))
