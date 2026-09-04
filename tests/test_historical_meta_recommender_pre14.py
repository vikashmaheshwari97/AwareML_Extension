from __future__ import annotations

import pandas as pd

from awareml.recommender.historical_preference import (
    HistoricalPreferenceRecommender,
    normalize_preference_weights,
)

FRAMEWORKS = ["AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha"]


def _stable_frame():
    rows = []
    for d in range(47):
        # Deliberate trade-off: AutoClass wins accuracy, OAML wins all efficiency metrics.
        metrics = {
            "AutoStreamML": (0.78, 12.0, 0.00030, 0.00012),
            "AutoClass": (0.90, 24.0, 0.00055, 0.00022),
            "EvoAutoML": (0.86, 36.0, 0.00070, 0.00030),
            "OAML": (0.82, 7.0, 0.00018, 0.00007),
            "ChaCha": (0.75, 18.0, 0.00040, 0.00016),
        }
        for fw in FRAMEWORKS:
            acc, rt, en, co2 = metrics[fw]
            rows.append(
                {
                    "dataset_id": f"D{d:02d}",
                    "framework": fw,
                    "seed_count": 3,
                    "accuracy_mean": acc + (d % 3) * 0.001,
                    "accuracy_std": 0.01,
                    "runtime_sec_mean": rt + (d % 2) * 0.1,
                    "runtime_sec_std": 0.2,
                    "energy_kwh_mean": en,
                    "energy_kwh_std": 0.00001,
                    "co2_kg_mean": co2,
                    "co2_kg_std": 0.00001,
                    "n_features": 10,
                    "n_classes": 2,
                    "n_samples_dataset": 5000,
                }
            )
    return pd.DataFrame(rows)


def _run_frame():
    rows = []
    base = _stable_frame()
    for _, row in base.iterrows():
        for seed, offset in [(42, 0.0), (43, 0.01), (44, -0.01)]:
            rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "framework": row["framework"],
                    "seed": seed,
                    "accuracy": float(row["accuracy_mean"]) + offset,
                    "runtime_sec": float(row["runtime_sec_mean"]) * (1.0 + (seed - 43) * 0.02),
                    "energy_kwh": float(row["energy_kwh_mean"]) * (1.0 + (seed - 43) * 0.03),
                    "co2_kg": float(row["co2_kg_mean"]) * (1.0 + (seed - 43) * 0.03),
                    "samples_processed": 5000,
                }
            )
    return pd.DataFrame(rows)


def test_weight_normalization():
    got = normalize_preference_weights({"accuracy": 50, "runtime": 25, "energy": 25, "co2": 0})
    assert abs(sum(got.values()) - 1.0) < 1e-12
    assert got["accuracy"] == 0.5


def test_stable_mode_uses_all_47_datasets_and_prefers_accuracy_winner():
    result = HistoricalPreferenceRecommender().recommend(
        {"accuracy": 100, "runtime": 0, "energy": 0, "co2": 0},
        train_frame=_stable_frame(),
    )
    assert result.dataset_count == 47
    assert result.winner == "AutoClass"
    assert len(result.ranking) == 5


def test_efficiency_preferences_can_change_the_global_prior():
    result = HistoricalPreferenceRecommender().recommend(
        {"accuracy": 10, "runtime": 35, "energy": 30, "co2": 25},
        train_frame=_stable_frame(),
    )
    assert result.winner == "OAML"


def test_best_seed_mode_uses_705_rows_and_one_whole_seed():
    runs = _run_frame()
    assert len(runs) == 705
    result = HistoricalPreferenceRecommender().recommend(
        {"accuracy": 60, "runtime": 20, "energy": 10, "co2": 10},
        seed_mode=HistoricalPreferenceRecommender.BEST_SEED,
        run_frame=runs,
    )
    assert result.run_count == 705
    assert result.dataset_count == 47
    assert result.seed_usage
    for fw, counts in result.seed_usage.items():
        assert sum(counts.values()) == 47
        assert set(counts).issubset({"42", "43", "44"})


def test_result_uses_validated_configuration_default_for_algorithm():
    result = HistoricalPreferenceRecommender().recommend(
        {"accuracy": 100},
        train_frame=_stable_frame(),
    )
    assert result.algorithm
    assert result.algorithm != "N/A"
