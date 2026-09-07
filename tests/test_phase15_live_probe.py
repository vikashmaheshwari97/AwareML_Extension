import pandas as pd

from awareml.explanation_integrity.live import build_live_cases


def test_live_probe_builds_all_four_sources_from_generic_state():
    ranked = pd.DataFrame([
        {
            "rank": 1,
            "framework": "AutoClass",
            "utility": 0.8,
            "accuracy": 0.84,
            "runtime": 8.0,
            "energy": 0.001,
            "co2": 0.0004,
        },
        {
            "rank": 2,
            "framework": "OAML",
            "utility": 0.7,
            "accuracy": 0.81,
            "runtime": 7.0,
            "energy": 0.0009,
            "co2": 0.00038,
        },
    ])
    state = {
        "dataset_name": "generic_stream.csv",
        "target": "label",
        "selected_framework": "AutoClass",
        "ranking_mode": "point",
        "preference_weights": {
            "accuracy": 0.5,
            "runtime": 0.2,
            "energy": 0.15,
            "co2": 0.15,
        },
        "v2_candidates": ranked,
        "run_results": [
            {
                "framework": "AutoClass",
                "accuracy": 0.83,
                "runtime_sec": 9.0,
                "energy_kwh": 0.0011,
                "co2_kg": 0.00045,
                "fairness": {
                    "dp_diff": 0.05,
                    "equal_opportunity_diff": 0.04,
                    "equalized_odds_gap": 0.06,
                    "group_brier_score_gap": 0.01,
                    "group_ece_gap": 0.02,
                },
                "explainability": {
                    "status": "ok",
                    "method": "SHAP",
                    "shap_values": {
                        "age": 0.3,
                        "income": -0.2,
                    },
                    "top_features": [
                        {"feature": "age", "shap_value": 0.3},
                        {"feature": "income", "shap_value": -0.2},
                    ],
                },
            }
        ],
    }

    cases = build_live_cases(state)
    stages = {case.source_stage for case in cases}

    assert {"B", "E", "F_XAI", "F_CHAT"}.issubset(stages)
    assert all(case.metadata["journal_evidence"] is False for case in cases)
