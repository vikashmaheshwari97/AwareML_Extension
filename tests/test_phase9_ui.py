import io
import json
import zipfile

import pandas as pd

from awareml.ui_v2.export import build_research_zip
from awareml.ui_v2.plots import decision_space_3d
from awareml.ui_v2.state import DEFAULT_STATE


def sample_ranking():
    return pd.DataFrame({
        "rank": [1, 2, 3, 4, 5],
        "framework": ["OAML", "AutoClass", "ChaCha", "AutoStreamML", "EvoAutoML"],
        "utility": [0.84, 0.79, 0.73, 0.69, 0.61],
        "pareto_efficient": [True, True, False, True, False],
        "accuracy": [0.90, 0.92, 0.87, 0.85, 0.83],
        "accuracy_lower": [0.85, 0.87, 0.82, 0.80, 0.78],
        "accuracy_upper": [0.95, 0.97, 0.92, 0.90, 0.88],
        "runtime": [2.4, 3.0, 2.0, 2.7, 4.0],
        "runtime_lower": [2.0, 2.5, 1.7, 2.2, 3.5],
        "runtime_upper": [2.8, 3.5, 2.3, 3.2, 4.5],
        "energy": [0.20, 0.24, 0.18, 0.22, 0.30],
        "energy_lower": [0.15, 0.19, 0.13, 0.17, 0.25],
        "energy_upper": [0.25, 0.29, 0.23, 0.27, 0.35],
        "co2": [0.08, 0.10, 0.07, 0.09, 0.13],
        "co2_lower": [0.06, 0.08, 0.05, 0.07, 0.11],
        "co2_upper": [0.10, 0.12, 0.09, 0.11, 0.15],
    })


def test_global_state_contract():
    required = {
        "dataset", "target", "run_results", "selected_framework",
        "v2_candidates", "preference_weights", "copilot_proposal", "copilot_review",
    }
    assert required.issubset(DEFAULT_STATE)


def test_3d_decision_space_has_five_real_traces():
    fig = decision_space_3d(sample_ranking(), selected_framework="OAML")
    assert len(fig.data) == 5
    assert all(trace.type == "scatter3d" for trace in fig.data)


def test_export_bundle_has_core_files():
    state = dict(DEFAULT_STATE)
    state["v2_candidates"] = sample_ranking()
    state["dataset_name"] = "unit-test"
    state["target"] = "class"

    bundle = build_research_zip(
        state,
        faithfulness_report={
            "deterministic": {"mean_evidence_fidelity_score": 0.94}
        },
    )

    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        names = set(archive.namelist())
        assert {
            "awareml_evidence.json",
            "run_metrics.csv",
            "research_report.html",
            "checksums.json",
        }.issubset(names)

        payload = json.loads(archive.read("awareml_evidence.json"))
        assert payload["integrity"]["raw_dataset_rows_exported"] is False
        assert payload["integrity"]["held_out_23_dataset_split_used_by_ui"] is False


def test_theme_mode_contract():
    assert DEFAULT_STATE["theme_mode"] in {"System", "Dark", "Light"}


def test_temporal_drift_and_refit_annotations():
    from awareml.ui_v2.plots import temporal_metric_figure

    results = [
        {
            "framework": "OAML",
            "points": [
                {"sample": 500, "accuracy": 0.70},
                {"sample": 1000, "accuracy": 0.75},
                {"sample": 1500, "accuracy": 0.73},
                {"sample": 2000, "accuracy": 0.78},
            ],
            "drift_events": [1500],
            "refit_events": [2000],
        }
    ]
    fig = temporal_metric_figure(results, "accuracy", "Accuracy", "Accuracy")
    names = [getattr(trace, "name", "") for trace in fig.data]
    assert "Drift" in names
    assert "Refit / retrain" in names


def test_dutch_demo_dataset_has_no_target_leakage_column():
    from pathlib import Path
    demo = Path(__file__).resolve().parents[1] / "data" / "demo" / "dutch_census_stream_awareml.csv"
    frame = pd.read_csv(demo)
    assert "occupation_binary" in frame.columns
    assert "sex" in frame.columns
    assert "occupation" not in frame.columns
    assert len(frame) == 18438


def test_friendly_evidence_label():
    from awareml.ui_v2.components import _friendly_evidence_label
    assert _friendly_evidence_label(
        "evidence.before.recommendation.top_framework"
    ) == "Top framework"
    assert _friendly_evidence_label(
        "evidence.before.candidates.EvoAutoML.accuracy"
    ) == "EvoAutoML · accuracy"


def test_generic_dataset_advisor_detects_target_sensitive_and_leakage():
    import pandas as pd
    from awareml.ui_v2.dataset_advisor import dataset_advice

    df = pd.DataFrame({
        "age": [25, 31, 42, 28, 55, 37],
        "gender": [0, 1, 1, 0, 1, 0],
        "answer_copy": [0, 1, 1, 0, 1, 0],
        "label": [0, 1, 1, 0, 1, 0],
    })
    advice = dataset_advice(df, target="label")
    assert advice["current_target"] == "label"
    assert "gender" in [x["column"] for x in advice["sensitive_candidates"]]
    assert any("answer_copy" in warning for warning in advice["warnings"])


def test_generic_dataset_advisor_is_not_dutch_specific():
    import pandas as pd
    from awareml.ui_v2.dataset_advisor import suggest_targets

    df = pd.DataFrame({
        "sensor_a": [0.1, 0.2, 0.4, 0.8],
        "sensor_b": [1.0, 1.2, 0.9, 1.1],
        "class": [0, 0, 1, 1],
    })
    candidates = suggest_targets(df)
    assert candidates
    assert candidates[0]["column"] == "class"


def test_humanize_rationale_text_replaces_raw_evidence_ids():
    from awareml.ui_v2.components import humanize_rationale_text
    text = "AutoClass is strong [evidence.before.candidates.AutoClass.accuracy] and wins [evidence.before.recommendation.top_framework]."
    cleaned = humanize_rationale_text(text)
    assert "AutoClass · accuracy" in cleaned
    assert "Top framework" in cleaned
    assert "evidence.before" not in cleaned


def test_normalized_3d_decision_space_has_five_traces():
    from awareml.ui_v2.plots import decision_space_3d_normalized
    fig = decision_space_3d_normalized(sample_ranking(), selected_framework="OAML")
    assert len(fig.data) == 5
    assert all(trace.type == "scatter3d" for trace in fig.data)


def test_specialist_pages_import():
    from awareml.ui_v2.pages_specialist import (
        decision_lab_v2_page,
        drift_temporal_v2_page,
        fairness_v2_page,
        explainability_v2_page,
        sustainability_v2_page,
        trust_calibration_v2_page,
        information_seeking_v2_page,
    )
    assert all(callable(fn) for fn in [
        decision_lab_v2_page, drift_temporal_v2_page, fairness_v2_page,
        explainability_v2_page, sustainability_v2_page,
        trust_calibration_v2_page, information_seeking_v2_page,
    ])
