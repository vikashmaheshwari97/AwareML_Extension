from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.recommender.v2_service import V2Recommender
from awareml.ui_v2.export import build_research_zip
from awareml.ui_v2.plots import decision_space_3d
from awareml.ui_v2.state import DEFAULT_STATE, phase_status
from awareml.ui_v2.dataset_advisor import dataset_advice

SNAPSHOT = ROOT / "data" / "meta" / "snapshots" / "recommender_train_v2.parquet"
PHASE8_REPORT = ROOT / "artifacts" / "phase8" / "phase8_faithfulness_report.json"


def profile_from_training_row(row):
    return {
        "dataset_family": str(row.get("dataset_family", "unknown")),
        "source_type": str(row.get("source_type", "unknown")),
        "drift_type": str(row.get("drift_type", "unknown")),
        "n_samples_dataset": int(row["n_samples_dataset"]),
        "n_features": int(row["n_features"]),
        "n_numeric_features": int(row["n_numeric_features"]),
        "n_categorical_features": int(row["n_categorical_features"]),
        "numeric_feature_fraction": float(row["numeric_feature_fraction"]),
        "categorical_feature_fraction": float(row["categorical_feature_fraction"]),
        "missing_fraction": float(row["missing_fraction"]),
        "n_classes": float(row["n_classes"]),
        "majority_class_fraction": float(row["majority_class_fraction"]),
        "minority_class_fraction": float(row["minority_class_fraction"]),
        "class_imbalance_ratio": float(row["class_imbalance_ratio"]),
        "class_entropy_normalized": float(row["class_entropy_normalized"]),
        "window_size": int(row["window_size"]),
        "time_budget_sec": float(row["time_budget_sec"]),
    }


def main():
    if not SNAPSHOT.exists():
        raise RuntimeError("Phase-6 recommender snapshot is missing.")
    if not PHASE8_REPORT.exists():
        raise RuntimeError("Phase-8 faithfulness report is missing.")

    status = phase_status()
    for phase in ["phase6", "phase7", "phase8"]:
        if not status[phase]["ready"]:
            raise RuntimeError("{} is not frozen/ready.".format(phase))

    train = pd.read_parquet(SNAPSHOT)
    row = train.iloc[0]
    profile = profile_from_training_row(row)

    recommender = V2Recommender(root=ROOT)
    ranked, meta = recommender.recommend_profile(
        profile,
        weights={"accuracy": 0.55, "runtime": 0.15, "energy": 0.15, "co2": 0.15},
        ranking_mode="point",
        coverage=0.90,
    )
    if len(ranked) != 5:
        raise RuntimeError("UI decision space requires five framework candidates.")

    fig = decision_space_3d(
        ranked,
        selected_framework=str(ranked.iloc[0]["framework"]),
    )
    if len(fig.data) != 5:
        raise RuntimeError("3D decision figure must contain five framework traces.")
    if any(getattr(trace, "type", None) != "scatter3d" for trace in fig.data):
        raise RuntimeError("Decision Space is not a real Plotly 3D scatter.")

    faith = json.loads(PHASE8_REPORT.read_text(encoding="utf-8"))
    state = dict(DEFAULT_STATE)
    state["dataset_name"] = "phase9_validation_profile"
    state["target"] = str(row.get("target", "class"))
    state["v2_profile"] = profile
    state["v2_candidates"] = ranked
    state["v2_ranking_meta"] = meta
    state["copilot_review"] = {"decision": "approved", "config_diff": []}

    bundle = build_research_zip(state, faithfulness_report=faith)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        names = set(archive.namelist())
        required = {
            "awareml_evidence.json",
            "run_metrics.csv",
            "research_report.html",
            "checksums.json",
        }
        if required - names:
            raise RuntimeError("Export bundle missing: {}".format(sorted(required - names)))
        payload = json.loads(archive.read("awareml_evidence.json"))
        assert payload["integrity"]["raw_dataset_rows_exported"] is False
        assert payload["integrity"]["held_out_23_dataset_split_used_by_ui"] is False

    advisor_demo = pd.DataFrame({
        "feature": [0, 1, 0, 1, 0, 1],
        "sex": [0, 1, 0, 1, 0, 1],
        "leaky_label_copy": [0, 1, 0, 1, 0, 1],
        "target": [0, 1, 0, 1, 0, 1],
    })
    advice = dataset_advice(advisor_demo, target="target")
    if advice.get("current_target") != "target":
        raise RuntimeError("Dataset advisor target handling failed.")
    sensitive_cols = [x["column"] for x in advice.get("sensitive_candidates", [])]
    if "sex" not in sensitive_cols:
        raise RuntimeError("Dataset advisor sensitive-attribute suggestion failed.")
    if not any("leaky_label_copy" in msg for msg in advice.get("warnings", [])):
        raise RuntimeError("Dataset advisor leakage warning failed.")

    from awareml.ui_v2.pages import PAGE_REGISTRY_V2

    expected_pages = {
        "Command Center", "Run Studio", "3D Decision Space",
        "Streaming Observatory", "Responsible AI", "Copilot Workspace",
        "Faithfulness Lab", "Export Center", "Advanced Labs",
    }
    if set(PAGE_REGISTRY_V2) != expected_pages:
        raise RuntimeError("Phase-9 page registry is incomplete.")

    print("=" * 72)
    print("AwareML Phase 9 UI validation: PASS")
    print("=" * 72)
    print("Frozen backend phases: 6, 7, 8 READY")
    print("Research workspaces:", len(PAGE_REGISTRY_V2))
    print("3D framework traces:", len(fig.data))
    print("Global research-state contract: PASS")
    print("Export bundle: PASS")
    print("Theme modes: System / Dark / Light")
    print("Generic dataset setup advisor: PASS")
    print("Raw rows exported: False")
    print("23-dataset held-out split used by UI: False")
    print("=" * 72)


if __name__ == "__main__":
    main()
