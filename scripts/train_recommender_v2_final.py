from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.recommender.v2_data import (
    load_recommender_train,
)
from awareml.recommender.v2_models import (
    available_model_specs,
    model_feature_columns,
    sanitize_predictions,
    target_column,
)
from awareml.recommender.v2_uncertainty import (
    empirical_residual_calibration,
)


SNAPSHOT_DIR = ROOT / "data" / "meta" / "snapshots"
MODEL_DIR = (
    ROOT
    / "data"
    / "meta"
    / "models"
    / "recommender_v2"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(
            lambda: fh.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def objective_correlations(
    frame: pd.DataFrame,
):
    cols = {
        "accuracy": "accuracy_mean",
        "runtime": "runtime_sec_mean",
        "energy": "energy_kwh_mean",
        "co2": "co2_kg_mean",
    }
    block = frame[
        list(cols.values())
    ].rename(
        columns={
            value: key
            for key, value in cols.items()
        }
    )
    corr = block.corr(
        method="spearman"
    )
    return {
        str(row): {
            str(col): float(corr.loc[row, col])
            for col in corr.columns
        }
        for row in corr.index
    }


def main() -> None:
    train_path = (
        SNAPSHOT_DIR
        / "recommender_train_v2.parquet"
    )
    selection_path = (
        SNAPSHOT_DIR
        / "recommender_v2_model_selection.json"
    )
    predictions_path = (
        SNAPSHOT_DIR
        / "recommender_v2_oof_predictions.parquet"
    )
    metrics_path = (
        SNAPSHOT_DIR
        / "recommender_v2_benchmark_metrics.parquet"
    )

    for path in [
        train_path,
        selection_path,
        predictions_path,
        metrics_path,
    ]:
        if not path.exists():
            raise RuntimeError(
                "Missing Phase-6.2 prerequisite: {}".format(
                    path
                )
            )

    train = load_recommender_train(
        train_path
    )
    selection_payload = json.loads(
        selection_path.read_text(
            encoding="utf-8"
        )
    )
    selected = (
        selection_payload.get(
            "selected_by_target"
        )
        or {}
    )
    expected_targets = {
        "accuracy",
        "runtime",
        "energy",
        "co2",
    }
    if set(selected) != expected_targets:
        raise RuntimeError(
            "Model selection is incomplete: {}".format(
                sorted(selected)
            )
        )

    oof = pd.read_parquet(
        predictions_path
    )
    specs = {
        spec.name: spec
        for spec in available_model_specs(
            include_xgboost=True
        )
    }
    features = model_feature_columns(True)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_entries = {}
    uncertainty = {}

    print("=" * 72)
    print(
        "AwareML Phase 6.3 — final models + uncertainty"
    )
    print("=" * 72)

    for target in [
        "accuracy",
        "runtime",
        "energy",
        "co2",
    ]:
        model_name = str(
            selected[target]["model"]
        )
        if model_name not in specs:
            raise RuntimeError(
                "Selected model is unavailable: {}".format(
                    model_name
                )
            )

        spec = specs[model_name]
        estimator = spec.factory(
            target,
            42,
        )
        y_col = target_column(target)
        estimator.fit(
            train[features],
            pd.to_numeric(
                train[y_col],
                errors="raise",
            ).to_numpy(dtype=float),
        )

        model_file = (
            MODEL_DIR
            / "{}.joblib".format(target)
        )
        joblib.dump(
            estimator,
            model_file,
            compress=3,
        )

        chosen_oof = oof[
            oof["target"].eq(target)
            & oof["model"].eq(model_name)
        ].copy()

        if len(chosen_oof) != 235:
            raise RuntimeError(
                "{} / {} has {} OOF rows, expected 235.".format(
                    target,
                    model_name,
                    len(chosen_oof),
                )
            )

        uncertainty[target] = (
            empirical_residual_calibration(
                chosen_oof
            )
        )

        # Final-training in-sample prediction is only a serialization smoke
        # check, not a reported evaluation result.
        final_pred = sanitize_predictions(
            target,
            estimator.predict(
                train[features]
            ),
        )
        if not np.isfinite(final_pred).all():
            raise RuntimeError(
                "Final {} model produced non-finite predictions.".format(
                    target
                )
            )

        model_entries[target] = {
            "model_name": model_name,
            "file": model_file.name,
            "sha256": sha256_file(
                model_file
            ),
            "training_rows": int(
                len(train)
            ),
            "training_datasets": int(
                train[
                    "dataset_id"
                ].nunique()
            ),
            "target_column": y_col,
            "benchmark_selection": selected[
                target
            ],
        }

        q90 = uncertainty[target][
            "intervals"
        ]["0.90"][
            "absolute_residual_quantile"
        ]
        print(
            "{:<8s} -> {:22s} q90_abs_residual={:.8g}".format(
                target,
                model_name,
                float(q90),
            )
        )

    try:
        import xgboost
        xgboost_version = xgboost.__version__
    except Exception:
        xgboost_version = None

    manifest = {
        "schema_version": "2.0",
        "phase": "6.3",
        "bundle_id": "awareml-recommender-v2",
        "created_at_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),
        "training_snapshot": {
            "file": train_path.name,
            "sha256": sha256_file(
                train_path
            ),
            "rows": int(len(train)),
            "datasets": int(
                train["dataset_id"].nunique()
            ),
            "frameworks": int(
                train["framework"].nunique()
            ),
        },
        "benchmark_artifacts": {
            "model_selection_sha256": sha256_file(
                selection_path
            ),
            "oof_predictions_sha256": sha256_file(
                predictions_path
            ),
            "benchmark_metrics_sha256": sha256_file(
                metrics_path
            ),
        },
        "feature_columns": features,
        "models": model_entries,
        "uncertainty": uncertainty,
        "objective_correlations": objective_correlations(
            train
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost_version,
        },
        "uncertainty_claim": (
            "Intervals are empirical LODO residual intervals and are not "
            "presented as exact conformal coverage guarantees."
        ),
    }

    manifest_path = (
        MODEL_DIR / "manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    checksum_path = Path(
        str(manifest_path) + ".sha256"
    )
    checksum_path.write_text(
        "{}  {}\n".format(
            sha256_file(manifest_path),
            manifest_path.name,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Manifest:",
        manifest_path,
    )
    print(
        "SHA256:",
        sha256_file(manifest_path),
    )
    print(
        "Phase 6.3: SUCCESS"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
