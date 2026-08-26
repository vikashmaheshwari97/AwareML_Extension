from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.recommender.v2_data import (
    load_recommender_train,
)
from awareml.recommender.v2_models import (
    model_feature_columns,
    sanitize_predictions,
)


MODEL_DIR = (
    ROOT
    / "data"
    / "meta"
    / "models"
    / "recommender_v2"
)
SNAPSHOT_DIR = (
    ROOT
    / "data"
    / "meta"
    / "snapshots"
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


def main() -> None:
    manifest_path = (
        MODEL_DIR / "manifest.json"
    )
    if not manifest_path.exists():
        raise RuntimeError(
            "Missing final V2 model manifest."
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )
    train = load_recommender_train(
        SNAPSHOT_DIR
        / "recommender_train_v2.parquet"
    )
    features = model_feature_columns(True)

    expected = {
        "accuracy",
        "runtime",
        "energy",
        "co2",
    }
    models = manifest.get("models") or {}
    uncertainty = (
        manifest.get("uncertainty") or {}
    )

    if set(models) != expected:
        raise RuntimeError(
            "Final model bundle target set is invalid."
        )
    if set(uncertainty) != expected:
        raise RuntimeError(
            "Uncertainty calibration target set is invalid."
        )

    for target in sorted(expected):
        entry = models[target]
        path = MODEL_DIR / entry["file"]
        if not path.exists():
            raise RuntimeError(
                "Missing model: {}".format(path)
            )
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(
                "Checksum mismatch: {}".format(
                    path.name
                )
            )

        estimator = joblib.load(path)
        pred = sanitize_predictions(
            target,
            estimator.predict(
                train[features].head(10)
            ),
        )
        if len(pred) != 10 or not np.isfinite(
            pred
        ).all():
            raise RuntimeError(
                "Prediction smoke test failed for {}.".format(
                    target
                )
            )

        intervals = uncertainty[target].get(
            "intervals"
        ) or {}
        if set(intervals) != {
            "0.80",
            "0.90",
            "0.95",
        }:
            raise RuntimeError(
                "Uncertainty coverage levels incomplete for {}.".format(
                    target
                )
            )

    print("=" * 72)
    print(
        "AwareML Phase 6.3 validation: PASS"
    )
    print("=" * 72)
    for target in [
        "accuracy",
        "runtime",
        "energy",
        "co2",
    ]:
        entry = models[target]
        q90 = uncertainty[target][
            "intervals"
        ]["0.90"][
            "absolute_residual_quantile"
        ]
        observed = uncertainty[target][
            "intervals"
        ]["0.90"][
            "oof_observed_coverage"
        ]
        print(
            "{:<8s} {:22s} q90={:.8g} observed={:.3f}".format(
                target,
                entry["model_name"],
                float(q90),
                float(observed),
            )
        )
    print(
        "Final-model checksums: PASS"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
