from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


META_DIR = ROOT / "data" / "meta"
SNAPSHOT_DIR = META_DIR / "snapshots"
MODEL_DIR = (
    META_DIR
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


def require(path: Path):
    if not path.exists():
        raise RuntimeError(
            "Missing Phase-6 freeze prerequisite: {}".format(
                path
            )
        )


def main() -> None:
    model_manifest = (
        MODEL_DIR / "manifest.json"
    )

    required = [
        model_manifest,
        MODEL_DIR / "accuracy.joblib",
        MODEL_DIR / "runtime.joblib",
        MODEL_DIR / "energy.joblib",
        MODEL_DIR / "co2.joblib",
        SNAPSHOT_DIR
        / "recommender_train_v2.parquet",
        SNAPSHOT_DIR
        / "recommender_v2_benchmark_metrics.parquet",
        SNAPSHOT_DIR
        / "recommender_v2_oof_predictions.parquet",
        SNAPSHOT_DIR
        / "recommender_v2_model_selection.json",
        SNAPSHOT_DIR
        / "recommender_v2_preference_sensitivity.parquet",
        SNAPSHOT_DIR
        / "recommender_v2_preference_sensitivity_detail.parquet",
    ]
    for path in required:
        require(path)

    manifest = json.loads(
        model_manifest.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get("phase") != "6.3"
        or set(
            manifest.get("models", {})
        )
        != {
            "accuracy",
            "runtime",
            "energy",
            "co2",
        }
    ):
        raise RuntimeError(
            "Final model manifest is not a valid Phase-6.3 bundle."
        )

    release = {
        "schema_version": "2.0",
        "phase": "6.5",
        "release_id": "awareml-recommender-v2",
        "frozen_at_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),
        "model_manifest": {
            "path": (
                "models/recommender_v2/manifest.json"
            ),
            "sha256": sha256_file(
                model_manifest
            ),
        },
        "artifacts": {
            str(
                path.relative_to(
                    META_DIR
                )
            ).replace("\\", "/"): sha256_file(
                path
            )
            for path in required
        },
        "selected_models": {
            target: entry[
                "model_name"
            ]
            for target, entry in (
                manifest["models"].items()
            )
        },
        "evaluation_protocol": (
            "47-dataset Leave-One-Dataset-Out "
            "on development/meta datasets"
        ),
        "held_out_test_policy": (
            "The frozen 23-dataset evaluation split remains untouched "
            "until Phase 10."
        ),
        "status": "frozen",
    }

    release_path = (
        MODEL_DIR
        / "release_manifest.json"
    )
    release_path.write_text(
        json.dumps(
            release,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    Path(
        str(release_path) + ".sha256"
    ).write_text(
        "{}  {}\n".format(
            sha256_file(release_path),
            release_path.name,
        ),
        encoding="utf-8",
    )

    active = (
        META_DIR
        / "active_recommender_v2.txt"
    )
    active.write_text(
        "models/recommender_v2/manifest.json\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print(
        "AwareML Phase 6.5 — integration + freeze: SUCCESS"
    )
    print("=" * 72)
    print(
        "Active marker:",
        active,
    )
    print(
        "Model manifest SHA256:",
        sha256_file(model_manifest),
    )
    print(
        "Release manifest SHA256:",
        sha256_file(release_path),
    )
    print(
        "Legacy data/meta/active_snapshot.txt was NOT modified."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
