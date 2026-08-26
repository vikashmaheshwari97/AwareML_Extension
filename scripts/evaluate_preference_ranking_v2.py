from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.recommender.v2_ranking import (
    normalize_weights,
)


SNAPSHOT_DIR = (
    ROOT
    / "data"
    / "meta"
    / "snapshots"
)
MODEL_DIR = (
    ROOT
    / "data"
    / "meta"
    / "models"
    / "recommender_v2"
)

SCENARIOS = {
    "balanced": {
        "accuracy": 0.25,
        "runtime": 0.25,
        "energy": 0.25,
        "co2": 0.25,
    },
    "accuracy_focus": {
        "accuracy": 0.70,
        "runtime": 0.10,
        "energy": 0.10,
        "co2": 0.10,
    },
    "speed_focus": {
        "accuracy": 0.20,
        "runtime": 0.50,
        "energy": 0.15,
        "co2": 0.15,
    },
    "sustainability_focus": {
        "accuracy": 0.20,
        "runtime": 0.10,
        "energy": 0.35,
        "co2": 0.35,
    },
    "energy_only": {
        "accuracy": 0.0,
        "runtime": 0.0,
        "energy": 1.0,
        "co2": 0.0,
    },
    "co2_only": {
        "accuracy": 0.0,
        "runtime": 0.0,
        "energy": 0.0,
        "co2": 1.0,
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(
            lambda: fh.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def score(values, maximize: bool):
    arr = np.asarray(values, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if abs(hi - lo) <= 1e-12:
        return np.full(
            len(arr),
            0.5,
            dtype=float,
        )
    out = (arr - lo) / (hi - lo)
    if not maximize:
        out = 1.0 - out
    return out


def main() -> None:
    predictions_path = (
        SNAPSHOT_DIR
        / "recommender_v2_oof_predictions.parquet"
    )
    manifest_path = (
        MODEL_DIR
        / "manifest.json"
    )

    if not predictions_path.exists():
        raise RuntimeError(
            "Missing Phase-6.2 OOF predictions."
        )
    if not manifest_path.exists():
        raise RuntimeError(
            "Run Phase 6.3 before Phase 6.4."
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )
    selected = {
        target: entry["model_name"]
        for target, entry in (
            manifest.get("models") or {}
        ).items()
    }

    oof = pd.read_parquet(
        predictions_path
    )

    chosen = []
    for target, model in selected.items():
        block = oof[
            oof["target"].eq(target)
            & oof["model"].eq(model)
        ][
            [
                "dataset_id",
                "framework",
                "y_true",
                "y_pred",
            ]
        ].copy()
        block = block.rename(
            columns={
                "y_true": target + "_true",
                "y_pred": target + "_pred",
            }
        )
        chosen.append(block)

    wide = chosen[0]
    for block in chosen[1:]:
        wide = wide.merge(
            block,
            on=[
                "dataset_id",
                "framework",
            ],
            how="inner",
            validate="one_to_one",
        )

    if len(wide) != 235:
        raise RuntimeError(
            "Selected OOF merge produced {} rows.".format(
                len(wide)
            )
        )

    detail_rows = []
    summary_rows = []

    for scenario_name, raw_weights in SCENARIOS.items():
        weights = normalize_weights(
            raw_weights
        )
        top1 = []
        top3 = []
        regrets = []

        for dataset_id, block in wide.groupby(
            "dataset_id",
            sort=True,
        ):
            block = block.copy()
            true_utility = np.zeros(
                len(block),
                dtype=float,
            )
            pred_utility = np.zeros(
                len(block),
                dtype=float,
            )

            for objective in [
                "accuracy",
                "runtime",
                "energy",
                "co2",
            ]:
                maximize = (
                    objective == "accuracy"
                )
                true_score = score(
                    block[
                        objective + "_true"
                    ],
                    maximize,
                )
                pred_score = score(
                    block[
                        objective + "_pred"
                    ],
                    maximize,
                )
                true_utility += (
                    weights[objective]
                    * true_score
                )
                pred_utility += (
                    weights[objective]
                    * pred_score
                )

            true_best = int(
                np.argmax(true_utility)
            )
            pred_order = np.argsort(
                -pred_utility
            )
            pred_best = int(
                pred_order[0]
            )

            top1.append(
                float(
                    true_best == pred_best
                )
            )
            top3.append(
                float(
                    true_best
                    in set(
                        pred_order[
                            :3
                        ].tolist()
                    )
                )
            )

            span = float(
                np.max(true_utility)
                - np.min(true_utility)
            )
            regret = (
                0.0
                if span <= 1e-12
                else max(
                    0.0,
                    (
                        float(
                            true_utility[
                                true_best
                            ]
                        )
                        - float(
                            true_utility[
                                pred_best
                            ]
                        )
                    )
                    / span,
                )
            )
            regrets.append(regret)

            detail_rows.append(
                {
                    "scenario": scenario_name,
                    "dataset_id": dataset_id,
                    "true_best_framework": str(
                        block.iloc[
                            true_best
                        ]["framework"]
                    ),
                    "predicted_best_framework": str(
                        block.iloc[
                            pred_best
                        ]["framework"]
                    ),
                    "top1_correct": float(
                        true_best == pred_best
                    ),
                    "top3_correct": float(
                        true_best
                        in set(
                            pred_order[
                                :3
                            ].tolist()
                        )
                    ),
                    "normalized_regret": regret,
                }
            )

        summary_rows.append(
            {
                "scenario": scenario_name,
                "top1_accuracy": float(
                    np.mean(top1)
                ),
                "top3_accuracy": float(
                    np.mean(top3)
                ),
                "normalized_regret": float(
                    np.mean(regrets)
                ),
                **{
                    "weight_" + key: value
                    for key, value in weights.items()
                },
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )
    detail = pd.DataFrame(
        detail_rows
    )

    summary_path = (
        SNAPSHOT_DIR
        / "recommender_v2_preference_sensitivity.parquet"
    )
    detail_path = (
        SNAPSHOT_DIR
        / "recommender_v2_preference_sensitivity_detail.parquet"
    )

    summary.to_parquet(
        summary_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    detail.to_parquet(
        detail_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    for path in [
        summary_path,
        detail_path,
    ]:
        Path(
            str(path) + ".sha256"
        ).write_text(
            "{}  {}\n".format(
                sha256_file(path),
                path.name,
            ),
            encoding="utf-8",
        )

    print("=" * 72)
    print(
        "AwareML Phase 6.4 — preference-aware ranking evaluation"
    )
    print("=" * 72)
    print(
        summary.to_string(
            index=False
        )
    )
    print()
    print(
        "Energy/CO2 Spearman correlation:",
        manifest[
            "objective_correlations"
        ]["energy"]["co2"],
    )
    print(
        "Phase 6.4: SUCCESS"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
