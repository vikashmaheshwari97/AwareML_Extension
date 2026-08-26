from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.faithfulness import (
    FaithfulnessEvaluator,
)
from awareml.llm import (
    OllamaClient,
)
from awareml.recommender.v2_service import (
    V2Recommender,
)


SNAPSHOT_DIR = (
    ROOT
    / "data"
    / "meta"
    / "snapshots"
)
OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "phase8"
)


GOAL = (
    "I need a highly accurate streaming classifier that adapts "
    "quickly to concept drift, keeps energy consumption moderate, "
    "and provides understandable evidence-grounded explanations."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def profile_from_row(row):
    return {
        "dataset_family": str(
            row.get(
                "dataset_family",
                "unknown",
            )
        ),
        "source_type": str(
            row.get(
                "source_type",
                "unknown",
            )
        ),
        "drift_type": str(
            row.get(
                "drift_type",
                "unknown",
            )
        ),
        "n_samples_dataset": int(
            row["n_samples_dataset"]
        ),
        "n_features": int(
            row["n_features"]
        ),
        "n_numeric_features": int(
            row["n_numeric_features"]
        ),
        "n_categorical_features": int(
            row["n_categorical_features"]
        ),
        "numeric_feature_fraction": float(
            row[
                "numeric_feature_fraction"
            ]
        ),
        "categorical_feature_fraction": float(
            row[
                "categorical_feature_fraction"
            ]
        ),
        "missing_fraction": float(
            row["missing_fraction"]
        ),
        "n_classes": float(
            row["n_classes"]
        ),
        "majority_class_fraction": float(
            row[
                "majority_class_fraction"
            ]
        ),
        "minority_class_fraction": float(
            row[
                "minority_class_fraction"
            ]
        ),
        "class_imbalance_ratio": float(
            row[
                "class_imbalance_ratio"
            ]
        ),
        "class_entropy_normalized": float(
            row[
                "class_entropy_normalized"
            ]
        ),
        "window_size": int(
            row["window_size"]
        ),
        "time_budget_sec": float(
            row["time_budget_sec"]
        ),
    }


def summary_from_results(
    results,
):
    if not results:
        return {
            "n_datasets": 0,
        }

    frame = pd.DataFrame(
        [
            {
                "dataset_id": row.dataset_id,
                "grounding_validity": (
                    row.grounding_validity
                ),
                "decision_alignment": (
                    row.decision_alignment
                ),
                "attribution_alignment": (
                    row.attribution_alignment
                ),
                "counterfactual_sensitivity": (
                    row.counterfactual_sensitivity
                ),
                "irrelevant_invariance": (
                    row.irrelevant_invariance
                ),
                "evidence_fidelity_score": (
                    row.evidence_fidelity_score
                ),
            }
            for row in results
        ]
    )

    return {
        "n_datasets": int(
            len(frame)
        ),
        "mean_grounding_validity": float(
            frame[
                "grounding_validity"
            ].mean()
        ),
        "mean_decision_alignment": float(
            frame[
                "decision_alignment"
            ].mean()
        ),
        "mean_attribution_alignment": float(
            frame[
                "attribution_alignment"
            ].mean()
        ),
        "mean_counterfactual_sensitivity": float(
            frame[
                "counterfactual_sensitivity"
            ].mean()
        ),
        "mean_irrelevant_invariance": float(
            frame[
                "irrelevant_invariance"
            ].mean()
        ),
        "mean_evidence_fidelity_score": float(
            frame[
                "evidence_fidelity_score"
            ].mean()
        ),
        "min_evidence_fidelity_score": float(
            frame[
                "evidence_fidelity_score"
            ].min()
        ),
        "max_evidence_fidelity_score": float(
            frame[
                "evidence_fidelity_score"
            ].max()
        ),
    }


def flatten_results(
    results,
):
    rows = []

    for result in results:
        for cf in result.counterfactuals:
            rows.append(
                {
                    "dataset_id": (
                        result.dataset_id
                    ),
                    "explanation_source": (
                        result
                        .explanation_source
                    ),
                    "model": (
                        result.model
                    ),
                    "original_top_framework": (
                        result
                        .original_top_framework
                    ),
                    "original_top_influence_objective": (
                        result
                        .original_top_influence_objective
                    ),
                    "grounding_validity": (
                        result
                        .grounding_validity
                    ),
                    "decision_alignment": (
                        result
                        .decision_alignment
                    ),
                    "attribution_alignment": (
                        result
                        .attribution_alignment
                    ),
                    "counterfactual_sensitivity_mean": (
                        result
                        .counterfactual_sensitivity
                    ),
                    "irrelevant_invariance": (
                        result
                        .irrelevant_invariance
                    ),
                    "evidence_fidelity_score": (
                        result
                        .evidence_fidelity_score
                    ),
                    "scenario": (
                        cf.scenario
                    ),
                    "changed_objectives": json.dumps(
                        cf.changed_objectives
                    ),
                    "counterfactual_top_framework": (
                        cf
                        .counterfactual_top_framework
                    ),
                    "decision_flipped": (
                        cf.decision_flipped
                    ),
                    "changed_objective_cited": (
                        cf
                        .changed_objective_cited
                    ),
                    "new_winner_acknowledged": (
                        cf
                        .new_winner_acknowledged
                    ),
                    "citation_change": (
                        cf.citation_change
                    ),
                    "counterfactual_sensitivity": (
                        cf
                        .counterfactual_sensitivity
                    ),
                    "original_rationale": (
                        cf.original_rationale.text
                    ),
                    "counterfactual_rationale": (
                        cf
                        .counterfactual_rationale
                        .text
                    ),
                    "original_evidence_keys": json.dumps(
                        cf
                        .original_rationale
                        .evidence_keys
                    ),
                    "counterfactual_evidence_keys": json.dumps(
                        cf
                        .counterfactual_rationale
                        .evidence_keys
                    ),
                }
            )

    return pd.DataFrame(rows)


def select_ollama_dataset_ids(
    dataset_ids,
    count,
):
    ids = sorted(
        set(
            str(value)
            for value in dataset_ids
        )
    )
    count = max(
        0,
        min(
            int(count),
            len(ids),
        ),
    )
    if count == 0:
        return []

    if count == 1:
        return [
            ids[len(ids) // 2]
        ]

    positions = np.linspace(
        0,
        len(ids) - 1,
        num=count,
        dtype=int,
    )
    return [
        ids[int(position)]
        for position in positions
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run AwareML Phase-8 evidence-faithfulness evaluation."
        )
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help=(
            "Evaluate a representative subset with the "
            "live local Ollama model in addition to the "
            "deterministic faithful baseline."
        ),
    )
    parser.add_argument(
        "--require-ollama",
        action="store_true",
    )
    parser.add_argument(
        "--ollama-datasets",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional exact Ollama model/tag for the live subset."
        ),
    )
    args = parser.parse_args()

    train_path = (
        SNAPSHOT_DIR
        / "recommender_train_v2.parquet"
    )
    if not train_path.exists():
        raise RuntimeError(
            "Phase-6 recommender snapshot is missing."
        )

    train = pd.read_parquet(
        train_path
    )
    datasets = (
        train
        .sort_values(
            [
                "dataset_id",
                "framework",
            ]
        )
        .drop_duplicates(
            "dataset_id"
        )
        .reset_index(drop=True)
    )

    if len(datasets) != 47:
        raise RuntimeError(
            "Phase 8 expects the frozen 47-dataset "
            "development/meta snapshot; found {}.".format(
                len(datasets)
            )
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommender = V2Recommender(
        root=ROOT
    )

    deterministic_evaluator = (
        FaithfulnessEvaluator(
            recommender=recommender
        )
    )

    deterministic_results = []

    print("=" * 72)
    print(
        "AwareML Phase 8 — evidence-faithfulness evaluation"
    )
    print("=" * 72)
    print(
        "Development/meta datasets:",
        len(datasets),
    )
    print(
        "Held-out 23-dataset test split used:",
        False,
    )
    print()

    for index, row in datasets.iterrows():
        dataset_id = str(
            row["dataset_id"]
        )
        print(
            "[{}/47] deterministic {}".format(
                index + 1,
                dataset_id,
            ),
            flush=True,
        )
        result = (
            deterministic_evaluator
            .evaluate_profile(
                dataset_id=dataset_id,
                profile=profile_from_row(
                    row
                ),
                goal=GOAL,
                use_llm=False,
            )
        )
        deterministic_results.append(
            result
        )

    deterministic_detail = (
        flatten_results(
            deterministic_results
        )
    )
    deterministic_summary = (
        summary_from_results(
            deterministic_results
        )
    )

    det_path = (
        OUTPUT_DIR
        / "deterministic_faithfulness_cases.parquet"
    )
    deterministic_detail.to_parquet(
        det_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    det_summary_path = (
        OUTPUT_DIR
        / "deterministic_faithfulness_summary.json"
    )
    det_summary_path.write_text(
        json.dumps(
            deterministic_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ollama_summary = {
        "requested": bool(
            args.ollama
            or args.require_ollama
        ),
        "status": "not_requested",
        "n_datasets": 0,
    }
    ollama_path = None
    ollama_summary_path = None

    if (
        args.ollama
        or args.require_ollama
    ):
        client = OllamaClient(
            model=args.model
        )
        status = client.status()

        if not status["reachable"]:
            ollama_summary = {
                "requested": True,
                "status": "unreachable",
                "status_detail": status,
                "n_datasets": 0,
            }
            if args.require_ollama:
                raise RuntimeError(
                    "Ollama is required but unreachable."
                )
        else:
            from awareml.faithfulness.rationale import (
                FaithfulRationaleGenerator,
            )

            live_evaluator = (
                FaithfulnessEvaluator(
                    recommender=recommender,
                    rationale_generator=(
                        FaithfulRationaleGenerator(
                            client=client
                        )
                    ),
                )
            )

            selected_ids = (
                select_ollama_dataset_ids(
                    datasets[
                        "dataset_id"
                    ].tolist(),
                    args.ollama_datasets,
                )
            )

            live_results = []

            print()
            print(
                "Live Ollama subset:",
                selected_ids,
            )

            for position, dataset_id in enumerate(
                selected_ids
            ):
                row = datasets[
                    datasets[
                        "dataset_id"
                    ].astype(str).eq(
                        dataset_id
                    )
                ].iloc[0]

                print(
                    "[{}/{}] ollama {}".format(
                        position + 1,
                        len(selected_ids),
                        dataset_id,
                    ),
                    flush=True,
                )

                live_results.append(
                    live_evaluator
                    .evaluate_profile(
                        dataset_id=dataset_id,
                        profile=profile_from_row(
                            row
                        ),
                        goal=GOAL,
                        use_llm=True,
                    )
                )

            live_detail = flatten_results(
                live_results
            )
            live_metrics = summary_from_results(
                live_results
            )

            ollama_path = (
                OUTPUT_DIR
                / "ollama_faithfulness_cases.parquet"
            )
            live_detail.to_parquet(
                ollama_path,
                index=False,
                engine="pyarrow",
                compression="zstd",
            )

            ollama_summary = {
                "requested": True,
                "status": "pass",
                "status_detail": status,
                "resolved_model": status.get(
                    "resolved_model"
                ),
                "dataset_ids": (
                    selected_ids
                ),
                **live_metrics,
            }

            ollama_summary_path = (
                OUTPUT_DIR
                / "ollama_faithfulness_summary.json"
            )
            ollama_summary_path.write_text(
                json.dumps(
                    ollama_summary,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )

    report = {
        "schema_version": "1.0",
        "phase": "8",
        "status": "pass",
        "created_at_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),
        "method": {
            "name": (
                "AwareML Evidence Fidelity (AEF)"
            ),
            "inspiration": [
                (
                    "FaithLM-style counterfactual "
                    "explanation sensitivity"
                ),
                (
                    "Faithfulness-Serum-style concern "
                    "for rationale/evidence alignment"
                ),
            ],
            "important_boundary": (
                "This Phase-8 implementation uses external "
                "evidence interventions and decision-influence "
                "analysis. It does not claim access to Ollama "
                "attention weights, PE-LRP, or internal model "
                "attribution."
            ),
            "components": {
                "grounding_validity": 0.25,
                "decision_alignment": 0.20,
                "attribution_alignment": 0.20,
                "counterfactual_sensitivity": 0.25,
                "irrelevant_invariance": 0.10,
            },
        },
        "development_source": {
            "file": train_path.name,
            "sha256": sha256_file(
                train_path
            ),
            "datasets": 47,
        },
        "held_out_23_dataset_split_used": False,
        "deterministic": (
            deterministic_summary
        ),
        "ollama": ollama_summary,
        "artifacts": {
            "deterministic_cases": (
                det_path.name
            ),
            "deterministic_summary": (
                det_summary_path.name
            ),
            "ollama_cases": (
                ollama_path.name
                if ollama_path
                else None
            ),
            "ollama_summary": (
                ollama_summary_path.name
                if ollama_summary_path
                else None
            ),
        },
    }

    report_path = (
        OUTPUT_DIR
        / "phase8_faithfulness_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    for path in [
        det_path,
        det_summary_path,
        report_path,
    ] + (
        [
            ollama_path,
            ollama_summary_path,
        ]
        if (
            ollama_path
            and ollama_summary_path
        )
        else []
    ):
        Path(
            str(path) + ".sha256"
        ).write_text(
            "{}  {}\n".format(
                sha256_file(path),
                path.name,
            ),
            encoding="utf-8",
        )

    print()
    print("=" * 72)
    print(
        "Phase 8 deterministic summary"
    )
    print("=" * 72)
    for key, value in (
        deterministic_summary.items()
    ):
        print(
            "{}: {}".format(
                key,
                value,
            )
        )

    print()
    print(
        "Ollama status:",
        ollama_summary[
            "status"
        ],
    )
    if (
        ollama_summary[
            "status"
        ] == "pass"
    ):
        print(
            "Ollama model:",
            ollama_summary.get(
                "resolved_model"
            ),
        )
        print(
            "Ollama mean AEF:",
            ollama_summary.get(
                "mean_evidence_fidelity_score"
            ),
        )

    print(
        "Held-out 23-dataset split used:",
        False,
    )
    print(
        "Report:",
        report_path,
    )
    print(
        "Phase 8 evaluation: SUCCESS"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
