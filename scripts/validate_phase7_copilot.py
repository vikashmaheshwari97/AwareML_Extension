from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.llm import (
    CopilotService,
    OllamaClient,
    ReviewStore,
)
from awareml.recommender.v2_service import (
    V2Recommender,
)


ARTIFACT_DIR = (
    ROOT
    / "artifacts"
    / "phase7"
)
SNAPSHOT_DIR = (
    ROOT
    / "data"
    / "meta"
    / "snapshots"
)

FORBIDDEN_RAW_KEYS = {
    "raw_rows",
    "rows_raw",
    "dataframe",
    "raw_dataframe",
    "dataset_rows",
    "records_raw",
    "participant_rows",
}


def profile_from_training_row(
    row,
):
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


def find_forbidden_keys(
    value,
    path="root",
):
    """Find exact forbidden raw-data keys.

    Important: do not use substring search. A safe metadata field such as
    `raw_dataset_rows_included=False` contains the text `dataset_rows`, but it
    is not itself a raw-row payload.
    """
    hits = []

    if isinstance(value, dict):
        for key, child in value.items():
            key_str = str(key)
            next_path = (
                "{}.{}".format(
                    path,
                    key_str,
                )
            )
            if (
                key_str.lower()
                in FORBIDDEN_RAW_KEYS
            ):
                hits.append(
                    next_path
                )
            hits.extend(
                find_forbidden_keys(
                    child,
                    next_path,
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(
            value
        ):
            hits.extend(
                find_forbidden_keys(
                    child,
                    "{}[{}]".format(
                        path,
                        index,
                    ),
                )
            )

    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ollama",
        action="store_true",
        help=(
            "Also execute one live local-Ollama "
            "goal parse and grounded answer."
        ),
    )
    parser.add_argument(
        "--require-ollama",
        action="store_true",
        help=(
            "Fail if the optional Ollama test "
            "cannot run successfully."
        ),
    )
    args = parser.parse_args()

    train_path = (
        SNAPSHOT_DIR
        / "recommender_train_v2.parquet"
    )
    if not train_path.exists():
        raise RuntimeError(
            "Phase 6 training snapshot is missing."
        )

    frame = pd.read_parquet(
        train_path
    )
    profile = profile_from_training_row(
        frame.iloc[0]
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    review_path = (
        ARTIFACT_DIR
        / "review_validation.jsonl"
    )
    if review_path.exists():
        review_path.unlink()

    recommender = V2Recommender(
        root=ROOT
    )
    service = CopilotService(
        recommender=recommender,
        review_store=ReviewStore(
            review_path
        ),
    )

    goal = (
        "I need a highly accurate streaming classifier "
        "that adapts quickly to concept drift, keeps energy "
        "consumption moderate, is fair, and provides "
        "understandable explanations."
    )

    proposal, ranked, evidence, meta = (
        service.propose_from_profile(
            goal,
            profile,
            sensitive_attribute=None,
            use_llm=False,
        )
    )

    if len(ranked) != 5:
        raise RuntimeError(
            "Copilot must receive five framework candidates."
        )
    if not proposal.requires_human_review:
        raise RuntimeError(
            "Human review gate is disabled."
        )
    if (
        proposal
        .interpretation
        .fairness_required
        is not True
    ):
        raise RuntimeError(
            "Fairness goal was not recognized."
        )
    if set(
        proposal
        .interpretation
        .primary_weights
        .normalized_dict()
    ) != {
        "accuracy",
        "runtime",
        "energy",
        "co2",
    }:
        raise RuntimeError(
            "Primary objective separation is invalid."
        )

    # Privacy gate: inspect exact object keys, not serialized substrings.
    #
    # The original validator checked whether the string "dataset_rows" appeared
    # anywhere in JSON text. That incorrectly rejected the safe metadata key:
    #
    #   raw_dataset_rows_included: false
    #
    # Here we only reject exact forbidden raw-payload keys.
    prompt_payload = (
        evidence.prompt_payload()
    )
    forbidden_hits = (
        find_forbidden_keys(
            prompt_payload
        )
    )
    if forbidden_hits:
        raise RuntimeError(
            "Raw-row privacy gate failed at exact key(s): {}".format(
                forbidden_hits
            )
        )

    privacy = (
        prompt_payload.get(
            "privacy"
        )
        or {}
    )
    if (
        privacy.get(
            "raw_dataset_rows_included"
        )
        is not False
    ):
        raise RuntimeError(
            "Privacy metadata must state raw_dataset_rows_included=False."
        )
    if (
        privacy.get(
            "raw_participant_rows_included"
        )
        is not False
    ):
        raise RuntimeError(
            "Privacy metadata must state raw_participant_rows_included=False."
        )

    review = service.review(
        proposal,
        decision=(
            "approved_with_edits"
        ),
        edits={
            "window_size": 750,
        },
        note=(
            "Validation edit to exercise "
            "the human review/config diff path."
        ),
        persist=True,
    )

    if not review.config_diff:
        raise RuntimeError(
            "Configuration diff was not recorded."
        )
    if (
        review.final_config.window_size
        != 750
    ):
        raise RuntimeError(
            "Human edit was not applied."
        )
    if not review_path.exists():
        raise RuntimeError(
            "Review audit record was not persisted."
        )

    during_result = {
        "framework": (
            proposal
            .proposed_config
            .framework
        ),
        "accuracy": 0.82,
        "f1_macro": 0.80,
        "samples": 2000,
        "drift_events": [
            1000,
            1800,
        ],
        "drift_summary": {
            "n_drift_events": 2,
            "recovery_rate": 0.5,
        },
        "points": [
            {
                "sample": 2000,
                "accuracy": 0.82,
                "f1_macro": 0.80,
                "rolling_accuracy": 0.79,
                "rolling_f1_macro": 0.77,
            }
        ],
        "fairness": {
            "status": (
                "insufficient_support"
            )
        },
        "explainability": {
            "status": "ok",
            "method": "permutation",
            "feature_importance": {
                "f1": 0.7,
                "f2": 0.3,
            },
        },
        "energy_kwh": 0.001,
        "co2_kg": 0.0004,
    }

    during = service.during_evidence(
        during_result
    )
    answer = service.ask(
        "How many drift events have occurred?",
        during,
        use_llm=False,
    )
    if (
        "evidence.during.drift.count"
        not in answer.evidence_keys
    ):
        raise RuntimeError(
            "During-run grounded evidence key missing."
        )

    post_results = []
    for _, row in ranked.iterrows():
        post_results.append(
            {
                "framework": row[
                    "framework"
                ],
                "accuracy": row[
                    "accuracy"
                ],
                "f1_macro": (
                    row["accuracy"]
                ),
                "runtime_sec": row[
                    "runtime"
                ],
                "energy_kwh": row[
                    "energy"
                ],
                "co2_kg": row[
                    "co2"
                ],
                "drift_summary": {},
                "fairness": {},
                "explainability": {},
            }
        )

    after = service.after_evidence(
        post_results
    )
    post_answer = service.ask(
        "Which framework has the highest accuracy?",
        after,
        use_llm=False,
    )
    if not post_answer.evidence_keys:
        raise RuntimeError(
            "Post-run answer is not evidence keyed."
        )

    what_if, _ = (
        service.what_if_weights(
            profile,
            {
                "accuracy": 0.10,
                "runtime": 0.10,
                "energy": 0.70,
                "co2": 0.10,
            },
        )
    )
    if len(what_if) != 5:
        raise RuntimeError(
            "Preference counterfactual did not rank five frameworks."
        )

    ollama_result = {
        "requested": bool(
            args.ollama
            or args.require_ollama
        ),
        "status": "not_requested",
    }

    if (
        args.ollama
        or args.require_ollama
    ):
        client = OllamaClient()
        status = client.status()
        ollama_result["status_detail"] = status

        if status["reachable"]:
            live_service = (
                CopilotService(
                    recommender=recommender
                )
            )
            live_proposal, _, live_evidence, live_meta = (
                live_service
                .propose_from_profile(
                    goal,
                    profile,
                    use_llm=True,
                )
            )
            live_answer = (
                live_service.ask(
                    (
                        "Why is the top framework "
                        "recommended?"
                    ),
                    live_evidence,
                    use_llm=True,
                )
            )

            ollama_result.update(
                {
                    "status": "pass",
                    "goal_parse_source": (
                        live_meta[
                            "goal_parse"
                        ].get("source")
                    ),
                    "answer_source": (
                        live_answer.source
                    ),
                    "model": (
                        live_answer.model
                        or status.get(
                            "resolved_model"
                        )
                    ),
                }
            )
        else:
            ollama_result[
                "status"
            ] = "unreachable"
            if args.require_ollama:
                raise RuntimeError(
                    "Ollama live validation was required but is unreachable."
                )

    report = {
        "schema_version": "1.0",
        "phase": "7",
        "status": "pass",
        "validated_at_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),
        "checks": {
            "phase6_recommender_integration": "pass",
            "goal_to_configuration": "pass",
            "four_primary_objectives_separate": "pass",
            "human_review_gate": "pass",
            "configuration_diff": "pass",
            "review_audit": "pass",
            "before_run_grounding": "pass",
            "during_run_grounding": "pass",
            "post_run_grounding": "pass",
            "preference_counterfactual": "pass",
            "raw_rows_sent_to_llm": False,
        },
        "ollama": ollama_result,
        "proposal_framework": (
            proposal
            .proposed_config
            .framework
        ),
        "review_diff_count": len(
            review.config_diff
        ),
        "during_answer": answer.text,
        "post_answer": post_answer.text,
    }

    report_path = (
        ARTIFACT_DIR
        / "validation_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print(
        "AwareML Phase 7 validation: PASS"
    )
    print("=" * 72)
    print(
        "Proposal framework:",
        proposal.proposed_config.framework,
    )
    print(
        "Human review/config diff:",
        "PASS",
    )
    print(
        "Before/during/after grounding:",
        "PASS",
    )
    print(
        "Raw dataset rows sent to LLM:",
        False,
    )
    print(
        "Ollama validation:",
        ollama_result["status"],
    )
    print(
        "Report:",
        report_path,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
