from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "phase8"
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


def verify_sidecar(path: Path):
    sidecar = Path(
        str(path) + ".sha256"
    )
    if not sidecar.exists():
        raise RuntimeError(
            "Missing checksum sidecar: {}".format(
                sidecar
            )
        )
    expected = (
        sidecar
        .read_text(
            encoding="utf-8"
        )
        .strip()
        .split()[0]
    )
    actual = sha256_file(path)
    if expected != actual:
        raise RuntimeError(
            "Checksum mismatch: {}".format(
                path.name
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-ollama",
        action="store_true",
    )
    args = parser.parse_args()

    report_path = (
        OUTPUT_DIR
        / "phase8_faithfulness_report.json"
    )
    det_path = (
        OUTPUT_DIR
        / "deterministic_faithfulness_cases.parquet"
    )
    det_summary_path = (
        OUTPUT_DIR
        / "deterministic_faithfulness_summary.json"
    )

    for path in [
        report_path,
        det_path,
        det_summary_path,
    ]:
        if not path.exists():
            raise RuntimeError(
                "Missing Phase-8 artifact: {}".format(
                    path
                )
            )
        verify_sidecar(path)

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    if report.get("status") != "pass":
        raise RuntimeError(
            "Phase-8 report is not PASS."
        )
    if report.get(
        "held_out_23_dataset_split_used"
    ) is not False:
        raise RuntimeError(
            "Held-out test split policy failed."
        )

    cases = pd.read_parquet(
        det_path
    )

    if cases[
        "dataset_id"
    ].nunique() != 47:
        raise RuntimeError(
            "Deterministic Phase-8 evaluation "
            "must cover 47 development datasets."
        )

    expected_scenarios = {
        "accuracy_evidence_flip",
        "runtime_evidence_flip",
        "energy_evidence_flip",
        "co2_evidence_flip",
        "sustainability_joint_evidence_flip",
    }
    if set(
        cases["scenario"]
    ) != expected_scenarios:
        raise RuntimeError(
            "Counterfactual scenario set is incomplete."
        )

    expected_rows = (
        47
        * len(expected_scenarios)
    )
    if len(cases) != expected_rows:
        raise RuntimeError(
            "Expected {} deterministic case rows; found {}.".format(
                expected_rows,
                len(cases),
            )
        )

    bounded_columns = [
        "grounding_validity",
        "decision_alignment",
        "attribution_alignment",
        "counterfactual_sensitivity_mean",
        "irrelevant_invariance",
        "evidence_fidelity_score",
        "changed_objective_cited",
        "new_winner_acknowledged",
        "citation_change",
        "counterfactual_sensitivity",
    ]
    for col in bounded_columns:
        if not cases[
            col
        ].between(
            0.0,
            1.0,
        ).all():
            raise RuntimeError(
                "{} outside [0,1].".format(
                    col
                )
            )

    ollama = report.get(
        "ollama",
        {}
    )
    if args.require_ollama:
        if (
            ollama.get("status")
            != "pass"
        ):
            raise RuntimeError(
                "Live Ollama faithfulness evaluation is required."
            )

        ollama_path = (
            OUTPUT_DIR
            / "ollama_faithfulness_cases.parquet"
        )
        ollama_summary = (
            OUTPUT_DIR
            / "ollama_faithfulness_summary.json"
        )
        for path in [
            ollama_path,
            ollama_summary,
        ]:
            if not path.exists():
                raise RuntimeError(
                    "Missing live Ollama artifact: {}".format(
                        path
                    )
                )
            verify_sidecar(
                path
            )

        live = pd.read_parquet(
            ollama_path
        )
        if live[
            "dataset_id"
        ].nunique() < 1:
            raise RuntimeError(
                "No live Ollama datasets were evaluated."
            )

    print("=" * 72)
    print(
        "AwareML Phase 8 validation: PASS"
    )
    print("=" * 72)
    print(
        "Development datasets:",
        cases[
            "dataset_id"
        ].nunique(),
    )
    print(
        "Counterfactual scenarios:",
        len(
            expected_scenarios
        ),
    )
    print(
        "Deterministic case rows:",
        len(cases),
    )
    print(
        "Mean deterministic AEF:",
        report[
            "deterministic"
        ][
            "mean_evidence_fidelity_score"
        ],
    )
    print(
        "Ollama status:",
        ollama.get(
            "status"
        ),
    )
    print(
        "Held-out 23-dataset split used:",
        False,
    )
    print(
        "All checksums: PASS"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
