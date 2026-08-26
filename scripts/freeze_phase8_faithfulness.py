from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "phase8"
)
DATA_DIR = (
    ROOT
    / "data"
    / "llm"
    / "faithfulness_v1"
)
ACTIVE = (
    ROOT
    / "data"
    / "llm"
    / "active_faithfulness.txt"
)
PHASE7_MANIFEST = (
    ROOT
    / "data"
    / "llm"
    / "copilot_v1"
    / "manifest.json"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-offline",
        action="store_true",
        help=(
            "Allow freezing without a live Ollama faithfulness "
            "evaluation. This is intended for development only."
        ),
    )
    args = parser.parse_args()

    report_path = (
        OUTPUT_DIR
        / "phase8_faithfulness_report.json"
    )
    if not report_path.exists():
        raise RuntimeError(
            "Run Phase-8 evaluation first."
        )
    if not PHASE7_MANIFEST.exists():
        raise RuntimeError(
            "Frozen Phase-7 Copilot manifest is missing."
        )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )
    if report.get("status") != "pass":
        raise RuntimeError(
            "Phase-8 report is not PASS."
        )

    ollama = report.get(
        "ollama",
        {}
    )
    if (
        not args.allow_offline
        and ollama.get("status")
        != "pass"
    ):
        raise RuntimeError(
            "Phase 8 freeze requires a live Ollama faithfulness "
            "evaluation. Re-run with --ollama --require-ollama."
        )

    if report.get(
        "held_out_23_dataset_split_used"
    ) is not False:
        raise RuntimeError(
            "Held-out test split policy failed."
        )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_names = [
        "phase8_faithfulness_report.json",
        "deterministic_faithfulness_cases.parquet",
        "deterministic_faithfulness_summary.json",
    ]

    if ollama.get(
        "status"
    ) == "pass":
        artifact_names.extend(
            [
                "ollama_faithfulness_cases.parquet",
                "ollama_faithfulness_summary.json",
            ]
        )

    artifacts = {}
    for name in artifact_names:
        path = OUTPUT_DIR / name
        if not path.exists():
            raise RuntimeError(
                "Missing Phase-8 artifact: {}".format(
                    name
                )
            )
        artifacts[name] = {
            "sha256": sha256_file(
                path
            ),
        }

    manifest = {
        "schema_version": "1.0",
        "phase": "8",
        "release_id": (
            "awareml-faithfulness-v1"
        ),
        "status": "frozen",
        "frozen_at_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),
        "method": (
            "AwareML Evidence Fidelity (AEF)"
        ),
        "scientific_scope": {
            "counterfactual_evidence_tests": True,
            "rationale_sensitivity": True,
            "citation_grounding": True,
            "decision_rationale_alignment": True,
            "external_objective_influence": True,
            "internal_llm_attention_attribution": False,
            "pe_lrp": False,
            "claim": (
                "FaithLM/Serum-inspired external evidence "
                "faithfulness evaluation; no claim of internal "
                "Ollama attribution access."
            ),
        },
        "phase7_manifest": {
            "path": (
                "data/llm/copilot_v1/manifest.json"
            ),
            "sha256": sha256_file(
                PHASE7_MANIFEST
            ),
        },
        "evaluation": {
            "development_datasets": 47,
            "held_out_23_dataset_split_used": False,
            "ollama": ollama,
        },
        "artifacts": artifacts,
    }

    manifest_path = (
        DATA_DIR
        / "manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    Path(
        str(manifest_path) + ".sha256"
    ).write_text(
        "{}  {}\n".format(
            sha256_file(
                manifest_path
            ),
            manifest_path.name,
        ),
        encoding="utf-8",
    )

    ACTIVE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    ACTIVE.write_text(
        "faithfulness_v1/manifest.json\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print(
        "AwareML Phase 8 freeze: SUCCESS"
    )
    print("=" * 72)
    print(
        "Active marker:",
        ACTIVE,
    )
    print(
        "Manifest:",
        manifest_path,
    )
    print(
        "SHA256:",
        sha256_file(
            manifest_path
        ),
    )
    print(
        "Live Ollama evaluation:",
        ollama.get(
            "status"
        ),
    )
    print(
        "Held-out 23-dataset split used:",
        False,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
