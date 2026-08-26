from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REPORT = (
    ROOT
    / "artifacts"
    / "phase7"
    / "validation_report.json"
)
PHASE6_RELEASE = (
    ROOT
    / "data"
    / "meta"
    / "models"
    / "recommender_v2"
    / "release_manifest.json"
)
LLM_DIR = (
    ROOT
    / "data"
    / "llm"
    / "copilot_v1"
)
ACTIVE = (
    ROOT
    / "data"
    / "llm"
    / "active_copilot.txt"
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
    if not REPORT.exists():
        raise RuntimeError(
            "Run validate_phase7_copilot.py first."
        )
    if not PHASE6_RELEASE.exists():
        raise RuntimeError(
            "Phase 6 release manifest is missing."
        )

    report = json.loads(
        REPORT.read_text(
            encoding="utf-8"
        )
    )
    if report.get("status") != "pass":
        raise RuntimeError(
            "Phase 7 validation report is not PASS."
        )

    checks = report.get(
        "checks",
        {},
    )
    if (
        checks.get(
            "raw_rows_sent_to_llm"
        )
        is not False
    ):
        raise RuntimeError(
            "Privacy gate is not satisfied."
        )

    LLM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "schema_version": "1.0",
        "phase": "7",
        "release_id": (
            "awareml-copilot-v1"
        ),
        "status": "frozen",
        "frozen_at_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),
        "capabilities": [
            "goal_to_configuration",
            "human_review_gate",
            "configuration_diff",
            "before_run_grounded_explanation",
            "during_run_grounded_chat",
            "post_run_grounded_chat",
            "preference_counterfactual",
            "ollama_local_first",
            "deterministic_fallback",
        ],
        "scientific_boundaries": {
            "llm_recommender_role": (
                "interpret human goals and explain evidence"
            ),
            "ml_recommender_role": (
                "predict accuracy/runtime/energy/co2 "
                "and empirically rank frameworks"
            ),
            "primary_objectives": [
                "accuracy",
                "runtime",
                "energy",
                "co2",
            ],
            "fairness_and_explainability": (
                "HCAI requirements/evidence, not silently "
                "mixed into the four-objective utility"
            ),
            "raw_dataset_rows_sent_to_llm": False,
            "23_dataset_test_split_used": False,
        },
        "phase6_release": {
            "path": (
                "data/meta/models/"
                "recommender_v2/"
                "release_manifest.json"
            ),
            "sha256": sha256_file(
                PHASE6_RELEASE
            ),
        },
        "validation_report": {
            "path": (
                "artifacts/phase7/"
                "validation_report.json"
            ),
            "sha256": sha256_file(
                REPORT
            ),
            "ollama": report.get(
                "ollama"
            ),
        },
    }

    manifest_path = (
        LLM_DIR
        / "manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    Path(
        str(manifest_path)
        + ".sha256"
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
        "copilot_v1/manifest.json\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print(
        "AwareML Phase 7 freeze: SUCCESS"
    )
    print("=" * 72)
    print(
        "Active Copilot marker:",
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
        "23-dataset held-out split touched:",
        False,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
