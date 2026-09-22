from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from awareml.recommender.v2_service import V2Recommender  # noqa: E402

from phase18_common import (  # noqa: E402
    DATASET_DIR,
    EXPECTED_DATASETS,
    PREFLIGHT_DIR,
    active_recommender_manifest,
    audit_datasets,
    audit_summary,
    current_git_head,
    load_inventory,
    resolution_template,
    sha256_file,
    write_json_atomic,
)


def env_candidate(name: str, windows: str, posix: str) -> tuple[bool, str]:
    override = os.getenv(name, "").strip()
    if override:
        p = Path(override).expanduser()
        return p.exists(), str(p)
    candidates = [ROOT / windows, ROOT / posix]
    for p in candidates:
        if p.exists():
            return True, str(p)
    return False, " | ".join(str(p) for p in candidates)


def _normalized_filename(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _validate_local_filenames(inventory) -> list[str]:
    if not DATASET_DIR.exists():
        raise RuntimeError(f"Testing dataset directory does not exist: {DATASET_DIR}")

    expected = [str(x) for x in inventory["filename"].tolist()]
    actual = sorted(p.name for p in DATASET_DIR.glob("*.csv") if p.is_file())
    actual_set = set(actual)
    missing = [name for name in expected if name not in actual_set]

    if missing:
        by_normalized = {}
        for name in actual:
            by_normalized.setdefault(_normalized_filename(name), []).append(name)
        lines = []
        for name in missing:
            near = by_normalized.get(_normalized_filename(name), [])
            if near:
                lines.append(f"  {name}  (near local match: {', '.join(near)})")
            else:
                lines.append(f"  {name}")
        raise RuntimeError(
            "Missing testing datasets (exact filename match required):\n" + "\n".join(lines)
        )

    return sorted(set(actual) - set(expected))


def main() -> None:
    print("=" * 76)
    print("AwareML Phase 18 preflight validation — 31-dataset task policy v2.5")
    print("=" * 76)
    print("Git HEAD:", current_git_head())
    print("Dataset directory:", DATASET_DIR)

    inventory = load_inventory()
    extras = _validate_local_filenames(inventory)
    print(f"Exact local filename check: PASS ({len(inventory)} declared CSVs found)")
    if extras:
        print("Extra CSVs in Testing Datasets (ignored by Phase 18):", ", ".join(extras))

    audit = audit_datasets(DATASET_DIR)
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)

    audit_path = PREFLIGHT_DIR / "dataset_audit.tsv"
    audit.to_csv(audit_path, sep="\t", index=False)

    summary = audit_summary(audit)
    summary_path = PREFLIGHT_DIR / "dataset_audit_summary.json"
    write_json_atomic(summary_path, summary)

    resolution = resolution_template(audit)
    resolution_path = PREFLIGHT_DIR / "dataset_resolution_template.tsv"
    resolution.to_csv(resolution_path, sep="\t", index=False)

    if len(audit) != EXPECTED_DATASETS:
        raise RuntimeError(f"Dataset audit produced {len(audit)} rows; expected {EXPECTED_DATASETS}.")

    missing = audit[~audit["exists"].astype(bool)]
    if not missing.empty:
        print()
        print("INPUT FILE STATUS: FAIL")
        print(missing[["dataset_id", "filename", "preflight_reason"]].to_string(index=False))
        raise SystemExit(2)

    overlap_sources = {
        str(value).strip()
        for value in audit["training_overlap_sources"].tolist()
        if str(value).strip()
    }
    if not overlap_sources:
        raise RuntimeError(
            "Could not verify held-out/development separation because no local V2 development snapshot "
            "could be read from data/meta/snapshots/."
        )

    model_manifest = active_recommender_manifest()
    if model_manifest is None:
        raise RuntimeError(
            "Frozen V2 recommender is not active. Expected data/meta/active_recommender_v2.txt "
            "to point to an existing manifest."
        )
    # Loads the bundle and verifies model checksums without fitting or predicting.
    V2Recommender(root=ROOT)

    evo_ok, evo_where = env_candidate(
        "AWAREML_EVO_PYTHON", ".venv-evo/Scripts/python.exe", ".venv-evo/bin/python"
    )
    oaml_ok, oaml_where = env_candidate(
        "AWAREML_OAML_PYTHON", ".venv-oaml/Scripts/python.exe", ".venv-oaml/bin/python"
    )

    print("Inventory: PASS (31 files declared; adult_binary_stream intentionally excluded)")
    print("Files present: PASS")
    print("Development separation evidence: CHECKED against {}".format(", ".join(sorted(overlap_sources))))
    print("Active V2 model bundle: CHECKSUM PASS")
    print("Active V2 model manifest:", model_manifest.relative_to(ROOT))
    print("Active V2 manifest SHA256:", sha256_file(model_manifest))
    print("EvoAutoML isolated environment:", "FOUND" if evo_ok else "NOT FOUND", evo_where)
    print("OAML isolated environment:", "FOUND" if oaml_ok else "NOT FOUND", oaml_where)

    display_cols = [
        "dataset_id",
        "cohort",
        "resolved_target",
        "source_target_unique",
        "source_task_assessment",
        "task_policy",
        "evaluation_target",
        "evaluation_target_unique",
        "preflight_status",
        "evaluation_role",
        "primary_heldout_eligible",
    ]
    print()
    print("=" * 76)
    print("COMPLETE 31-DATASET TASK-POLICY AUDIT")
    print("=" * 76)
    print(audit[display_cols].to_string(index=False))

    overlaps = audit[
        audit["overlap_with_training_id"].astype(bool)
        | audit["overlap_with_training_sha256"].astype(bool)
    ]
    if not overlaps.empty:
        print()
        print("DEVELOPMENT-OVERLAP DATASETS — EXCLUDED FROM PRIMARY HELD-OUT CLAIM")
        cols = [
            "dataset_id",
            "filename",
            "sha256",
            "training_overlap_matched_dataset_ids",
            "preflight_reason",
        ]
        print(overlaps[cols].to_string(index=False))

    target_blockers = audit[~audit["primary_heldout_eligible"].astype(bool)]
    if not target_blockers.empty:
        print()
        print("TASK-POLICY ITEMS REQUIRING RESOLUTION")
        cols = [
            "dataset_id",
            "filename",
            "resolved_target",
            "target_unique",
            "source_target_dtype",
            "source_task_assessment",
            "task_policy",
            "evaluation_target",
            "evaluation_target_unique",
            "preflight_status",
        ]
        print(target_blockers[cols].to_string(index=False))

    eligible = int(audit["primary_heldout_eligible"].astype(bool).sum())
    unresolved = EXPECTED_DATASETS - eligible
    print()
    print("Dataset audit summary:", json.dumps(summary, indent=2, ensure_ascii=False))
    print("Audit file:", audit_path)
    print("Resolution template:", resolution_path)
    print()
    print("=" * 76)
    if unresolved:
        print("PRE-FREEZE STATUS: BLOCKED")
        print(f"Protocol-ready datasets: {eligible}/{EXPECTED_DATASETS}")
        print(f"Datasets still failing protocol checks: {unresolved}")
        print("Task transformations are explicit in the inventory and audit; no transformation is outcome-driven.")
        print("Do NOT run --freeze yet.")
    else:
        print("PRE-FREEZE STATUS: PASS — all 31 datasets satisfy the frozen task-policy and separation rules")
    print("=" * 76)
    print("No AutoML framework was executed by this preflight.")


if __name__ == "__main__":
    main()
