from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


ROOT = Path.cwd()
PAYLOAD = ROOT / "payload"
BACKUP = (
    ROOT
    / ".phase14_final_freeze_backup"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


def sha256(path: Path):
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes():
    paths = [
        ROOT / "data/journal/objective_selection_benchmark_v1/frozen/manifest.json",
        ROOT / "data/journal/recommender_multiobjective_validation_v1/frozen/manifest.json",
        ROOT / "awareml/llm/objective_selection_v3.py",
        ROOT / "awareml/llm/objective_selection_v31.py",
        ROOT / "awareml/recommender/v2_service.py",
        ROOT / "awareml/recommender/v2_ranking.py",
        ROOT / "data/meta/models/recommender_v2/manifest.json",
        ROOT / "data/meta/snapshots/meta_logs_v2.json",
        ROOT / "data/meta/snapshots/recommender_train_v2.parquet",
    ]
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in paths
        if path.exists()
    }


def backup(path: Path):
    if not path.exists():
        return
    dest = BACKUP / path.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


def copy_payload():
    for source in PAYLOAD.rglob("*"):
        if not source.is_file():
            continue
        dest = ROOT / source.relative_to(PAYLOAD)
        backup(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def patch_pages_for_dataset_fingerprint():
    path = ROOT / "awareml/ui/pages.py"
    text = path.read_text(encoding="utf-8")
    original = text

    import_line = (
        "from awareml.analysis.repeatability_registry import "
        "canonical_dataframe_sha256\n"
    )
    if import_line not in text:
        anchor = "from awareml.types import ObjectiveWeights, RunConfig\n"
        if anchor not in text:
            raise RuntimeError("Could not locate Run Studio import anchor.")
        text = text.replace(anchor, anchor + import_line, 1)

    if '["dataset_content_sha256"] = canonical_dataframe_sha256(df)' not in text:
        anchor = '        df = _state().get("dataset")\n'
        block = (
            '        df = _state().get("dataset")\n'
            '        if isinstance(df, pd.DataFrame):\n'
            '            _state()["dataset_content_sha256"] = '
            'canonical_dataframe_sha256(df)\n'
        )
        if anchor not in text:
            raise RuntimeError("Could not locate active dataset state block.")
        text = text.replace(anchor, block, 1)

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(
            "Patched Run Studio: active dataset content SHA256 is stored "
            "for exact repeatability matching."
        )


def patch_fairness_winner_handling():
    path = ROOT / "awareml/ui_v2/pages_specialist.py"
    text = path.read_text(encoding="utf-8")
    original = text

    old = (
        '    cards = st.columns(4)\n'
        '    best = (\n'
        '        fair.sort_values("Comparable mean gap").iloc[0]\n'
        '        if fair["Comparable mean gap"].notna().any()\n'
        '        else None\n'
        '    )\n'
        '    cards[0].metric(\n'
        '        "Lowest comparable mean disparity",\n'
        '        best["Framework"] if best is not None else "N/A",\n'
        '        fmt(best["Comparable mean gap"], 3) if best is not None else None,\n'
        '    )\n'
    )

    new = (
        '    # Phase 14: preserve every raw fairness value, but do not declare a\n'
        '    # constant/near-constant predictor the fairness winner merely because\n'
        '    # some parity gaps collapse to zero.\n'
        '    prediction_status_by_framework = {\n'
        '        r.get("framework"): (\n'
        '            (r.get("fairness") or {}).get("prediction_behavior_status")\n'
        '            or "unavailable"\n'
        '        )\n'
        '        for r in results\n'
        '    }\n'
        '    fair["Prediction behavior"] = fair["Framework"].map(\n'
        '        prediction_status_by_framework\n'
        '    ).fillna("unavailable")\n'
        '    fair["Fairness winner eligibility"] = fair["Prediction behavior"].map(\n'
        '        lambda status: (\n'
        '            "EXCLUDED · degenerate predictions"\n'
        '            if str(status) in {"constant", "near_constant"}\n'
        '            else "ELIGIBLE"\n'
        '        )\n'
        '    )\n'
        '\n'
        '    eligible_fair = fair[\n'
        '        (~fair["Prediction behavior"].isin(["constant", "near_constant"]))\n'
        '        & fair["Comparable mean gap"].notna()\n'
        '    ].copy()\n'
        '\n'
        '    cards = st.columns(4)\n'
        '    best = (\n'
        '        eligible_fair.sort_values("Comparable mean gap").iloc[0]\n'
        '        if not eligible_fair.empty\n'
        '        else None\n'
        '    )\n'
        '    cards[0].metric(\n'
        '        "Lowest interpretable mean disparity",\n'
        '        best["Framework"] if best is not None else "N/A",\n'
        '        fmt(best["Comparable mean gap"], 3) if best is not None else None,\n'
        '    )\n'
        '\n'
        '    excluded_frameworks = fair.loc[\n'
        '        fair["Prediction behavior"].isin(["constant", "near_constant"]),\n'
        '        "Framework",\n'
        '    ].astype(str).tolist()\n'
        '    if excluded_frameworks:\n'
        '        st.warning(\n'
        '            "Fairness-winner claim excludes degenerate/near-degenerate "\n'
        '            "predictors: {}. Their raw fairness values remain visible for "\n'
        '            "auditability and are not changed.".format(\n'
        '                ", ".join(excluded_frameworks)\n'
        '            )\n'
        '        )\n'
    )

    if "Lowest interpretable mean disparity" not in text:
        if old not in text:
            raise RuntimeError(
                "Could not locate Fairness Lab winner block. "
                "No production fairness values were changed."
            )
        text = text.replace(old, new, 1)

    if '"Fairness winner eligibility",' not in text:
        anchor = (
            '        "Unavailable criteria",\n'
            '        "Status",\n'
            '        "Window N",\n'
        )
        replacement = (
            '        "Unavailable criteria",\n'
            '        "Prediction behavior",\n'
            '        "Fairness winner eligibility",\n'
            '        "Status",\n'
            '        "Window N",\n'
        )
        if anchor in text:
            text = text.replace(anchor, replacement, 1)

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(
            "Patched Fairness Lab: degenerate predictors remain visible "
            "but are excluded from the fairness-winner claim."
        )


def patch_analysis_exports():
    path = ROOT / "awareml/analysis/__init__.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    original = text

    line = (
        "from .repeatability_registry import "
        "PAPER_READY_MIN_REPETITIONS, canonical_dataframe_sha256\n"
    )
    if line not in text:
        text += "\n" + line

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print("Patched analysis exports.")


def migrate_legacy_repeatability():
    # Register old flat Phase-14 evidence without deleting or rewriting it.
    legacy_root = ROOT / "artifacts/phase14/repeatability"
    legacy_results = legacy_root / "phase14_repeated_results.json"
    legacy_manifest = legacy_root / "repeatability_manifest.json"

    if not legacy_results.exists() or not legacy_manifest.exists():
        print("No legacy flat repeatability artifact to migrate.")
        return

    try:
        manifest = json.loads(legacy_manifest.read_text(encoding="utf-8"))
        rows = json.loads(legacy_results.read_text(encoding="utf-8"))
        dataset_path = Path(str(manifest.get("dataset_path") or ""))

        if not dataset_path.exists():
            print(
                "Legacy artifact retained in place; source dataset path "
                "is unavailable, so exact content-hash migration was skipped."
            )
            return

        from awareml.analysis.repeatability_registry import (
            PAPER_READY_MIN_REPETITIONS,
            build_dataset_identity,
            canonical_dataframe_sha256,
            register_run,
        )

        df = pd.read_csv(dataset_path)
        content_hash = canonical_dataframe_sha256(df)
        identity = build_dataset_identity(
            dataset_name=manifest.get("dataset_name") or dataset_path.name,
            dataset_content_sha256=content_hash,
            target=manifest.get("target"),
            sensitive_attribute=manifest.get("sensitive_attribute"),
            positive_label=manifest.get("positive_label"),
        )

        legacy_stamp = str(
            manifest.get("created_utc")
            or datetime.now(timezone.utc).isoformat()
        )
        safe_stamp = (
            legacy_stamp.replace("-", "")
            .replace(":", "")
            .replace("+", "_")
            .replace(".", "_")
        )[:24]

        run_dir = (
            legacy_root
            / identity["directory_name"]
            / ("legacy_" + safe_stamp)
        )

        if run_dir.exists():
            print("Legacy repeatability artifact already migrated.")
            return

        run_dir.mkdir(parents=True, exist_ok=False)

        file_hash = (
            manifest.get("dataset_sha256")
            or manifest.get("dataset_file_sha256")
        )

        updated_rows = []
        for row in rows:
            row = dict(row)
            row["dataset_content_sha256"] = content_hash
            row["dataset_file_sha256"] = file_hash
            row["dataset_identity_key"] = identity["identity_key"]
            row["dataset_identity_directory"] = identity["directory_name"]

            provenance = dict(row.get("dataset_provenance") or {})
            provenance.update({
                "source_file_name": dataset_path.name,
                "source_sha256": file_hash,
                "content_sha256": content_hash,
                "dataset_identity_key": identity["identity_key"],
            })
            row["dataset_provenance"] = provenance
            updated_rows.append(row)

        migrated_manifest = dict(manifest)
        migrated_manifest.update({
            "schema_version": "phase14-repeatability-run-v2-migrated",
            "dataset_name": dataset_path.name,
            "dataset_content_sha256": content_hash,
            "dataset_file_sha256": file_hash,
            "dataset_identity_key": identity["identity_key"],
            "dataset_identity_directory": identity["directory_name"],
            "paper_ready_min_repetitions": PAPER_READY_MIN_REPETITIONS,
            "paper_ready": int(
                manifest.get("repetitions") or 0
            ) >= PAPER_READY_MIN_REPETITIONS,
            "migrated_from_legacy_flat_layout": True,
        })

        (run_dir / "phase14_repeated_results.json").write_text(
            json.dumps(
                updated_rows,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        (run_dir / "repeatability_manifest.json").write_text(
            json.dumps(
                migrated_manifest,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        for filename in ("repeatability_table.csv", "hardware_table.csv"):
            source = legacy_root / filename
            if source.exists():
                shutil.copy2(source, run_dir / filename)

        register_run(
            legacy_root,
            identity=identity,
            run_dir=run_dir,
            manifest=migrated_manifest,
        )

        print(
            "Migrated legacy flat repeatability evidence into the "
            "dataset-specific registry without deleting original files:"
        )
        print(" ", run_dir)

    except Exception as exc:
        print(
            "Legacy migration skipped safely: {}. "
            "Original artifact was not changed.".format(exc)
        )


def main():
    if not (ROOT / "app.py").exists() or not (ROOT / "awareml").exists():
        raise RuntimeError(
            "Run this installer from the AwareML_Extension project root."
        )
    if not PAYLOAD.exists():
        raise RuntimeError(
            "payload folder is missing. Extract the complete ZIP first."
        )

    before = protected_hashes()
    BACKUP.mkdir(parents=True, exist_ok=True)

    copy_payload()
    patch_pages_for_dataset_fingerprint()
    patch_fairness_winner_handling()
    patch_analysis_exports()
    migrate_legacy_repeatability()

    after = protected_hashes()
    changed = [
        key
        for key in before
        if before.get(key) != after.get(key)
    ]
    if changed:
        raise RuntimeError(
            "Protected frozen/model artifacts changed unexpectedly: {}"
            .format(changed)
        )

    print("=" * 104)
    print("AwareML Phase-14 Final Freeze Upgrade: APPLIED")
    print("=" * 104)
    print("Backup:", BACKUP)
    print()
    print("Installed:")
    print("  1. Dataset-specific repeatability registry")
    print("  2. Exact match by dataframe SHA256 + target + sensitive + positive label")
    print("  3. Timestamped run directories; previous evidence is never overwritten")
    print("  4. Sustainability Lab auto-loads the latest exact matching run")
    print("  5. Formal paper-ready repetition gate = 5")
    print("  6. Beginner-friendly no-evidence message + CLI command")
    print("  7. Per-run dataset file/content SHA256 provenance")
    print("  8. Degenerate predictors excluded only from fairness-winner claims")
    print("  9. Raw fairness values remain unchanged and auditable")
    print()
    print(
        "Protected Phase-12/13/V3/V3.1/Recommender-V2 "
        "artifacts unchanged: True"
    )
    print()
    print("Run next:")
    print(
        r"  pytest -q .\tests\test_phase14_hardening.py "
        r".\tests\test_phase14_integrated_labs.py "
        r".\tests\test_phase14_robustness_hotfix.py "
        r".\tests\test_phase14_final_freeze.py"
    )
    print(r"  python -m scripts.validate_phase14_hardening")
    print(r"  python -m scripts.validate_phase14_robustness_hotfix")
    print(r"  python -m scripts.validate_phase14_final_freeze")
    print("=" * 104)


if __name__ == "__main__":
    main()
