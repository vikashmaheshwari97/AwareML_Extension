from __future__ import annotations

import argparse
from pathlib import Path

from hpc.production.phase15.common import (
    EXPECTED_MODEL,
    EXPECTED_MODEL_DIGEST,
    EXPECTED_OLLAMA_VERSION,
    GENERATION_OPTIONS,
    controlled_cases,
    expected_task_rows,
    find_project_root,
    git_commit,
    git_diff_summary,
    git_is_dirty,
    now_utc,
    prompt_version,
    protocol_sha256,
    stamp_utc,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument(
        "--output-root",
        default="artifacts/phase15/hpc_runs",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    root = find_project_root()
    dirty = git_is_dirty(root)
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "Refusing to prepare final Phase-15 campaign from a dirty git "
            "working tree.\n" + git_diff_summary(root)
        )

    cases = controlled_cases()
    campaign_id = (
        args.campaign_id
        or "phase15_llama3_8b_{}".format(stamp_utc())
    )
    campaign_root = root / args.output_root / campaign_id
    if campaign_root.exists():
        raise SystemExit(
            "Campaign already exists; refusing to overwrite: {}"
            .format(campaign_root)
        )

    (campaign_root / "cases").mkdir(parents=True)
    (campaign_root / "logs").mkdir(parents=True)
    (campaign_root / "reduced").mkdir(parents=True)

    manifest = {
        "schema_version": "phase15_hpc_campaign_v1",
        "campaign_id": campaign_id,
        "created_utc": now_utc(),
        "project_root": str(root),
        "git_commit": git_commit(root),
        "git_dirty": dirty,
        "protocol_sha256": protocol_sha256(root),
        "case_count": len(cases),
        "model_lock": {
            "model": EXPECTED_MODEL,
            "digest": EXPECTED_MODEL_DIGEST,
            "ollama_version": EXPECTED_OLLAMA_VERSION,
        },
        "prompt_version": prompt_version(),
        "generation_options": dict(GENERATION_OPTIONS),
        "tasks": expected_task_rows(cases),
        "status": "prepared",
    }
    write_json(campaign_root / "campaign_manifest.json", manifest)

    lines = [
        "task_id\tcase_id\tdataset_id\tsource_stage\tsource_name\tcase_sha256"
    ]
    for row in manifest["tasks"]:
        lines.append(
            "{task_id}\t{case_id}\t{dataset_id}\t{source_stage}\t"
            "{source_name}\t{case_sha256}".format(**row)
        )
    (campaign_root / "task_manifest.tsv").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print("Phase-15 HPC campaign prepared")
    print("=" * 96)
    print("Campaign:", campaign_id)
    print("Root:", campaign_root)
    print("Cases:", len(cases))
    print("Git commit:", manifest["git_commit"])
    print("Protocol SHA256:", manifest["protocol_sha256"])
    print("Model:", EXPECTED_MODEL)
    print("Model digest:", EXPECTED_MODEL_DIGEST)
    print("Ollama:", EXPECTED_OLLAMA_VERSION)
    print("Prompt:", manifest["prompt_version"])
    print("Generation:", GENERATION_OPTIONS)
    print()
    print(str(campaign_root))


if __name__ == "__main__":
    main()
