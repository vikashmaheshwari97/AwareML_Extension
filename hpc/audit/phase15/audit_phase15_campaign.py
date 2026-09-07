from __future__ import annotations

import argparse
from pathlib import Path

from hpc.production.phase15.common import (
    EXPECTED_CASE_COUNT,
    EXPECTED_MODEL,
    EXPECTED_MODEL_DIGEST,
    EXPECTED_OLLAMA_VERSION,
    EXPECTED_PROMPT_VERSION,
    controlled_cases,
    expected_task_rows,
    find_project_root,
    git_commit,
    git_is_dirty,
    prompt_version,
    protocol_sha256,
    read_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()

    root = find_project_root()
    campaign_root = Path(args.campaign_root).resolve()
    manifest = read_json(
        campaign_root / "campaign_manifest.json"
    )

    checks = {}
    cases = controlled_cases()
    rows = expected_task_rows(cases)

    checks["case_count_24"] = (
        len(cases) == EXPECTED_CASE_COUNT
        and manifest.get("case_count") == EXPECTED_CASE_COUNT
    )
    checks["task_rows_exact"] = manifest.get("tasks") == rows
    checks["prompt_v4"] = (
        prompt_version() == EXPECTED_PROMPT_VERSION
        and manifest.get("prompt_version")
        == EXPECTED_PROMPT_VERSION
    )
    checks["model_tag"] = (
        manifest.get("model_lock", {}).get("model")
        == EXPECTED_MODEL
    )
    checks["model_digest"] = (
        manifest.get("model_lock", {}).get("digest")
        == EXPECTED_MODEL_DIGEST
    )
    checks["ollama_version"] = (
        manifest.get("model_lock", {}).get("ollama_version")
        == EXPECTED_OLLAMA_VERSION
    )
    checks["protocol_sha"] = (
        manifest.get("protocol_sha256")
        == protocol_sha256(root)
    )
    checks["git_commit"] = (
        manifest.get("git_commit") == git_commit(root)
    )
    checks["git_clean"] = not git_is_dirty(root)

    print("=" * 92)
    print("Phase-15 HPC campaign audit")
    print("=" * 92)
    failed = []
    for name, ok in checks.items():
        print("{:<60} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)

    if failed:
        raise SystemExit(
            "Campaign audit FAILED: " + ", ".join(failed)
        )

    print("=" * 92)
    print("Campaign audit: PASS")


if __name__ == "__main__":
    main()
