from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional

from hpc.production.phase15.common import (
    EXPECTED_CASE_COUNT,
    now_utc,
    read_json,
    sha256_file,
    stamp_utc,
    write_json,
)


CORRECTNESS_METRICS = [
    "claim_precision",
    "supported_claim_rate",
    "numeric_correctness",
    "citation_validity",
    "citation_coverage",
    "decision_consistency",
    "unsupported_claim_rate",
    "contradiction_rate",
]

FAITHFULNESS_METRICS = [
    "faithfulness_score",
    "changed_evidence_acknowledged",
    "numeric_update_accuracy",
    "stale_claim_rate",
    "decision_update_consistency",
]


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return float(mean(clean)) if clean else None


def aggregate(rows, names):
    return {
        name: safe_mean(row.get(name) for row in rows)
        for name in names
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--strict-hardware", action="store_true")
    args = parser.parse_args()

    campaign_root = Path(args.campaign_root).resolve()
    campaign = read_json(
        campaign_root / "campaign_manifest.json"
    )

    correctness_rows = []
    faithfulness_rows = []
    failures = []
    runtime_rows = []
    full_case_results = []

    for task in campaign["tasks"]:
        task_id = int(task["task_id"])
        case_id = task["case_id"]
        case_dir = (
            campaign_root
            / "cases"
            / "task_{:03d}__{}".format(task_id, case_id)
        )
        success_path = case_dir / "SUCCESS.json"

        if not success_path.exists():
            failures.append(
                {
                    "task_id": task_id,
                    "case_id": case_id,
                    "reason": "missing_success_marker",
                }
            )
            continue

        try:
            success = read_json(success_path)
            result_path = Path(success["result_path"])
            if not result_path.is_absolute():
                result_path = case_dir / result_path
            if sha256_file(result_path) != success.get("result_sha256"):
                raise RuntimeError("result SHA256 mismatch")
            result = read_json(result_path)
            if not result.get("complete"):
                raise RuntimeError("result marked incomplete")

            full_case_results.append(result)

            report = result["correctness_reports"][0]
            c_metrics = report.get("metrics") or {}
            correctness_rows.append(
                {
                    "task_id": task_id,
                    "case_id": case_id,
                    "dataset_id": task["dataset_id"],
                    "source_stage": task["source_stage"],
                    "source_name": task["source_name"],
                    **{
                        name: c_metrics.get(name)
                        for name in CORRECTNESS_METRICS
                    },
                }
            )

            faith = result["faithfulness_records"][0]
            faithfulness_rows.append(
                {
                    "task_id": task_id,
                    "case_id": case_id,
                    "dataset_id": task["dataset_id"],
                    "source_stage": task["source_stage"],
                    "source_name": task["source_name"],
                    **{
                        name: faith.get(name)
                        for name in FAITHFULNESS_METRICS
                    },
                }
            )

            runtime_rows.append(
                {
                    "task_id": task_id,
                    "case_id": case_id,
                    "model": success.get("model"),
                    "model_digest": success.get("model_digest"),
                    "ollama_version": success.get("ollama_version"),
                    "prompt_version": success.get("prompt_version"),
                    "git_commit": success.get("git_commit"),
                    "protocol_sha256": success.get("protocol_sha256"),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "task_id": task_id,
                    "case_id": case_id,
                    "reason": "{}: {}".format(
                        type(exc).__name__,
                        exc,
                    ),
                }
            )

    complete = (
        len(correctness_rows) == EXPECTED_CASE_COUNT
        and len(faithfulness_rows) == EXPECTED_CASE_COUNT
        and not failures
    )

    if not complete and not args.allow_partial:
        print("Collection is incomplete.")
        print("Correctness:", len(correctness_rows))
        print("Faithfulness:", len(faithfulness_rows))
        print("Failures:", len(failures))
        for row in failures:
            print(row)
        raise SystemExit(2)

    runtime_keys = [
        "model",
        "model_digest",
        "ollama_version",
        "prompt_version",
        "git_commit",
        "protocol_sha256",
    ]
    runtime_consistency = {
        key: sorted(
            {
                str(row.get(key))
                for row in runtime_rows
                if row.get(key) is not None
            }
        )
        for key in runtime_keys
    }

    grouped_c = defaultdict(list)
    grouped_f = defaultdict(list)
    for row in correctness_rows:
        grouped_c[row["source_stage"]].append(row)
    for row in faithfulness_rows:
        grouped_f[row["source_stage"]].append(row)

    by_source = {}
    for source in sorted(set(grouped_c) | set(grouped_f)):
        by_source[source] = {
            "n_correctness": len(grouped_c[source]),
            "n_faithfulness": len(grouped_f[source]),
            "correctness": aggregate(
                grouped_c[source],
                CORRECTNESS_METRICS,
            ),
            "faithfulness": aggregate(
                grouped_f[source],
                FAITHFULNESS_METRICS,
            ),
        }

    summary = {
        "schema_version": "phase15_hpc_empirical_summary_v1",
        "created_utc": now_utc(),
        "campaign_id": campaign["campaign_id"],
        "complete": complete,
        "case_count_expected": EXPECTED_CASE_COUNT,
        "correctness_cases": len(correctness_rows),
        "faithfulness_cases": len(faithfulness_rows),
        "failure_count": len(failures),
        "overall": {
            "correctness": aggregate(
                correctness_rows,
                CORRECTNESS_METRICS,
            ),
            "faithfulness": aggregate(
                faithfulness_rows,
                FAITHFULNESS_METRICS,
            ),
        },
        "by_source": by_source,
        "runtime_consistency": runtime_consistency,
        "journal_ready_runtime_consistency": all(
            len(values) == 1
            for values in runtime_consistency.values()
        ),
        "note": (
            "Empirical result collection only. Review before any journal-facing "
            "claim; results are intentionally not auto-frozen."
        ),
    }

    stamp = stamp_utc()
    out = campaign_root / "reduced" / stamp
    out.mkdir(parents=True, exist_ok=False)

    write_json(out / "phase15_empirical_summary.json", summary)
    write_json(
        out / "phase15_empirical_full.json",
        {
            "campaign_manifest": campaign,
            "summary": summary,
            "case_results": full_case_results,
        },
    )
    write_json(out / "failures.json", failures)
    write_csv(out / "correctness_by_case.csv", correctness_rows)
    write_csv(out / "faithfulness_by_case.csv", faithfulness_rows)

    source_rows = []
    for source, values in by_source.items():
        row = {
            "source_stage": source,
            "n_correctness": values["n_correctness"],
            "n_faithfulness": values["n_faithfulness"],
        }
        row.update(
            {
                "correctness_" + key: value
                for key, value in values["correctness"].items()
            }
        )
        row.update(
            {
                "faithfulness_" + key: value
                for key, value in values["faithfulness"].items()
            }
        )
        source_rows.append(row)
    write_csv(out / "summary_by_source.csv", source_rows)

    files = [
        p
        for p in out.iterdir()
        if p.is_file()
    ]
    collection_manifest = {
        "schema_version": "phase15_hpc_collection_manifest_v1",
        "created_utc": now_utc(),
        "campaign_id": campaign["campaign_id"],
        "complete": complete,
        "files": {
            p.name: sha256_file(p)
            for p in files
        },
    }
    write_json(out / "collection_manifest.json", collection_manifest)

    print("=" * 96)
    print("Phase-15 HPC reduction complete")
    print("=" * 96)
    print("Campaign:", campaign["campaign_id"])
    print("Complete:", complete)
    print("Correctness cases:", len(correctness_rows))
    print("Faithfulness cases:", len(faithfulness_rows))
    print("Failures:", len(failures))
    print("Output:", out)
    print()
    print("Overall correctness:")
    for key, value in summary["overall"]["correctness"].items():
        print("  {:<32} {}".format(key, value))
    print("Overall faithfulness:")
    for key, value in summary["overall"]["faithfulness"].items():
        print("  {:<32} {}".format(key, value))


if __name__ == "__main__":
    main()
