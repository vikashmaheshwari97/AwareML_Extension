from __future__ import annotations

import argparse
from pathlib import Path

from hpc.production.phase15.common import read_json, sha256_file


def success_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        marker = read_json(path)
        result = Path(marker["result_path"])
        if not result.is_absolute():
            result = path.parent / result
        if not result.exists():
            return False
        if sha256_file(result) != marker.get("result_sha256"):
            return False
        payload = read_json(result)
        return bool(payload.get("complete"))
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--print-array", action="store_true")
    parser.add_argument("--write-file", default=None)
    args = parser.parse_args()

    campaign_root = Path(args.campaign_root).resolve()
    manifest = read_json(
        campaign_root / "campaign_manifest.json"
    )

    pending = []
    for row in manifest["tasks"]:
        task_id = int(row["task_id"])
        case_id = row["case_id"]
        case_dir = (
            campaign_root
            / "cases"
            / "task_{:03d}__{}".format(task_id, case_id)
        )
        if not success_valid(case_dir / "SUCCESS.json"):
            pending.append(task_id)

    array = ",".join(str(value) for value in pending)

    if args.write_file:
        Path(args.write_file).write_text(
            "\n".join(str(value) for value in pending)
            + ("\n" if pending else ""),
            encoding="utf-8",
        )

    if args.print_array:
        print(array)
        return

    print("Campaign:", campaign_root)
    print("Successful:", len(manifest["tasks"]) - len(pending))
    print("Pending/failed:", len(pending))
    print("Task IDs:", array if array else "<none>")


if __name__ == "__main__":
    main()
