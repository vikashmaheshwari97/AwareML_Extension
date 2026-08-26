from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.experiments.registry import load_dataset_registry


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze a dataset split from a newline-separated dataset-id file.")
    ap.add_argument("--ids", required=True, help="Text file containing one dataset_id per line")
    ap.add_argument("--output", required=True, help="Output YAML manifest")
    ap.add_argument("--manifest-id", required=True)
    ap.add_argument("--purpose", choices=["meta_train", "heldout_test", "development", "audit"], required=True)
    ap.add_argument("--registry", default="data/meta/registry/datasets_v1_47.yaml")
    args = ap.parse_args()

    ids = [x.strip() for x in Path(args.ids).read_text(encoding="utf-8").splitlines() if x.strip() and not x.lstrip().startswith("#")]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate dataset ids found; refusing to freeze manifest.")
    registry = load_dataset_registry(args.registry)
    missing = sorted(set(ids) - set(registry.by_id))
    if missing:
        raise SystemExit("Unknown dataset ids: " + ", ".join(missing))
    payload = {
        "schema_version": "1.0",
        "manifest_id": args.manifest_id,
        "purpose": args.purpose,
        "expected_count": len(ids),
        "frozen": True,
        "dataset_ids": ids,
        "notes": "Frozen by scripts/freeze_dataset_manifest.py; do not edit in place. Create a new manifest version instead.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Frozen {len(ids)} datasets -> {out}")


if __name__ == "__main__":
    main()
