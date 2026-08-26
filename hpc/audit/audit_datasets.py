
from pathlib import Path

import hashlib

import csv

import yaml

import pandas as pd



ROOT = Path.home() / "AwareML_Extension"

MANIFEST = ROOT / "data/meta/manifests/train_v1_47.yaml"

REGISTRY = ROOT / "data/meta/registry/datasets_v1_47.yaml"

DATA_DIR = ROOT / "data/stream_datasets"

OUT = ROOT / "hpc/audit/dataset_audit_preview.tsv"



TARGET_NAMES = {

    "target", "class", "label", "y", "income",

    "output", "response", "category"

}



SENSITIVE_NAMES = {

    "sex", "gender", "race", "ethnicity", "age",

    "religion", "nationality", "marital_status"

}



with open(MANIFEST) as f:

    manifest = yaml.safe_load(f)



with open(REGISTRY) as f:

    registry = yaml.safe_load(f)



entries = {d["dataset_id"]: d for d in registry["datasets"]}

rows = []



for dataset_id in manifest["dataset_ids"]:

    entry = entries.get(dataset_id)

    if entry is None:

        raise RuntimeError(f"{dataset_id}: missing from registry")



    file_name = entry["file_name"]

    path = DATA_DIR / file_name



    if not path.exists():

        rows.append({

            "dataset_id": dataset_id,

            "file_name": file_name,

            "exists": False,

        })

        continue



    # SHA256 + line count in one sequential pass.

    sha = hashlib.sha256()

    newline_count = 0

    with open(path, "rb") as f:

        while True:

            block = f.read(8 * 1024 * 1024)

            if not block:

                break

            sha.update(block)

            newline_count += block.count(b"\n")



    sample = pd.read_csv(path, nrows=5000)

    columns = list(sample.columns)



    named_targets = [

        c for c in columns

        if str(c).strip().lower() in TARGET_NAMES

    ]



    if len(named_targets) == 1:

        target = named_targets[0]

        target_source = "named_candidate"

    else:

        target = columns[-1]

        target_source = "last_column_REVIEW"



    sensitive_candidates = [

        c for c in columns

        if str(c).strip().lower() in SENSITIVE_NAMES

        and c != target

    ]



    rows.append({

        "dataset_id": dataset_id,

        "file_name": file_name,

        "exists": True,

        "rows_approx": max(0, newline_count - 1),

        "n_columns": len(columns),

        "target_candidate": target,

        "target_source": target_source,

        "target_nunique_sample": int(sample[target].nunique(dropna=False)),

        "target_dtype_sample": str(sample[target].dtype),

        "last_column": columns[-1],

        "sensitive_candidates_REVIEW": ",".join(map(str, sensitive_candidates)),

        "sha256": sha.hexdigest(),

        "size_bytes": path.stat().st_size,

        "columns": ",".join(map(str, columns)),

    })



# Detect exact duplicate files.

sha_to_ids = {}

for r in rows:

    if r.get("sha256"):

        sha_to_ids.setdefault(r["sha256"], []).append(r["dataset_id"])



for r in rows:

    ids = sha_to_ids.get(r.get("sha256"), [])

    r["duplicate_sha_with"] = ",".join(x for x in ids if x != r["dataset_id"])



fieldnames = [

    "dataset_id",

    "file_name",

    "exists",

    "rows_approx",

    "n_columns",

    "target_candidate",

    "target_source",

    "target_nunique_sample",

    "target_dtype_sample",

    "last_column",

    "sensitive_candidates_REVIEW",

    "sha256",

    "duplicate_sha_with",

    "size_bytes",

    "columns",

]



OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "w", newline="") as f:

    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")

    writer.writeheader()

    writer.writerows(rows)



print(f"Audited {len(rows)} datasets")

print(f"Output: {OUT}")

print(f"Missing files: {sum(not r.get('exists', False) for r in rows)}")

print(f"Unique SHA256: {len(sha_to_ids)}")

