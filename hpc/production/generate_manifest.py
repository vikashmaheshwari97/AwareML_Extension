
from pathlib import Path

import csv

import yaml



ROOT = Path.home() / "AwareML_Extension"



REGISTRY = ROOT / "data/meta/registry/datasets_v1_47_audited.yaml"

TRAIN = ROOT / "data/meta/manifests/train_v1_47.yaml"

AUDIT = ROOT / "hpc/audit/dataset_audit_preview.tsv"

OUT = ROOT / "hpc/production/meta_runs_v2.tsv"



FRAMEWORKS = ["AutoStreamML", "AutoClass", "ChaCha", "OAML", "EvoAutoML"]

SEEDS = [42, 43, 44]



ENVIRONMENTS = {

    "AutoStreamML": "main",

    "AutoClass": "main",

    "ChaCha": "main",

    "OAML": "oaml",

    "EvoAutoML": "evo",

}



with open(REGISTRY) as f:

    registry = yaml.safe_load(f)



with open(TRAIN) as f:

    train = yaml.safe_load(f)



with open(AUDIT, newline="") as f:

    audit = {

        r["dataset_id"]: r

        for r in csv.DictReader(f, delimiter="\t")

    }



datasets = {d["dataset_id"]: d for d in registry["datasets"]}



rows = []

task_id = 0



for dataset_id in train["dataset_ids"]:

    d = datasets[dataset_id]

    a = audit[dataset_id]



    path = ROOT / "data/stream_datasets" / d["file_name"]



    if not path.exists():

        raise RuntimeError(f"Missing dataset: {path}")



    if d["fingerprint_sha256"] != a["sha256"]:

        raise RuntimeError(f"SHA256 mismatch: {dataset_id}")



    available_rows = int(a["rows_approx"])

    max_samples = min(30000, available_rows)



    for framework in FRAMEWORKS:

        for seed in SEEDS:

            rows.append({

                "task_id": task_id,

                "dataset_id": dataset_id,

                "file_name": d["file_name"],

                "target": d["target"],

                "framework": framework,

                "environment": ENVIRONMENTS[framework],

                "seed": seed,

                "max_samples": max_samples,

                "window_size": 1000,

                "time_budget_sec": 60,

                "sha256": d["fingerprint_sha256"],

            })

            task_id += 1



assert len(rows) == 705

assert [r["task_id"] for r in rows] == list(range(705))



fields = [

    "task_id",

    "dataset_id",

    "file_name",

    "target",

    "framework",

    "environment",

    "seed",

    "max_samples",

    "window_size",

    "time_budget_sec",

    "sha256",

]



OUT.parent.mkdir(parents=True, exist_ok=True)



with open(OUT, "w", newline="") as f:

    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")

    writer.writeheader()

    writer.writerows(rows)



print("Manifest:", OUT)

print("Datasets:", len(train["dataset_ids"]))

print("Frameworks:", len(FRAMEWORKS))

print("Seeds:", len(SEEDS))

print("Tasks:", len(rows))

