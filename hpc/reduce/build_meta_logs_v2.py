
from __future__ import annotations



import csv

import json

from collections import Counter

from datetime import datetime, timezone

from pathlib import Path



ROOT = Path.home() / "AwareML_Extension"



OLD_ROOT = ROOT / "artifacts/meta_v2_campaign_737025"

NEW_ROOT = ROOT / "artifacts/meta_v2_chacha_v2"



OLD_MANIFEST = ROOT / "hpc/production/campaign_737025/meta_runs_v2.tsv"

NEW_MANIFEST = ROOT / "hpc/production/campaign_chacha_ovr_v1/meta_runs_chacha_ovr_v1.tsv"



OLD_SOURCE_HASH = ROOT / "hpc/production/campaign_737025/source_tree.sha256"

NEW_SOURCE_HASH = ROOT / "hpc/production/campaign_chacha_ovr_v1/source_tree_chacha_ovr_v1.sha256"



OUT = ROOT / "data/meta/snapshots/meta_logs_v2.json"





def load_manifest(path):

    with path.open(newline="", encoding="utf-8") as f:

        return {

            int(r["task_id"]): r

            for r in csv.DictReader(f, delimiter="\t")

        }





def read_json(path):

    with path.open(encoding="utf-8") as f:

        return json.load(f)





def source_hash(path):

    if not path.exists():

        return None

    return path.read_text(encoding="utf-8").strip() or None





def get_task_context(task_dir):

    p = task_dir / "task_context.json"

    if not p.exists():

        return {}

    try:

        return read_json(p)

    except Exception:

        return {}





def load_record(task_dir, manifest_row, campaign_id, campaign_source_hash):

    success_path = task_dir / "SUCCESS.json"

    summary_path = task_dir / "summary.json"



    if not success_path.exists():

        raise RuntimeError("Missing SUCCESS.json: {}".format(task_dir))



    if not summary_path.exists():

        raise RuntimeError("Missing summary.json: {}".format(task_dir))



    summary = read_json(summary_path)



    if not isinstance(summary, list) or len(summary) != 1:

        raise RuntimeError(

            "Expected exactly one summary result in {}".format(summary_path)

        )



    result = summary[0]



    if result.get("status") != "ok":

        raise RuntimeError(

            "{} has non-ok result: {}".format(task_dir, result.get("status"))

        )



    if result.get("framework") != manifest_row["framework"]:

        raise RuntimeError(

            "Framework mismatch in {}: manifest={} summary={}".format(

                task_dir,

                manifest_row["framework"],

                result.get("framework"),

            )

        )



    context = get_task_context(task_dir)



    record = dict(result)



    # Canonical experiment identity.

    record["dataset_id"] = manifest_row["dataset_id"]

    record["file_name"] = manifest_row["file_name"]

    record["target"] = manifest_row["target"]

    record["framework"] = manifest_row["framework"]

    record["seed"] = int(manifest_row["seed"])



    # Frozen protocol fields.

    record["max_samples_requested"] = int(manifest_row["max_samples"])

    record["window_size"] = int(manifest_row["window_size"])

    record["time_budget_sec"] = int(manifest_row["time_budget_sec"])

    record["dataset_sha256"] = manifest_row["sha256"]



    # Campaign provenance.

    record["campaign_id"] = campaign_id

    record["campaign_task_id"] = int(manifest_row["task_id"])

    record["source_tree_sha256"] = (

        context.get("source_tree_sha256")

        or context.get("source_tree_chacha_ovr_v1.sha256")

        or campaign_source_hash

    )



    record["slurm_job_id"] = context.get("slurm_job_id")

    record["slurm_array_task_id"] = context.get("slurm_array_task_id")

    record["slurm_cpus_per_task"] = context.get("slurm_cpus_per_task")

    record["slurm_mem_per_node"] = context.get("slurm_mem_per_node")

    record["node"] = context.get("node")

    record["platform"] = context.get("platform")



    # Preserve the important distinction between requested stream length and

    # the number of samples actually processed before the wall-clock budget.

    record["samples_processed"] = int(result.get("samples") or 0)



    return record





old_manifest = load_manifest(OLD_MANIFEST)

new_manifest = load_manifest(NEW_MANIFEST)



old_hash = source_hash(OLD_SOURCE_HASH)

new_hash = source_hash(NEW_SOURCE_HASH)



records = []



# ----------------------------------------------------------------------

# Campaign 737025:

# keep ONLY the four non-ChaCha frameworks.

# ----------------------------------------------------------------------



keep_old = {

    "AutoStreamML",

    "AutoClass",

    "OAML",

    "EvoAutoML",

}



for task_id, row in sorted(old_manifest.items()):

    if row["framework"] not in keep_old:

        continue



    task_dir = OLD_ROOT / "task_{:04d}".format(task_id)



    records.append(

        load_record(

            task_dir=task_dir,

            manifest_row=row,

            campaign_id="737025",

            campaign_source_hash=old_hash,

        )

    )



# ----------------------------------------------------------------------

# ChaCha replacement campaign:

# use ALL 141 new OVR-v1 rows.

# ----------------------------------------------------------------------



for task_id, row in sorted(new_manifest.items()):

    if row["framework"] != "ChaCha":

        raise RuntimeError(

            "Non-ChaCha row found in ChaCha manifest: {}".format(row)

        )



    task_dir = NEW_ROOT / "task_{:04d}".format(task_id)



    records.append(

        load_record(

            task_dir=task_dir,

            manifest_row=row,

            campaign_id="chacha-ovr-v1",

            campaign_source_hash=new_hash,

        )

    )



# ----------------------------------------------------------------------

# Final scientific integrity checks.

# ----------------------------------------------------------------------



if len(records) != 705:

    raise RuntimeError(

        "Expected exactly 705 final records, got {}".format(len(records))

    )



keys = [

    (r["dataset_id"], r["framework"], int(r["seed"]))

    for r in records

]



if len(set(keys)) != 705:

    duplicates = [

        key for key, n in Counter(keys).items()

        if n > 1

    ]

    raise RuntimeError(

        "Duplicate dataset/framework/seed combinations: {}".format(duplicates)

    )



framework_counts = Counter(r["framework"] for r in records)



expected_framework_counts = {

    "AutoStreamML": 141,

    "AutoClass": 141,

    "ChaCha": 141,

    "OAML": 141,

    "EvoAutoML": 141,

}



if dict(framework_counts) != expected_framework_counts:

    raise RuntimeError(

        "Framework counts incorrect. Expected {}, got {}".format(

            expected_framework_counts,

            dict(framework_counts),

        )

    )



dataset_counts = Counter(r["dataset_id"] for r in records)



if len(dataset_counts) != 47:

    raise RuntimeError(

        "Expected 47 datasets, got {}".format(len(dataset_counts))

    )



bad_dataset_counts = {

    dataset_id: n

    for dataset_id, n in dataset_counts.items()

    if n != 15

}



if bad_dataset_counts:

    raise RuntimeError(

        "Each dataset must have 15 rows "

        "(5 frameworks x 3 seeds). Bad counts: {}".format(

            bad_dataset_counts

        )

    )



seed_counts = Counter(int(r["seed"]) for r in records)



expected_seed_counts = {

    42: 235,

    43: 235,

    44: 235,

}



if dict(seed_counts) != expected_seed_counts:

    raise RuntimeError(

        "Seed counts incorrect: {}".format(dict(seed_counts))

    )



# Explicitly verify every final ChaCha record is the new OVR extension.

for r in records:

    if r["framework"] != "ChaCha":

        continue



    params = r.get("parameters") or {}



    if r.get("campaign_id") != "chacha-ovr-v1":

        raise RuntimeError("Old ChaCha result leaked into final V2.")



    if r.get("backend") != "FLAML AutoVW / ChaCha OVR extension":

        raise RuntimeError(

            "Unexpected ChaCha backend: {}".format(r.get("backend"))

        )



    if params.get("native_autovw_active") is not True:

        raise RuntimeError("ChaCha native AutoVW is not active.")



    if params.get("fallback_used") is not False:

        raise RuntimeError("ChaCha fallback result found.")



    if params.get("multiclass_strategy") != "one_vs_rest_native_autovw":

        raise RuntimeError(

            "Unexpected ChaCha multiclass strategy: {}".format(

                params.get("multiclass_strategy")

            )

        )



# Sort deterministically.

framework_order = {

    "AutoStreamML": 0,

    "AutoClass": 1,

    "EvoAutoML": 2,

    "OAML": 3,

    "ChaCha": 4,

}



records.sort(

    key=lambda r: (

        r["dataset_id"],

        framework_order[r["framework"]],

        int(r["seed"]),

    )

)



payload = {

    "schema_version": "2.0",

    "snapshot_id": "meta_logs_v2",

    "created_at_utc": datetime.now(timezone.utc).isoformat(),

    "total_records": len(records),

    "dataset_count": 47,

    "framework_count": 5,

    "seeds": [42, 43, 44],

    "selection_policy": {

        "campaign_737025": {

            "included_frameworks": [

                "AutoStreamML",

                "AutoClass",

                "OAML",

                "EvoAutoML",

            ],

            "excluded_frameworks": [

                "ChaCha",

            ],

            "reason": (

                "All ChaCha rows from campaign 737025 are superseded "

                "by the unified ChaCha OVR-v1 implementation."

            ),

            "source_tree_sha256": old_hash,

        },

        "chacha_ovr_v1": {

            "included_frameworks": [

                "ChaCha",

            ],

            "source_tree_sha256": new_hash,

            "extension": "AwareML one-vs-rest native AutoVW extension",

        },

    },

    "protocol": {

        "evaluation": "prequential_test_then_train",

        "max_samples": 30000,

        "window_size": 1000,

        "time_budget_sec": 60,

        "sustainability_tracking": True,

        "country_iso": "EST",

        "xai_method": "permutation",

        "xai_max_rows": 150,

    },

    "framework_counts": dict(framework_counts),

    "seed_counts": {

        str(k): v

        for k, v in sorted(seed_counts.items())

    },

    "records": records,

}



OUT.parent.mkdir(parents=True, exist_ok=True)



with OUT.open("w", encoding="utf-8") as f:

    json.dump(

        payload,

        f,

        indent=2,

        ensure_ascii=False,

        allow_nan=False,

    )



print("====================================================")

print("AwareML META-LOG V2 BUILD: SUCCESS")

print("====================================================")

print("Output:", OUT)

print("Records:", len(records))

print("Datasets:", len(dataset_counts))

print("Frameworks:", len(framework_counts))

print("")

print("Framework counts:")

for framework in expected_framework_counts:

    print("  {:15s} {}".format(

        framework,

        framework_counts[framework],

    ))

print("")

print("Seed counts:")

for seed in [42, 43, 44]:

    print("  {}: {}".format(seed, seed_counts[seed]))

print("")

print("Unique (dataset, framework, seed):", len(set(keys)))

print("Old ChaCha rows included: 0")

print("New ChaCha OVR rows included:", framework_counts["ChaCha"])

print("====================================================")

