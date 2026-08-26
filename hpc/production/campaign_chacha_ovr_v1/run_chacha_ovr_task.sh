#!/usr/bin/env bash

#!/usr/bin/env bash

set -euo pipefail



ROOT="$HOME/AwareML_Extension"

MANIFEST="$ROOT/hpc/production/meta_runs_chacha_ovr_v1.tsv"



MAIN_PY="$HOME/.conda/envs/awareml-main/bin/python"

OAML_PY="$HOME/.conda/envs/awareml-oaml/bin/python"

EVO_PY="$HOME/.conda/envs/awareml-evo/bin/python"



TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

LINE_NO=$((TASK_ID + 2))



LINE="$(sed -n "${LINE_NO}p" "$MANIFEST")"



if [[ -z "$LINE" ]]; then

    echo "ERROR: no manifest row for task $TASK_ID"

    exit 2

fi



IFS=$'\t' read -r manifest_task_id dataset_id file_name target framework environment seed max_samples window_size time_budget_sec expected_sha <<< "$LINE"

if [[ "$manifest_task_id" != "$TASK_ID" ]]; then

    echo "ERROR: manifest task mismatch: expected=$TASK_ID got=$manifest_task_id"

    exit 3

fi



case "$framework:$environment" in

    AutoStreamML:main|AutoClass:main|ChaCha:main|OAML:oaml|EvoAutoML:evo)

        ;;

    *)

        echo "ERROR: invalid framework/environment routing: $framework / $environment"

        exit 4

        ;;

esac



DATA="$ROOT/data/stream_datasets/$file_name"



if [[ ! -f "$DATA" ]]; then

    echo "ERROR: dataset missing: $DATA"

    exit 5

fi



for PY in "$MAIN_PY" "$OAML_PY" "$EVO_PY"; do

    if [[ ! -x "$PY" ]]; then

        echo "ERROR: Python executable missing: $PY"

        exit 6

    fi

done



export AWAREML_CODECARBON_ENABLED=true

export AWAREML_COUNTRY_ISO=EST



export AWAREML_OAML_MODE=online

export AWAREML_OAML_PYTHON="$OAML_PY"

export AWAREML_EVO_PYTHON="$EVO_PY"



export PYTHONUNBUFFERED=1



export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"



TASK_PAD="$(printf '%04d' "$TASK_ID")"



OUT_ROOT="$ROOT/artifacts/meta_v2_chacha_v2"

TASK_ROOT="$OUT_ROOT/task_${TASK_PAD}"

SUCCESS="$TASK_ROOT/SUCCESS.json"

FAILED="$TASK_ROOT/FAILED.txt"



if [[ -f "$SUCCESS" ]]; then

    echo "Task $TASK_ID already passed QC. Skipping."

    exit 0

fi



if [[ -d "$TASK_ROOT" ]]; then

    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

    mv "$TASK_ROOT" "${TASK_ROOT}.previous.${STAMP}"

fi



mkdir -p "$TASK_ROOT/runs"



failure_marker() {

    rc=$?

    if [[ $rc -ne 0 ]]; then

        {

            echo "task_id=$TASK_ID"

            echo "dataset_id=$dataset_id"

            echo "framework=$framework"

            echo "seed=$seed"

            echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"

            echo "node=${SLURMD_NODENAME:-$(hostname)}"

            echo "exit_code=$rc"

            echo "failed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

        } > "$FAILED"

    fi

    exit "$rc"

}

trap failure_marker EXIT



SOURCE_HASH="unknown"

if [[ -f "$ROOT/hpc/production/source_tree_chacha_ovr_v1.sha256" ]]; then

    SOURCE_HASH="$(cat "$ROOT/hpc/production/source_tree_chacha_ovr_v1.sha256")"

fi



export TASK_CONTEXT_TASK_ID="$TASK_ID"

export TASK_CONTEXT_DATASET="$dataset_id"

export TASK_CONTEXT_FRAMEWORK="$framework"

export TASK_CONTEXT_SEED="$seed"

export TASK_CONTEXT_SHA="$expected_sha"

export TASK_CONTEXT_SOURCE_HASH="$SOURCE_HASH"



"$MAIN_PY" - <<'PY' > "$TASK_ROOT/task_context.json"

import json

import os

import platform



print(json.dumps({

    "task_id": int(os.environ["TASK_CONTEXT_TASK_ID"]),

    "dataset_id": os.environ["TASK_CONTEXT_DATASET"],

    "framework": os.environ["TASK_CONTEXT_FRAMEWORK"],

    "seed": int(os.environ["TASK_CONTEXT_SEED"]),

    "dataset_sha256": os.environ["TASK_CONTEXT_SHA"],

    "source_tree_sha256": os.environ["TASK_CONTEXT_SOURCE_HASH"],

    "slurm_job_id": os.getenv("SLURM_JOB_ID"),

    "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),

    "slurm_cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK"),

    "slurm_mem_per_node": os.getenv("SLURM_MEM_PER_NODE"),

    "node": os.getenv("SLURMD_NODENAME") or platform.node(),

    "platform": platform.platform(),

}, indent=2))

PY



if [[ "${AWAREML_DRY_RUN:-0}" == "1" ]]; then

    echo "DRY RUN"

    echo "task=$TASK_ID dataset=$dataset_id framework=$framework seed=$seed"

    echo "target=$target samples=$max_samples window=$window_size budget=$time_budget_sec"

    echo "data=$DATA"

    echo "main_python=$MAIN_PY"

    echo "oaml_python=$AWAREML_OAML_PYTHON"

    echo "evo_python=$AWAREML_EVO_PYTHON"

    trap - EXIT

    exit 0

fi



echo "============================================================"

echo "AwareML Meta-V2 production task"

echo "task_id       = $TASK_ID"

echo "dataset       = $dataset_id"

echo "framework     = $framework"

echo "environment   = $environment"

echo "seed          = $seed"

echo "max_samples   = $max_samples"

echo "window_size   = $window_size"

echo "time_budget   = $time_budget_sec"

echo "node          = ${SLURMD_NODENAME:-$(hostname)}"

echo "job           = ${SLURM_JOB_ID:-unknown}"

echo "============================================================"



"$MAIN_PY" "$ROOT/scripts/run_recorded_benchmark.py" "$DATA" --dataset-id "$dataset_id" --target "$target" --frameworks "$framework" --max-samples "$max_samples" --window "$window_size" --time-budget "$time_budget_sec" --seed "$seed" --track-sustainability --xai-method permutation --xai-max-rows 150 --oaml-mode online --experiment-root "$TASK_ROOT/runs" --protocol-version meta-v2-chacha-ovr-v1 --nonce "chacha-ovr-v1-task-${TASK_ID}" --output "$TASK_ROOT/summary.json"



"$MAIN_PY" "$ROOT/hpc/production/validate_task.py" --summary "$TASK_ROOT/summary.json" --framework "$framework" --sha256 "$expected_sha" --success "$SUCCESS"



rm -f "$FAILED"



trap - EXIT

echo "Task $TASK_ID completed and passed QC."

