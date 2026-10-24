#!/usr/bin/env bash
#SBATCH --job-name=awareml-v3
#SBATCH --exclusive
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=02:10:00

set -euo pipefail
CAMPAIGN="${1:?Pass the frozen campaign directory}"
MAIN_PY="${2:?Pass the absolute main Python interpreter}"
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"

export AWAREML_OAML_MODE=online
export PYTHONUNBUFFERED=1
"$MAIN_PY" scripts/awareml_v3.py run --campaign "$CAMPAIGN" --task-id "${SLURM_ARRAY_TASK_ID:?Submit as an array}"
