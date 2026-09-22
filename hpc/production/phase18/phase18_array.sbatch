#!/usr/bin/env bash
#SBATCH --job-name=awareml-p18
#SBATCH --array=0-464%8
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=artifacts/phase18_final_heldout_31_v1/slurm/%A_%a.out
#SBATCH --error=artifacts/phase18_final_heldout_31_v1/slurm/%A_%a.err

set -euo pipefail

ROOT="${AWAREML_ROOT:-$PWD}"
cd "$ROOT"
mkdir -p artifacts/phase18_final_heldout_31_v1/slurm

PYTHON="${AWAREML_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Main AwareML Python is not executable: $PYTHON" >&2
  exit 2
fi

# The current adapters auto-discover these isolated environments from the repo root.
# You can override them on the cluster if needed:
# export AWAREML_EVO_PYTHON=/absolute/path/to/evo/python
# export AWAREML_OAML_PYTHON=/absolute/path/to/oaml/python
# export AWAREML_OAML_MODE=online

"$PYTHON" hpc/production/phase18/run_phase18_task.py \
  --task-id "${SLURM_ARRAY_TASK_ID}"
