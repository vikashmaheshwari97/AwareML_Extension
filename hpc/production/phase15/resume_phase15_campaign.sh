#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <campaign-root> [concurrency]" >&2
    exit 2
fi

CAMPAIGN_ROOT="$(readlink -f "$1")"
CONCURRENCY="${2:-${PHASE15_CONCURRENCY:-4}}"

resolve_root() {
    if [[ -n "${AWAREML_ROOT:-}" && -d "${AWAREML_ROOT}/awareml" ]]; then
        printf '%s\n' "$AWAREML_ROOT"
        return
    fi
    for candidate in "$HOME/AwareMLExtension" "$HOME/AwareML_Extension"; do
        if [[ -d "$candidate/awareml" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    exit 3
}
ROOT="$(resolve_root)"
cd "$ROOT"

PY="${PHASE15_PYTHON:-$HOME/.conda/envs/awareml-main/bin/python}"
if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3 || true)"
fi

ARRAY="$("$PY" "$ROOT/hpc/production/phase15/resume_phase15_campaign.py" \
    --campaign-root "$CAMPAIGN_ROOT" \
    --print-array)"

if [[ -z "$ARRAY" ]]; then
    echo "All 24 Phase-15 cases already have valid SUCCESS markers."
    exit 0
fi

GPU_GRES="${PHASE15_GPU_GRES:-gpu:a100-40g:1}"
ACCOUNT_ARGS=()
if [[ -n "${PHASE15_SLURM_ACCOUNT:-}" ]]; then
    ACCOUNT_ARGS=(--account "$PHASE15_SLURM_ACCOUNT")
fi

echo "Resubmitting task IDs: $ARRAY"

sbatch \
    "${ACCOUNT_ARGS[@]}" \
    --partition=gpu \
    --gres="$GPU_GRES" \
    --time="${PHASE15_TIME_LIMIT:-01:00:00}" \
    --cpus-per-task="${PHASE15_CPUS:-4}" \
    --mem="${PHASE15_MEMORY:-16G}" \
    --array="${ARRAY}%${CONCURRENCY}" \
    --output="$CAMPAIGN_ROOT/logs/resume-%A_%a.out" \
    --error="$CAMPAIGN_ROOT/logs/resume-%A_%a.err" \
    "$ROOT/hpc/production/phase15/phase15_llm_array.sbatch" \
    "$CAMPAIGN_ROOT"
