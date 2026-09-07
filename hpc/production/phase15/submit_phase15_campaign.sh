#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_ID=""
CONCURRENCY="${PHASE15_CONCURRENCY:-4}"
GPU_GRES="${PHASE15_GPU_GRES:-gpu:a100-40g:1}"
TIME_LIMIT="${PHASE15_TIME_LIMIT:-01:00:00}"
MEMORY="${PHASE15_MEMORY:-16G}"
CPUS="${PHASE15_CPUS:-4}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --campaign-id)
            CAMPAIGN_ID="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --gpu-gres)
            GPU_GRES="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --mem)
            MEMORY="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

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
    echo "ERROR: cannot locate AwareML repo. Set AWAREML_ROOT." >&2
    exit 3
}

ROOT="$(resolve_root)"
cd "$ROOT"

PY="${PHASE15_PYTHON:-$HOME/.conda/envs/awareml-main/bin/python}"
if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3 || true)"
fi

if [[ -z "$CAMPAIGN_ID" ]]; then
    CAMPAIGN_ID="phase15_llama3_8b_$(date -u +%Y%m%dT%H%M%SZ)"
fi

"$PY" "$ROOT/hpc/production/phase15/prepare_phase15_campaign.py" \
    --campaign-id "$CAMPAIGN_ID"

CAMPAIGN_ROOT="$ROOT/artifacts/phase15/hpc_runs/$CAMPAIGN_ID"

"$PY" "$ROOT/hpc/audit/phase15/audit_phase15_campaign.py" \
    --campaign-root "$CAMPAIGN_ROOT"

ACCOUNT_ARGS=()
if [[ -n "${PHASE15_SLURM_ACCOUNT:-}" ]]; then
    ACCOUNT_ARGS=(--account "$PHASE15_SLURM_ACCOUNT")
fi

echo
echo "Submitting 24 cases with concurrency ${CONCURRENCY}"
echo "GPU request: ${GPU_GRES}"
echo "Campaign: ${CAMPAIGN_ROOT}"

sbatch \
    "${ACCOUNT_ARGS[@]}" \
    --partition=gpu \
    --gres="$GPU_GRES" \
    --time="$TIME_LIMIT" \
    --cpus-per-task="$CPUS" \
    --mem="$MEMORY" \
    --array="0-23%${CONCURRENCY}" \
    --output="$CAMPAIGN_ROOT/logs/slurm-%A_%a.out" \
    --error="$CAMPAIGN_ROOT/logs/slurm-%A_%a.err" \
    "$ROOT/hpc/production/phase15/phase15_llm_array.sbatch" \
    "$CAMPAIGN_ROOT"

echo
echo "Monitor with:"
echo "  squeue -u $USER"
echo "Campaign:"
echo "  $CAMPAIGN_ROOT"
