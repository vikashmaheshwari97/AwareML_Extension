#!/usr/bin/env bash
#SBATCH --job-name=p15-llm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-23%4

set -euo pipefail

CAMPAIGN_ROOT="${1:-${PHASE15_CAMPAIGN_ROOT:-}}"
if [[ -z "$CAMPAIGN_ROOT" ]]; then
    echo "ERROR: campaign root must be argument 1 or PHASE15_CAMPAIGN_ROOT." >&2
    exit 2
fi
CAMPAIGN_ROOT="$(readlink -f "$CAMPAIGN_ROOT")"

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

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PY="${PHASE15_PYTHON:-$HOME/.conda/envs/awareml-main/bin/python}"
if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3 || true)"
fi
if [[ -z "$PY" || ! -x "$PY" ]]; then
    echo "ERROR: Python not found. Set PHASE15_PYTHON." >&2
    exit 4
fi

OLLAMA_BIN="${PHASE15_OLLAMA_BIN:-}"
if [[ -z "$OLLAMA_BIN" ]]; then
    OLLAMA_BIN="$(command -v ollama || true)"
fi
if [[ -z "$OLLAMA_BIN" && -x "$HOME/.local/bin/ollama" ]]; then
    OLLAMA_BIN="$HOME/.local/bin/ollama"
fi
if [[ -z "$OLLAMA_BIN" || ! -x "$OLLAMA_BIN" ]]; then
    echo "ERROR: Ollama not found. Run setup_phase15_ollama.sbatch first." >&2
    exit 5
fi

export PATH="$(dirname "$OLLAMA_BIN"):$PATH"
export OLLAMA_MODEL="llama3:8b"
export OLLAMA_MODELS="${PHASE15_OLLAMA_MODELS:-$HOME/.ollama/models}"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=10m
export OLLAMA_TMPDIR="${TMPDIR:-/tmp}/phase15_ollama_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OLLAMA_TMPDIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false

PORT="$("$PY" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"
export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_BASE_URL="http://127.0.0.1:${PORT}"

mkdir -p "$CAMPAIGN_ROOT/logs"
SERVER_LOG="$CAMPAIGN_ROOT/logs/ollama_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

"$OLLAMA_BIN" serve >"$SERVER_LOG" 2>&1 &
OLLAMA_PID=$!

cleanup() {
    rc=$?
    kill "$OLLAMA_PID" >/dev/null 2>&1 || true
    rm -rf "$OLLAMA_TMPDIR" >/dev/null 2>&1 || true
    exit "$rc"
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 90); do
    if "$PY" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("${OLLAMA_BASE_URL}/api/tags", timeout=2).read()
PY
    then
        READY=1
        break
    fi
    sleep 2
done

if [[ "$READY" != "1" ]]; then
    echo "ERROR: Ollama server did not become ready." >&2
    tail -100 "$SERVER_LOG" || true
    exit 6
fi

echo "============================================================"
echo "Phase-15 empirical array task"
echo "campaign = $CAMPAIGN_ROOT"
echo "task     = $SLURM_ARRAY_TASK_ID"
echo "job      = ${SLURM_JOB_ID:-unknown}"
echo "node     = ${SLURMD_NODENAME:-$(hostname)}"
echo "gpu      = ${CUDA_VISIBLE_DEVICES:-unknown}"
echo "ollama   = $OLLAMA_BASE_URL"
echo "============================================================"

"$PY" "$ROOT/hpc/production/phase15/run_phase15_case.py" \
    --campaign-root "$CAMPAIGN_ROOT" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --timeout-sec "${PHASE15_TIMEOUT_SEC:-300}" \
    --retries "${PHASE15_RETRIES:-1}" \
    --base-url "$OLLAMA_BASE_URL" \
    --strict-runtime
