#!/usr/bin/env bash
#SBATCH --job-name=p15-ollama-setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

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
    exit 2
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
    exit 3
fi

REQ_OLLAMA_VERSION="0.32.14"
REQ_MODEL="llama3:8b"
REQ_DIGEST="365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1"

OLLAMA_BIN="${PHASE15_OLLAMA_BIN:-}"
if [[ -z "$OLLAMA_BIN" ]]; then
    OLLAMA_BIN="$(command -v ollama || true)"
fi
if [[ -z "$OLLAMA_BIN" && -x "$HOME/.local/bin/ollama" ]]; then
    OLLAMA_BIN="$HOME/.local/bin/ollama"
fi

if [[ -z "$OLLAMA_BIN" ]]; then
    echo "Ollama not found. Installing pinned v${REQ_OLLAMA_VERSION} in \$HOME/.local ..."
    mkdir -p "$HOME/.local" "$HOME/.cache/phase15_ollama"

    ARCHIVE="${PHASE15_OLLAMA_ARCHIVE:-$HOME/.cache/phase15_ollama/ollama-linux-amd64-v${REQ_OLLAMA_VERSION}.tar.zst}"
    if [[ ! -f "$ARCHIVE" ]]; then
        URL="https://github.com/ollama/ollama/releases/download/v${REQ_OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
        echo "Downloading: $URL"
        curl -fL --retry 3 --retry-delay 3 "$URL" -o "$ARCHIVE"
    fi

    if tar --help 2>/dev/null | grep -q -- '--zstd'; then
        tar --zstd -xf "$ARCHIVE" -C "$HOME/.local"
    elif command -v unzstd >/dev/null 2>&1; then
        unzstd -c "$ARCHIVE" | tar -xf - -C "$HOME/.local"
    else
        echo "ERROR: need tar --zstd or unzstd to extract Ollama." >&2
        exit 4
    fi
    OLLAMA_BIN="$HOME/.local/bin/ollama"
fi

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "ERROR: Ollama binary is not executable: $OLLAMA_BIN" >&2
    exit 5
fi

export PATH="$(dirname "$OLLAMA_BIN"):$PATH"
export OLLAMA_MODEL="$REQ_MODEL"
export OLLAMA_MODELS="${PHASE15_OLLAMA_MODELS:-$HOME/.ollama/models}"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=10m
export OLLAMA_TMPDIR="${TMPDIR:-/tmp}/phase15_ollama_tmp"
mkdir -p "$OLLAMA_MODELS" "$OLLAMA_TMPDIR"

VERSION_TEXT="$("$OLLAMA_BIN" --version 2>&1 || true)"
echo "Ollama CLI: $VERSION_TEXT"
if ! grep -q "$REQ_OLLAMA_VERSION" <<<"$VERSION_TEXT"; then
    echo "ERROR: journal runtime requires Ollama $REQ_OLLAMA_VERSION." >&2
    exit 6
fi

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

SETUP_ROOT="$ROOT/artifacts/phase15/hpc_setup"
mkdir -p "$SETUP_ROOT"
LOG="$SETUP_ROOT/ollama_setup_${SLURM_JOB_ID:-manual}.log"

"$OLLAMA_BIN" serve >"$LOG" 2>&1 &
OLLAMA_PID=$!
cleanup() {
    kill "$OLLAMA_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Waiting for Ollama server..."
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
    tail -100 "$LOG" || true
    exit 7
fi

if ! "$OLLAMA_BIN" list | awk '{print $1}' | grep -qx "$REQ_MODEL"; then
    echo "Pulling $REQ_MODEL into shared model store $OLLAMA_MODELS ..."
    "$OLLAMA_BIN" pull "$REQ_MODEL"
fi

"$PY" - <<PY
import requests, sys
base = "${OLLAMA_BASE_URL}"
required_model = "${REQ_MODEL}"
required_digest = "${REQ_DIGEST}"
tags = requests.get(base + "/api/tags", timeout=10).json().get("models", [])
row = next((x for x in tags if (x.get("name") or x.get("model")) == required_model), None)
if row is None:
    raise SystemExit("Required model is absent after pull.")
digest = row.get("digest")
print("Model:", required_model)
print("Digest:", digest)
if digest != required_digest:
    raise SystemExit(
        "MODEL DIGEST MISMATCH. Expected {} got {}. "
        "Do not run the journal campaign until reviewed."
        .format(required_digest, digest)
    )
PY

echo "Running one deterministic GPU smoke generation..."
"$PY" - <<PY
import requests
payload = {
    "model": "${REQ_MODEL}",
    "prompt": "Reply with exactly: PHASE15_READY",
    "stream": False,
    "keep_alive": "10m",
    "options": {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "num_predict": 16,
    },
}
r = requests.post("${OLLAMA_BASE_URL}/api/generate", json=payload, timeout=300)
r.raise_for_status()
print(r.json().get("response", "").strip())
PY

"$OLLAMA_BIN" ps || true
nvidia-smi || true

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
"$PY" "$ROOT/hpc/environment_freeze/phase15/capture_phase15_environment.py" \
    --output "$SETUP_ROOT/environment_${STAMP}.json" \
    --ollama-base-url "$OLLAMA_BASE_URL"

echo
echo "Phase-15 Ollama/model setup: PASS"
echo "Shared models: $OLLAMA_MODELS"
echo "Setup log: $LOG"
