#!/usr/bin/env bash
set -euo pipefail

# TensorBoard launcher for this repo.
# Uses the same conda env path setup as `ptor.sh`, but runs non-interactively.

export PATH="/playpen/hisham26/conda/envs/pytorch_env/bin:$PATH"
export CONDA_PREFIX="/playpen/hisham26/conda/envs/pytorch_env"
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

LOGDIR="${TB_LOGDIR:-results/tensorboard}"
PORT="${TB_PORT:-6006}"
HOST="${TB_HOST:-127.0.0.1}"

if [[ "${TB_BIND_ALL:-0}" == "1" ]]; then
  HOST="0.0.0.0"
fi

echo "[tb] logdir: $LOGDIR"
echo "[tb] host:   $HOST"
echo "[tb] port:   $PORT"

if ! command -v tensorboard >/dev/null 2>&1; then
  echo "[tb] ERROR: tensorboard not found in PATH. Try: ./ptor.sh then: pip install tensorboard" >&2
  exit 1
fi

echo
echo "[tb] Open in your browser:"
echo "  http://localhost:${PORT}/"

if [[ -n "${SSH_CONNECTION:-}" ]]; then
  remote_host="$(hostname -f 2>/dev/null || hostname)"
  echo
  echo "[tb] Detected SSH session. If you're on a different machine, use port forwarding:"
  echo "  ssh -L ${PORT}:127.0.0.1:${PORT} ${USER}@${remote_host}"
fi

echo
echo "[tb] Starting TensorBoard (Ctrl+C to stop)..."
exec tensorboard --logdir "$LOGDIR" --port "$PORT" --host "$HOST"

