#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [--host host] [--user user] [--key key] [--port port] [--local-port local_port] [--remote-port remote_port]

Environment variables:
  RUNPOD_HOST        RunPod SSH host
  RUNPOD_USER        SSH user (default: root)
  RUNPOD_KEY         SSH private key path (default: ~/.ssh/id_ed25519)
  RUNPOD_SSH_PORT    SSH port (default: 22)
  RUNPOD_LOCAL_PORT  Local forwarded port (default: 8000)
  RUNPOD_REMOTE_PORT Remote port on pod (default: 8000)

Example:
  RUNPOD_HOST=e1ig4vrkxnvfvo-64411a75@ssh.runpod.io \
  RUNPOD_KEY=~/.ssh/id_ed25519 \
  $0
EOF
}

HOST="${RUNPOD_HOST:-}"
USER="${RUNPOD_USER:-root}"
KEY="${RUNPOD_KEY:-$HOME/.ssh/id_ed25519}"
PORT="${RUNPOD_SSH_PORT:-22}"
LOCAL_PORT="${RUNPOD_LOCAL_PORT:-8000}"
REMOTE_PORT="${RUNPOD_REMOTE_PORT:-8000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --user) USER="$2"; shift 2;;
    --key) KEY="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --local-port) LOCAL_PORT="$2"; shift 2;;
    --remote-port) REMOTE_PORT="$2"; shift 2;;
    --help|-h) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "Error: RUNPOD_HOST is required or pass --host."
  usage
  exit 1
fi

echo "Connecting to RunPod pod at ${HOST} on port ${PORT}..."
ssh -i "$KEY" -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" -p "$PORT" "${USER}@${HOST}"
