#!/usr/bin/env bash
# Run the ECG Heartbeat CBM Explorer.
# Usage: ./explorer/run.sh [8000] [0.0.0.0]
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_ROOT/.venv_new"

if [[ ! -d "$VENV" ]]; then
  echo "Virtual environment not found at $VENV" >&2
  exit 1
fi

PORT="${1:-8000}"
HOST="${2:-127.0.0.1}"

echo "Starting ECG CBM Explorer at http://${HOST}:${PORT}"
cd "$REPO_ROOT/explorer"
"$VENV/bin/uvicorn" app:app --host "$HOST" --port "$PORT" --reload
