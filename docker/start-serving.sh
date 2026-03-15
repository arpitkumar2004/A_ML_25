#!/usr/bin/env sh
set -eu

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"

exec python -m uvicorn src.serving.app:app --host "$HOST" --port "$PORT" --workers "$UVICORN_WORKERS"
