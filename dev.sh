#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/external/semantic_uncertainty:${PYTHONPATH:-}"

exec "$@"
