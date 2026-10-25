#!/usr/bin/env bash
# demo/api/run_local.sh
# Run the FastAPI inference API locally for development.
# Usage: bash demo/api/run_local.sh [port]
#
# Requires the project root in PYTHONPATH so demo.* imports resolve.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PORT="${1:-8000}"

echo "Starting Multimodal Fault Console API on http://localhost:$PORT"
echo "Repo root: $REPO_ROOT"
echo ""

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT" uvicorn demo.api.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --reload \
    --log-level info
