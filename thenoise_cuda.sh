#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# thenoise_cuda.sh — deprecated shim. thenoise.sh now detects/installs CUDA
# itself, so this only pins the backend and hands everything over.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Warning: thenoise_cuda.sh is deprecated; use ./thenoise.sh (it installs CUDA torch automatically, or force it with THENOISE_BACKEND=cuda)." >&2

exec env THENOISE_BACKEND=cuda bash "$SCRIPT_DIR/thenoise.sh" "$@"
