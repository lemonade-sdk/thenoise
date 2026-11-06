#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The repo's launchers install backend-specific PyTorch before the package.
# --help triggers bootstrap, but neither loads a checkpoint nor generates an image.
case "${1:-}" in
  rocm) ./thenoise.sh --help >/dev/null ;;
  cuda) ./thenoise_cuda.sh --help >/dev/null ;;
  *) echo "Usage: $0 rocm|cuda" >&2; exit 2 ;;
esac

# Do NOT run 'uv sync': it may replace the manually installed GPU torch wheel.
uv pip install -e '.[dev]'
