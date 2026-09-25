#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# thenoise.sh — Bootstrap the venv (if needed) and launch the project.
# All CLI arguments are forwarded to `python -m thenoise`.
# ---------------------------------------------------------------------------

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

# ---- 1. Check that uv is available ----------------------------------------
if ! command -v uv &>/dev/null; then
  cat <<'EOF'
Error: uv is not installed.

Install it with:
  curl -LsSf https://astral.sh/uv/install.sh | sh

Then reload your shell (or run: source ~/.bashrc) and try again.
EOF
  exit 1
fi

# ---- 2. Create the venv if it does not exist ------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment ($VENV_DIR) with Python 3.14 ..."
  uv venv "$VENV_DIR" --python 3.14 --managed-python
fi

# ---- 3. Install torch (CUDA build) ----------------------------------------
if ! "$VENV_DIR/bin/python" -c "import torch" &>/dev/null; then
  echo "Installing CUDA torch ..."

  uv pip install \
    "torch==2.14" \
    "torchvision==0.29" \
    --index-url https://download.pytorch.org/whl/cu132
fi

# ---- 4. Install the project in editable mode ------------------------------
uv pip install -e "$PROJECT_DIR"

# ---- 5. Launch the project, forwarding all arguments ----------------------
exec "$VENV_DIR/bin/python" -m thenoise "$@"