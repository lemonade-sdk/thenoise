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
  echo "Creating virtual environment ($VENV_DIR) with Python 3.13 ..."
  # --managed-python forces uv to use its own standalone CPython build rather
  # than a system python3.13. Triton JIT-compiles its HIP driver module at
  # runtime and needs Python.h; a system python without its matching -dev
  # package has no headers, which makes torch.compile fail on first generation.
  # uv's managed builds always ship headers, so this keeps setup sudo-free.
  uv venv "$VENV_DIR" --python 3.13 --managed-python
fi

# ---- 3. Install torch (ROCm build) ----------------------------------------
if ! "$VENV_DIR/bin/python" -c "import torch" &>/dev/null; then
  echo "Installing ROCm torch ..."

  detect_gfx_arch() {
    local version=""
    local props v
    for props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
      [ -f "$props" ] || continue
      v="$(awk '/^gfx_target_version/ {print $2}' "$props" 2>/dev/null)"
      if [ -n "$v" ] && [ "$v" != "0" ]; then
        version="$v"
        break
      fi
    done

    # Dynamic mapping mirroring ROCm's gfx_target_version_to_arch(). The
    # packed value is YYMMSS -> gfx{YY}{MM}{SS}, where MM and SS are encoded
    # in hex so values >= 10 still collapse to two (or one) characters.
    if [ -z "$version" ] || [ -n "${version//[0-9]/}" ]; then
      return 1
    fi

    local gen=$((version / 10000))
    local major=$(( (version / 100) % 100 ))
    local minor=$((version % 100))
    printf "gfx%d%x%x\n" "$gen" "$major" "$minor"
  }

  # An explicit GFX_ARCH env var always wins over auto-detection.
  if [ -n "${GFX_ARCH:-}" ]; then
    :
  elif GFX_ARCH="$(detect_gfx_arch)" && [ -n "$GFX_ARCH" ]; then
    echo "Auto-detected GFX_ARCH=$GFX_ARCH"
  else
    echo "Warning: could not auto-detect GFX_ARCH from /sys/class/kfd; falling back to gfx1151." >&2
    GFX_ARCH="gfx1151"
  fi

  uv pip install \
    "torch[device-$GFX_ARCH]==2.11" \
    "torchvision[device-$GFX_ARCH]==0.26" \
    --index-url https://repo.amd.com/rocm/whl-multi-arch/
fi

# ---- 4. Install the project in editable mode ------------------------------
uv pip install -e "$PROJECT_DIR"

# ---- 5. Set ROCm-specific environment variables ---------------------------
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export MIOPEN_FIND_MODE=FAST
export TORCH_BLAS_PREFER_HIPBLASLT=1

# ---- 6. Launch the project, forwarding all arguments ----------------------
exec "$VENV_DIR/bin/python" -m thenoise "$@"
