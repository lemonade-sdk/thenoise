#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# thenoise.sh — Bootstrap the venv (if needed) and launch the project.
# All CLI arguments are forwarded to `python -m thenoise`.
# ---------------------------------------------------------------------------

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

# Never import packages from the caller's user site or PYTHONPATH. This also
# applies to the final server process, not just the bootstrap probes.
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

# ---- 1. Check that uv is available ----------------------------------------
if ! command -v uv &>/dev/null; then
  cat <<'UV_HELP'
Error: uv is not installed.

Install it with:
  curl -LsSf https://astral.sh/uv/install.sh | sh

Then reload your shell (or run: source ~/.bashrc) and try again.
UV_HELP
  exit 1
fi

# ---- 2. Create the venv if it does not exist ------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment ($VENV_DIR) with Python 3.14 ..."
  # --managed-python includes Python.h, needed by Triton's HIP JIT on ROCm.
  uv venv "$VENV_DIR" --python 3.14 --managed-python
fi

# Check the *local* torch and torchvision together. Importing torchvision also
# catches native-extension mismatches such as 'torchvision::nms does not exist'.
# Accept both ROCm and CUDA: a working CUDA venv must stay supported here.
probe_torch() {
  "$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata
from pathlib import Path
import sys

import torch
import torchvision

venv = Path(sys.prefix).resolve()
for package in (torch, torchvision):
    if not Path(package.__file__).resolve().is_relative_to(venv):
        raise RuntimeError(f"{package.__name__} was imported outside the project venv")

hip = bool(torch.version.hip)
cuda = bool(torch.version.cuda)
if hip == cuda:
    raise RuntimeError("expected a ROCm or CUDA PyTorch build")

backend = "rocm" if hip else "cuda"
available = "yes" if torch.cuda.is_available() else "no"
print(backend, importlib.metadata.version("torch"),
      importlib.metadata.version("torchvision"), available)
PY
}

# Match the existing ROCm wheel selection; use the CUDA companion launcher's
# index and versions on an NVIDIA-only machine with no usable torch installed.
detect_gfx_arch() {
  local props v gen major minor
  for props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [ -f "$props" ] || continue
    v="$(awk '/^gfx_target_version/ {print $2}' "$props" 2>/dev/null)"
    if [ -n "$v" ] && [ "$v" != "0" ] && [ -z "${v//[0-9]/}" ]; then
      gen=$((v / 10000))
      major=$(((v / 100) % 100))
      minor=$((v % 100))
      printf 'gfx%d%x%x\n' "$gen" "$major" "$minor"
      return 0
    fi
  done
  return 1
}

select_install_backend() {
  if [ -n "${GFX_ARCH:-}" ]; then
    INSTALL_BACKEND="rocm"
  elif GFX_ARCH="$(detect_gfx_arch)"; then
    INSTALL_BACKEND="rocm"
    echo "Auto-detected GFX_ARCH=$GFX_ARCH"
  elif command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    INSTALL_BACKEND="cuda"
  else
    INSTALL_BACKEND="rocm"
    GFX_ARCH="gfx1151"
    echo "Warning: no GPU detected; falling back to GFX_ARCH=gfx1151." >&2
  fi
}

install_torch() {
  if [ "$INSTALL_BACKEND" = "rocm" ]; then
    echo "Installing ROCm torch for $GFX_ARCH ..."
    uv pip install --python "$VENV_DIR/bin/python" \
      --reinstall-package torch --reinstall-package torchvision \
      "torch[device-$GFX_ARCH]==2.14.0+rocm10.1.0rc2" \
      "torchvision[device-$GFX_ARCH]==0.29.0a0+rocm10.1.0rc2" \
      --index-url https://rc.repo.amd.com/rocm/whl-next/
  else
    echo "Installing CUDA torch ..."
    uv pip install --python "$VENV_DIR/bin/python" \
      --reinstall-package torch --reinstall-package torchvision \
      "torch==2.14" "torchvision==0.29" \
      --index-url https://download.pytorch.org/whl/cu132
  fi
}

# ---- 3. Validate or install the appropriate GPU torch --------------------
installed=""
if installed="$(probe_torch 2>/dev/null)"; then
  read -r backend torch_version torchvision_version available <<< "$installed"
fi

if [ -z "$installed" ] || { [ -n "${GFX_ARCH:-}" ] && [ "$backend" != "rocm" ]; }; then
  select_install_backend
  install_torch
  installed=""
elif [ "$available" != "yes" ]; then
  # A working package import is not enough: CUDA wheels can import on AMD-only
  # machines without ever being able to use the GPU. Do not reinstall a build
  # which already matches the detected hardware; that usually means a driver
  # problem, not a bad wheel.
  select_install_backend
  if [ "$backend" = "$INSTALL_BACKEND" ]; then
    echo "Error: $backend PyTorch is installed but cannot access a GPU. Check GPU drivers." >&2
    exit 1
  fi
  install_torch
  installed=""
fi

# ---- 4. Install the project without replacing its GPU-specific torch ------
if [ -z "$installed" ]; then
  installed="$(probe_torch)" || {
    echo "Error: PyTorch/torchvision import or GPU validation failed after bootstrap." >&2
    exit 1
  }
fi
read -r backend torch_version torchvision_version available <<< "$installed"
if [ "$available" != "yes" ]; then
  echo "Error: $backend PyTorch cannot access a GPU. Check GPU drivers." >&2
  exit 1
fi

# Explicitly target our venv even if the caller activated another venv.
# The exact installed versions are constraints for the project resolver: if a
# new dependency conflicts, fail instead of silently switching GPU backends.
uv pip install --python "$VENV_DIR/bin/python" -e "$PROJECT_DIR" \
  "torch==$torch_version" "torchvision==$torchvision_version"

# Catch dependency changes or imports which break the GPU wheel after install.
post_install="$(probe_torch)" || {
  echo "Error: PyTorch/torchvision was broken by the project dependency install." >&2
  exit 1
}
if [ "$post_install" != "$installed" ]; then
  echo "Error: PyTorch/torchvision changed during the project dependency install." >&2
  echo "Before: $installed" >&2
  echo "After:  $post_install" >&2
  exit 1
fi

# ---- 5. Set ROCm-specific environment variables ---------------------------
if [ "$backend" = "rocm" ]; then
  export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
  export MIOPEN_FIND_MODE=FAST
fi

# ---- 6. Launch the project, forwarding all arguments ----------------------
exec "$VENV_DIR/bin/python" -m thenoise "$@"
