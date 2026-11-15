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

# An explicit backend overrides auto-detection. An already working torch
# installation is otherwise preserved without prompting on every launch.
BACKEND_PREFERENCE="${THENOISE_BACKEND:-auto}"
case "$BACKEND_PREFERENCE" in
  auto|rocm|cuda) ;;
  *)
    echo "Error: THENOISE_BACKEND must be auto, rocm, or cuda." >&2
    exit 2
    ;;
esac

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

# Use nvidia-smi's actual GPU listing, rather than only checking whether the
# binary is installed (it can exist even when no NVIDIA GPU is accessible).
detect_nvidia() {
  command -v nvidia-smi &>/dev/null &&
    nvidia-smi -L 2>/dev/null | grep -Eq '^GPU [0-9]+:'
}

select_install_backend() {
  local has_amd=no has_nvidia=no answer=""

  # GFX_ARCH still chooses the ROCm wheel architecture; it is not an explicit
  # backend preference. Use THENOISE_BACKEND to skip the mixed-GPU prompt.
  if [ -n "${GFX_ARCH:-}" ]; then
    has_amd=yes
  elif GFX_ARCH="$(detect_gfx_arch)"; then
    has_amd=yes
    echo "Auto-detected GFX_ARCH=$GFX_ARCH"
  fi
  if detect_nvidia; then
    has_nvidia=yes
  fi

  if [ "$BACKEND_PREFERENCE" != auto ]; then
    INSTALL_BACKEND="$BACKEND_PREFERENCE"
  elif [ "$has_amd" = yes ]; then
    INSTALL_BACKEND=rocm
    if [ "$has_nvidia" = yes ]; then
      if [ -t 0 ]; then
        printf 'Detected AMD (%s) and NVIDIA GPUs. AMD/ROCm is the default. Install NVIDIA CUDA instead? [y/N] ' "$GFX_ARCH" >&2
        if IFS= read -r answer; then
          case "$answer" in
            y|Y|yes|YES|Yes) INSTALL_BACKEND=cuda ;;
          esac
        fi
      else
        echo 'Detected AMD and NVIDIA GPUs; defaulting to ROCm. Set THENOISE_BACKEND=cuda to use NVIDIA.' >&2
      fi
    fi
  elif [ "$has_nvidia" = yes ]; then
    INSTALL_BACKEND=cuda
  else
    INSTALL_BACKEND=rocm
    echo 'Warning: no GPU detected; falling back to ROCm/gfx1151.' >&2
  fi

  # Keep the original gfx1151 fallback for machines where sysfs probing fails
  # (or users who explicitly request ROCm without a detected GFX architecture).
  if [ "$INSTALL_BACKEND" = rocm ] && [ -z "${GFX_ARCH:-}" ]; then
    GFX_ARCH=gfx1151
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

if [ -z "$installed" ] || \
   { [ "$BACKEND_PREFERENCE" = rocm ] && [ "$backend" != rocm ]; } || \
   { [ "$BACKEND_PREFERENCE" = cuda ] && [ "$backend" != cuda ]; } || \
   { [ "$BACKEND_PREFERENCE" = auto ] && [ -n "${GFX_ARCH:-}" ] && [ "$backend" != rocm ] && ! detect_nvidia; }; then
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
