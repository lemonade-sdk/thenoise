#!/usr/bin/env bash
# build_portable.sh — Assemble a relocatable "portable" thenoise bundle.
#
# Produces a self-contained directory (standalone CPython + PyTorch ROCm + all
# deps + thenoise itself + a bundled clang so torch.compile/Triton JIT works
# without a system gcc) with a `bin/thenoise` launcher.
#
# The result is a single directory you can copy to any Linux x86_64 machine
# with a matching GPU and run without installing anything. This script is
# normally invoked from CI (see .github/workflows/build-thenoise-rocm.yml) but
# works standalone given a checkout + network + gcc + python3 (for the optional
# Triton precompile).
#
# Usage: build_portable.sh <gfx_target>
#
# Environment overrides (all optional):
#   THENOISE_ROOT   output bundle root  (default: $RUNNER_TEMP/thenoise-build/thenoise)
#   PBS_TAG         python-build-standalone release tag  (default: 20260602)
#   PBS_PY          CPython version from that tag        (default: 3.13.13)
#   PYVER           CPython ABI short tag                (default: 3.13)
#   TORCH_VER       torch version + rocm stamp           (default: 2.11.0+rocm7.14.0)
#   TORCHVISION_VER torchvision version + rocm stamp     (default: 0.26.0+rocm7.14.0)
#   TORCH_INDEX     AMD torch wheel index                (default: https://repo.amd.com/rocm/whl-multi-arch/)
set -euo pipefail

GFX_TARGET="${1:?usage: build_portable.sh <gfx_target>}"
GFX_ARCH="${GFX_TARGET//X/0}"
ROOT="${THENOISE_ROOT:-${RUNNER_TEMP:-/tmp}/thenoise-build/thenoise}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PBS_TAG="${PBS_TAG:-20260602}"
PBS_PY="${PBS_PY:-3.13.13}"
PYVER="${PYVER:-3.13}"
TORCH_VER="${TORCH_VER:-2.11.0+rocm7.14.0}"
TORCHVISION_VER="${TORCHVISION_VER:-0.26.0+rocm7.14.0}"
TORCH_INDEX="${TORCH_INDEX:-https://repo.amd.com/rocm/whl-multi-arch/}"

SP="lib/python${PYVER}/site-packages"
SP_DIR="$ROOT/$SP"
PY="$ROOT/bin/python${PYVER}"

say() { printf '\033[1;34m[build] %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[build:warning] %s\033[0m\n' "$*"; }

# pip helper with a raised recursion limit. Some large dependency graphs (e.g.
# vLLM's) make pip's resolvelib RecursionError; thenoise's is simpler, but this
# is cheap insurance and harmless.
pip_deep() {
  "$PY" -c 'import sys; sys.setrecursionlimit(20000); from pip._internal.cli.main import main; sys.exit(main(sys.argv[1:]))' "$@"
}

# ROCm wheel packages (e.g. rocm-sdk-core, which ships the HIP runtime) install
# ONLY versioned shared libraries (libamdhip64.so.7) with no unversioned
# libfoo.so dev symlink — that symlink normally comes from a system ROCm -dev
# package, so a relocatable wheel bundle is missing it. Consumers expect the
# bare name: torch's ctypes.CDLL("libamdhip64.so"), Triton's driver.py, and the
# precompiled hip_utils glue (which bakes the bare name into the C source). So
# create libfoo.so -> libfoo.so.<N> symlinks for every versioned library.
ensure_soname_symlinks() {
  local libdir
  for libdir in "$SP_DIR"/_rocm_sdk_*/lib "$SP_DIR"/torch/lib; do
    [ -d "$libdir" ] || continue
    # Process highest version first so the highest soname wins if a lib ships
    # multiple versions, and skip any lib that already has a symlink.
    local name base f
    for f in $(find "$libdir" -maxdepth 1 -name 'lib*.so.*' -printf '%p\n' 2>/dev/null | sort -Vr); do
      [ -e "$f" ] || continue
      name="${f##*/}"                 # libamdhip64.so.7
      base="${name%.so.*}.so"         # libamdhip64.so
      if [ -e "$libdir/$base" ] || [ -L "$libdir/$base" ]; then
        continue
      fi
      ln -s "$name" "$libdir/$base"
      say "linked $base -> $name"
    done
  done
}

# ---------------------------------------------------------------- 1. Python --
say "Downloading portable CPython ${PBS_PY} (${PBS_TAG})"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/cpython-${PBS_PY}+${PBS_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz"
# Extract into a dedicated sibling temp dir, then move into place. We only ever
# remove that clearly-named temp dir and the bundle root itself — never a
# user-supplied parent directory (safe for local testing, not just CI).
PY_TMP="${ROOT}-python-dl"
rm -rf "$PY_TMP"
mkdir -p "$PY_TMP"
curl -fsSL "$PBS_URL" | tar -xz -C "$PY_TMP"      # extracts to python/
rm -rf "$ROOT"
mkdir -p "$(dirname "$ROOT")"
mv "$PY_TMP/python" "$ROOT"
rm -rf "$PY_TMP"
"$PY" --version
# python-build-standalone install_only builds ship pip; make sure it + the
# build backend are current.
"$PY" -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------- 2. torch + deps --
say "Installing torch ${TORCH_VER} for ${GFX_TARGET}"
export PATH="$ROOT/bin:$PATH"
pip_deep install --index-url "$TORCH_INDEX" \
  --extra-index-url https://pypi.org/simple/ \
  "torch[device-${GFX_ARCH}]==${TORCH_VER}"
pip_deep install --index-url "$TORCH_INDEX" \
  --extra-index-url https://pypi.org/simple/ \
  "torchvision[device-${GFX_ARCH}]==${TORCHVISION_VER}"

say "Installing thenoise + dependencies (from $REPO_ROOT)"
# torch is intentionally absent from pyproject.toml. But a constraints file is cheap insurance
# to pin the already-installed ROCm torch/torchvision .
cat > /tmp/thenoise-constraints.txt <<EOF
torch==$TORCH_VER
torchvision==$TORCHVISION_VER
EOF
pip_deep install --constraint /tmp/thenoise-constraints.txt "$REPO_ROOT"

# ROCm wheels ship only versioned .so files (libfoo.so.N) without the unversioned
# libfoo.so dev symlinks consumers rely on. Create them now so the bundle is
# self-consistent (see ensure_soname_symlinks below).
say "Ensuring unversioned .so symlinks (libfoo.so -> libfoo.so.N)"
ensure_soname_symlinks


# ------------------------------------------------- 3. precompile Triton glue --
# Triton's ROCm backend JIT-compiles a small C module (hip_utils) from
# driver.c at first use, which needs gcc + Python.h on the TARGET machine.
# Precompile it now (CI has gcc) and patch Triton to load the prebuilt .so so
# the portable bundle never needs a C compiler for the glue. The actual Triton
# kernels are still JIT-compiled at runtime by the bundled clang below.
precompile_triton() {
  local TRITON_AMD="$SP_DIR/triton/backends/amd"
  local DRIVER_C="$TRITON_AMD/driver.c"
  local DRIVER_PY="$TRITON_AMD/driver.py"
  if [ ! -f "$DRIVER_C" ] || [ ! -f "$DRIVER_PY" ]; then
    warn "triton amd backend not found; skipping hip_utils precompile (clang bundling still covers runtime JIT)"
    return 0
  fi
  say "Pre-compiling Triton hip_utils for ${GFX_TARGET}"
  # Bake the bare HIP lib name in so it resolves via LD_LIBRARY_PATH at runtime.
  "$PY" - "$DRIVER_C" <<'PY'
import sys
src = open(sys.argv[1]).read()
src = src.replace('/*py_libhip_search_path*/', 'libamdhip64.so', 1)
open('/tmp/hip_utils.c', 'w').write(src)
print('wrote /tmp/hip_utils.c')
PY
  if ! gcc /tmp/hip_utils.c -O3 -shared -fPIC -Wno-psabi \
      -o "$TRITON_AMD/hip_utils_prebuilt.so" \
      -I"$TRITON_AMD/include" \
      -I"$ROOT/include/python${PYVER}"; then
    warn "gcc precompile of hip_utils failed; clang bundling still covers runtime JIT"
    return 0
  fi
  # Patch driver.py to load the prebuilt .so when present, else fall through to
  # the normal JIT compile. Best-effort: if the exact pattern isn't found for
  # this Triton version, warn and leave the file untouched (clang covers it).
  if grep -q 'compile_module_from_src(src=src, name="hip_utils"' "$DRIVER_PY"; then
    sed -i 's|mod = compile_module_from_src(src=src, name="hip_utils", include_dirs=include_dirs)|_pso = os.path.join(dirname, "hip_utils_prebuilt.so")\n        if os.path.exists(_pso):\n            import importlib.util as _ilu\n            _sp = _ilu.spec_from_file_location("hip_utils", _pso)\n            mod = _ilu.module_from_spec(_sp)\n            _sp.loader.exec_module(mod)\n        else:\n            mod = compile_module_from_src(src=src, name="hip_utils", include_dirs=include_dirs)|' "$DRIVER_PY"
    "$PY" -c "import ast; ast.parse(open('$DRIVER_PY').read()); print('patched driver.py — syntax OK')"
  else
    warn "compile_module_from_src(hip_utils) pattern not found; driver.py left as-is"
  fi
}
precompile_triton

# -------------------------------------------------------------- 4. launcher --
say "Creating launcher"
LAUNCHER_SRC="$(dirname "$0")/thenoise-launcher"
sed 's/^        //' "$LAUNCHER_SRC" > "$ROOT/bin/thenoise"
chmod +x "$ROOT/bin/thenoise"

# --------------------------------------------------------------- 5. trim ----
say "Trimming bundle size"
cd "$ROOT"

# pip/wheel are not needed at runtime (keep setuptools — torch's
# cpp_extension/Triton may need it).
rm -rf "$SP_DIR"/pip* "$SP_DIR"/wheel* 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# torch test/benchmark (NOT torch.testing — imported at runtime)
rm -rf "$SP_DIR/torch/test" "$SP_DIR/torch/benchmarks" 2>/dev/null || true

# Heavy packages not needed for inference (defensive — most are optional extras
# that were never installed).
rm -rf "$SP_DIR"/pyarrow* "$SP_DIR"/opencv* "$SP_DIR"/cv2* 2>/dev/null || true
rm -rf "$SP_DIR"/onnx* "$SP_DIR"/pandas* "$SP_DIR"/plotly* 2>/dev/null || true
rm -rf "$SP_DIR"/datasets* "$SP_DIR"/evaluate* "$SP_DIR"/peft* "$SP_DIR"/timm* 2>/dev/null || true
rm -rf "$SP_DIR"/boto3* "$SP_DIR"/botocore* "$SP_DIR"/s3transfer* 2>/dev/null || true
rm -rf "$SP_DIR"/tensorizer* 2>/dev/null || true

# Trim LLVM toolchain: keep only clang + its runtime libs (Triton JIT needs
# clang; clang-<ver> tracks the ROCm release, so match clang-* not a hardcoded
# major). ~350MB saved.
LLVM="$SP_DIR/_rocm_sdk_core/lib/llvm"
find "$LLVM/bin" -type f ! -name "clang" ! -name "clang-*" -delete 2>/dev/null || true
find "$LLVM/bin" -type l ! -name "clang" -delete 2>/dev/null || true
find "$LLVM/lib" -name "*.so*" \
  ! -name "libclang-cpp*" \
  ! -name "libLLVM*" \
  -delete 2>/dev/null || true
rm -rf "$LLVM/lib/clang/"*/include/cuda_wrappers 2>/dev/null || true

# ROCm SDK python wrapper scripts aren't needed.
rm -rf "$SP_DIR/_rocm_sdk_core/bin" 2>/dev/null || true

# Trim Python stdlib we don't need. Keep include/ — Triton JIT needs Python.h.
rm -rf "lib/python${PYVER}/test" "lib/python${PYVER}/tkinter" "lib/python${PYVER}/idlelib" 2>/dev/null || true
rm -rf "lib/python${PYVER}/turtledemo" "lib/python${PYVER}/ensurepip" 2>/dev/null || true

# Ensure the bundled clang is executable (permissions can be lost in tar).
chmod +x "$LLVM/bin/clang"* 2>/dev/null || true

# Do NOT strip .so files — AMD ROCm wheels use special ELF alignment that strip
# corrupts.

# ------------------------------------------------------------- 6. verify ----
say "Verifying bundle"
verify_env() {
  local d
  for d in "$SP_DIR"/_rocm_sdk_*/lib "$SP_DIR"/torch/lib; do
    [ -d "$d" ] && LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
  done
  export LD_LIBRARY_PATH
}
verify_env
"$PY" -c "import torch; assert 'rocm' in torch.__version__, torch.__version__; print('torch', torch.__version__)"
"$PY" -c "import torchvision; print('torchvision', torchvision.__version__)"
"$PY" -c "import thenoise; print('thenoise import OK')"
"$PY" -m thenoise --help >/dev/null 2>&1 || true
bash -n "$ROOT/bin/thenoise"
echo "=== Bundle size ==="
du -sh "$ROOT"
du -sh "$SP_DIR"/torch "$SP_DIR"/_rocm_sdk_core "$SP_DIR"/triton 2>/dev/null || true

# Export the root for later workflow steps (artifact upload).
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "THENOISE_ROOT=$ROOT" >> "$GITHUB_ENV"
fi

say "Done: $ROOT"
