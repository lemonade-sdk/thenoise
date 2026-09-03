#!/usr/bin/env bash
# qualify_thenoise.sh — GPU smoke test for a portable thenoise bundle.
#
# Runs on a real Strix Halo / Strix Point box (gfx1151). Verifies the bundle
# imports, sees the GPU, downloads the Anima model, and runs one real
# generation end-to-end (including a first-run torch.compile). Writes a small
# JSON report.
#
# Usage: qualify_thenoise.sh --root <bundle_root> --model-dir <dir>
#                           [--out <report.json>] [--variant <anima-variant>]
set -euo pipefail

ROOT="" MODEL_DIR="" REPORT="" VARIANT="turbo-v1.0"
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --out) REPORT="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ROOT" ] && [ -n "$MODEL_DIR" ] || { echo "usage: qualify_thenoise.sh --root <dir> --model-dir <dir> [--out <json>]" >&2; exit 2; }
REPORT="${REPORT:-${MODEL_DIR}/../qualification.json}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/bin/python3"
SP="$(echo "$ROOT"/lib/python3.*/site-packages)"
OUT="$(dirname "$REPORT")"
mkdir -p "$OUT"
PASS=0; FAIL=1

env_setup() {
  for libdir in "$SP"/_rocm_sdk_*/lib "$SP"/torch/lib; do
    [ -d "$libdir" ] && LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
  done
  LD_LIBRARY_PATH="$SP/_rocm_sdk_core/lib/llvm/lib:${LD_LIBRARY_PATH}"
  export LD_LIBRARY_PATH
  # upload-artifact drops the +x bit and unversioned .so symlinks; restore them.
  chmod +x "$ROOT"/bin/* 2>/dev/null || true
  chmod +x "$SP/_rocm_sdk_core/lib/llvm/bin/"clang* 2>/dev/null || true
  local ROCM_LIB="$SP/_rocm_sdk_core/lib"
  local lib link name
  for lib in "$ROCM_LIB"/*.so.*; do
    [ -e "$lib" ] || continue
    name="$(basename "$lib")"; link="$ROCM_LIB/${name%%.*}.so"
    if [ ! -e "$link" ]; then ln -sf "$name" "$link"; fi
  done
}

report() { # report <status> <msg>
  python3 - "$REPORT" "$1" "$2" <<'PY'
import json, sys
json.dump({"status": sys.argv[2], "message": sys.argv[3]}, open(sys.argv[1], "w"), indent=2)
PY
}

env_setup

echo "=== import checks ==="
if ! "$PY" -c "import torch; print('torch', torch.__version__)" > "$OUT/torch.txt" 2>&1; then
  cat "$OUT/torch.txt"; report "$FAIL" "torch import failed"; exit 1
fi
cat "$OUT/torch.txt"
"$PY" -c "import thenoise; print('thenoise import OK')" || { report "$FAIL" "thenoise import failed"; exit 1; }

echo "=== GPU check ==="
if ! "$PY" -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0; print('GPU', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))" > "$OUT/gpu.txt" 2>&1; then
  cat "$OUT/gpu.txt"; report "$FAIL" "no usable GPU"; exit 1
fi
cat "$OUT/gpu.txt"

echo "=== download Anima model ($VARIANT) ==="
# Cache the model on the persistent self-hosted runner across runs.
CACHE_HINT="$MODEL_DIR/split_files/diffusion_models/anima-$VARIANT.safetensors"
if [ -f "$CACHE_HINT" ]; then
  echo "model already present; using cache"
else
  "$PY" "$REPO_ROOT/scripts/download_anima.py" --out "$MODEL_DIR" --variant "$VARIANT"
fi

DIT="$MODEL_DIR/split_files/diffusion_models/anima-$VARIANT.safetensors"
VAE="$MODEL_DIR/split_files/vae/qwen_image_vae.safetensors"
TE="$MODEL_DIR/split_files/text_encoders/qwen_3_06b_base.safetensors"
for f in "$DIT" "$VAE" "$TE"; do
  [ -f "$f" ] || { echo "missing checkpoint: $f"; report "$FAIL" "missing checkpoint $f"; exit 1; }
done

echo "=== run one generation (first run compiles via torch.compile) ==="
# Give the first-run compile a generous timeout (it traces + compiles kernels).
if ! timeout 1800 "$ROOT/bin/thenoise" generate \
    --dit "$DIT" --vae "$VAE" --text-encoder "$TE" \
    --prompt "a fox walking in the snow" --steps 8 --guidance-scale 1 \
    --width 256 --height 256 \
    --out "$OUT/qualification.png" > "$OUT/generate.log" 2>&1; then
  tail -40 "$OUT/generate.log"
  report "$FAIL" "generation failed"
  exit 1
fi
tail -15 "$OUT/generate.log"
if [ ! -s "$OUT/qualification.png" ]; then
  report "$FAIL" "no output PNG produced"
  exit 1
fi
echo "output PNG: $(du -h "$OUT/qualification.png" | cut -f1)"

report "$PASS" "bundle qualified on gfx1151 (imports + GPU + Anima generation OK)"
echo "=== qualification PASS ==="
