#!/usr/bin/env bash
# create_portable_archive.sh — Package existing portable thenoise bundle(s) into
# the same tar.gz artifact(s) the CI workflow produces, and print the intended
# release tags. Use this to produce release artifacts manually while CI is down,
# after building bundle(s) with build_portable.sh (+ running qualify).
#
# GFX target(s) are derived automatically from the bundle root:
#   * If <root> is itself a bundle (has bin/python*), package just it. Its gfx
#     target is taken from a trailing -gfxNNNN in the dir name, or via --gfx.
#   * Otherwise every gfx* subdirectory under <root> is treated as a bundle and
#     all are packaged (GFX = the gfx token in the subdir name). This lets you
#     build every target into one parent dir (build_portable.sh is
#     GPU-independent: the gfx target only selects which ROCm torch wheel is
#     installed) and package them all in one go.
#
# Usage: create_portable_archive.sh [root_dir] [--gfx gfxNNNN]
#   root_dir  directory containing the portable bundle(s)
#             (default: $THENOISE_ROOT)
#   --gfx     explicit gfx target override for a single-bundle root
#
# Env: SPLIT_MB  per-part size threshold in MB for splitting (default 1900)
#
# Mirrors the "Generate release tag" + "Create archive (split if >1.9 GB ...)"
# steps in .github/workflows/build-thenoise-rocm.yml. Outputs (in the current
# working directory, one per bundle):
#   <tag>-<gfx>-x64.tar.gz                           (single archive if <= SPLIT_MB)
#   <tag>-<gfx>-x64.partNN-of-TT.tar.gz + .partcount (if larger; GitHub release limit)
# where <tag> is the release tag (thenoise-<version>-rocm<rocm>).
set -euo pipefail

ROOT="${1:-${THENOISE_ROOT:-}}"
GFX_OVERRIDE=""
case "${2:-}" in
  --gfx) GFX_OVERRIDE="${3:?--gfx requires a value}" ;;
esac
SPLIT_MB="${SPLIT_MB:-1900}"

if [ -z "$ROOT" ]; then
  echo "error: no bundle root given (pass as arg 1 or set \$THENOISE_ROOT)" >&2
  exit 1
fi
[ -d "$ROOT" ] || { echo "error: root not found: $ROOT" >&2; exit 1; }

say() { printf '\033[1;34m[archive] %s\033[0m\n' "$*"; }

# package_bundle <bundle_root> <gfx_target>
package_bundle() {
  local bundle="$1" gfx="$2"
  local PY
  PY="$(ls "$bundle"/bin/python[0-9.]* 2>/dev/null | head -n1 || true)"
  if [ -z "$PY" ]; then
    echo "error: no python in $bundle/bin; not a portable thenoise bundle?" >&2
    exit 1
  fi

  # Read versions the same way the workflow does (Set job outputs step).
  local thenoise_version torch_version rocm
  thenoise_version=$("$PY" -c "import importlib.metadata as m; print(m.version('thenoise'))")
  torch_version=$("$PY" -c "import torch; print(torch.__version__)")
  rocm="$(printf '%s' "$torch_version" | grep -oP 'rocm[\d.]+' || echo rocm)"

  # Release tag scheme (matches CI): thenoise-<version>-rocm<rocm>. The gfx
  # target is NOT part of the tag — it only distinguishes the archive files.
  local tag base
  tag="thenoise-${thenoise_version}-${rocm}"
  base="${tag}-${gfx}-x64"

  say "Bundle root:  $bundle"
  say "TheNoise:     $thenoise_version"
  say "PyTorch:      $torch_version"
  say "GPU target:   $gfx"
  say "Release tag:  $tag"

  say "Creating: ${base}.tar.gz"
  tar -czf "${base}.tar.gz" -C "$bundle" .
  local size_mb
  size_mb=$(du -m "${base}.tar.gz" | cut -f1)
  echo "[archive] size: ${size_mb} MB"
  if [ "$size_mb" -gt "$SPLIT_MB" ]; then
    echo "[archive] splitting into parts (${SPLIT_MB} MB each)..."
    split -b ${SPLIT_MB}M --numeric-suffixes=1 -a 2 --additional-suffix=.tar.gz \
      "${base}.tar.gz" "${base}.part"
    rm "${base}.tar.gz"
    parts=( "${base}".part*.tar.gz )
    local total total_padded nn
    total=${#parts[@]}; total_padded=$(printf "%02d" "$total")
    for f in "${parts[@]}"; do
      nn="${f##*.part}"; nn="${nn%%.tar.gz}"
      mv "$f" "${base}.part${nn}-of-${total_padded}.tar.gz"
    done
    echo "$total" > "${base}.partcount"
    echo "[archive] split into $total part(s)"
  else
    echo "[archive] single archive (under ${SPLIT_MB} MB)"
  fi

  say "Archive files:"
  ls -la "${base}".* 2>/dev/null || true
  say "Release name: v${thenoise_version}"
}

# Mode B — <root> is itself a single bundle.
if ls "$ROOT"/bin/python[0-9.]* >/dev/null 2>&1; then
  if [ -n "$GFX_OVERRIDE" ]; then
    gfx="$GFX_OVERRIDE"
  else
    gfx="$(basename "$ROOT" | grep -oE 'gfx[0-9]+$' || true)"
    if [ -z "$gfx" ]; then
      echo "error: cannot derive gfx target from '$ROOT'; pass --gfx gfxNNNN" >&2
      exit 1
    fi
  fi
  package_bundle "$ROOT" "$gfx"
else
  # Mode A — <root> contains one or more gfx* bundle subdirs; package all.
  found=0
  for bundle in "$ROOT"/gfx*; do
    [ -d "$bundle" ] || continue
    gfx="$(basename "$bundle" | grep -oE 'gfx[0-9]+$')"
    package_bundle "$bundle" "$gfx"
    found=1
  done
  if [ "$found" -eq 0 ]; then
    echo "error: no gfx* bundle subdirs found under $ROOT" >&2
    exit 1
  fi
fi
