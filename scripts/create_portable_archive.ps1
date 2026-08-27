# ===========================================================================
# create_portable_archive.ps1 — Package existing portable thenoise bundle(s)
# into release archive(s) the CI workflow produces, and print the intended
# release tags. Windows counterpart of scripts/create_portable_archive.sh
# (which emits .tar.gz); this script emits .zip.
#
# GFX target(s) are derived automatically from the bundle root:
#   * If <root> is itself a bundle (has python.exe), package just it. Its gfx
#     target is taken from a trailing -gfxNNNN in the dir name, or via -Gfx.
#   * Otherwise every gfx* subdirectory under <root> is treated as a bundle and
#     all are packaged (GFX = the gfx token in the subdir name). This lets you
#     build every target into one parent dir and package them all in one go.
#
# Usage: create_portable_archive.ps1 [root_dir] [-Gfx gfxNNNN]
#   root_dir  directory containing the portable bundle(s)
#             (default: $env:THENOISE_ROOT)
#   -Gfx      explicit gfx target override for a single-bundle root
#
# Env: SPLIT_MB  per-part size threshold in MB for splitting (default 1900)
#
# Outputs (in the current working directory, one per bundle):
#   <tag>-<gfx>-win-x64.zip                          (single archive if <= SPLIT_MB)
#   <tag>-<gfx>-win-x64.partNN-of-TT.zip + .partcount (if larger)
# where <tag> is the release tag (thenoise-<version>-rocm<rocm>). The "-win"
# in the file name distinguishes Windows bundles from the Linux "-x64" ones.
# Windows bundles are shipped as .zip (native on Windows; there are no symlinks
# to preserve — the ROCm DLLs are plain files).
#
# tar (bsdtar) is used for archiving (built into Windows 10+); it writes a zip
# when the output name ends in .zip. GNU split is not available on Windows, so
# splitting is done with a PowerShell chunker.
# ===========================================================================
param(
  [Parameter(Position = 0)] [string]$RootDir,
  [Alias("gfx")] [string]$Gfx
)

$ErrorActionPreference = "Stop"

if (-not $RootDir) { $RootDir = $env:THENOISE_ROOT }
if (-not $RootDir) {
  Write-Host "error: no bundle root given (pass as arg 1 or set `$env:THENOISE_ROOT)" -ForegroundColor Red
  exit 1
}
if (-not (Test-Path $RootDir -PathType Container)) {
  Write-Host "error: root not found: $RootDir" -ForegroundColor Red
  exit 1
}
$SplitMb = if ($env:SPLIT_MB) { [int]$env:SPLIT_MB } else { 1900 }

function say { param([string]$m) Write-Host "[archive] $m" -ForegroundColor Blue }

# Split-File <path> <partSizeMb>
# Chunks a file into <path>.partNN-of-TT.zip parts and writes <path>.partcount.
function Split-File {
  param([string]$Path, [int]$PartSizeMb)
  $chunk = $PartSizeMb * 1024 * 1024
  $total = [math]::Ceiling((Get-Item $Path).Length / $chunk)
  $fs = [System.IO.File]::OpenRead($Path)
  try {
    $buf = New-Object byte[] $chunk
    for ($i = 0; $i -lt $total; $i++) {
      $read = $fs.Read($buf, 0, $chunk)
      $partName = "{0}.part{1:00}-of-{2:00}.zip" -f $Path, ($i + 1), $total
      $out = [System.IO.File]::OpenWrite($partName)
      try { $out.Write($buf, 0, $read) } finally { $out.Close() }
    }
  } finally { $fs.Close() }
  Remove-Item $Path -Force
  Set-Content -Path "$Path.partcount" -Value "$total"
  say "split into $total part(s)"
}

# Package-Bundle <bundle_root> <gfx_target>
function Package-Bundle {
  param([string]$bundle, [string]$gfx)
  $py = Get-ChildItem $bundle -Filter "python.exe" -File -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $py) {
    Write-Host "error: no python.exe in $bundle; not a portable thenoise bundle?" -ForegroundColor Red
    exit 1
  }
  $Python = $py.FullName

  # Read versions the same way the Linux workflow does.
  $thenoiseVersion = (& $Python -c "import importlib.metadata as m; print(m.version('thenoise'))").Trim()
  $torchVersion = (& $Python -c "import torch; print(torch.__version__)").Trim()
  $rocm = "rocm"
  if ($torchVersion -match 'rocm[\d.]+') { $rocm = $Matches[0] }

  # Release tag scheme (matches CI): thenoise-<version>-rocm<rocm>. The gfx
  # target is NOT part of the tag - it only distinguishes the archive files.
  $tag = "thenoise-${thenoiseVersion}-${rocm}"
  $base = "${tag}-${gfx}-win-x64"

  say "Bundle root:  $bundle"
  say "TheNoise:     $thenoiseVersion"
  say "PyTorch:      $torchVersion"
  say "GPU target:   $gfx"
  say "Release tag:  $tag"

  say "Creating: ${base}.zip"
  # -a lets bsdtar auto-detect the format from the .zip extension and write a
  # real zip (without -a it writes a gzip tar).
  tar -a -cf "${base}.zip" -C $bundle .
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  $sizeMb = [math]::Ceiling((Get-Item "${base}.zip").Length / 1MB)
  say "size: ${sizeMb} MB"
  if ($sizeMb -gt $SplitMb) {
    say "splitting into parts (${SplitMb} MB each)..."
    Split-File "${base}.zip" $SplitMb
    say "Archive files:"
    Get-ChildItem "${base}.*" -ErrorAction SilentlyContinue | ForEach-Object { say $_.Name }
  } else {
    say "single archive (under ${SplitMb} MB)"
    say "Archive files:"
    Get-ChildItem "${base}.*" -ErrorAction SilentlyContinue | ForEach-Object { say $_.Name }
  }

  say "Release name: v${thenoiseVersion}"
}

# Mode B - <root> is itself a single bundle.
if (Get-ChildItem $RootDir -Filter "python.exe" -File -ErrorAction SilentlyContinue) {
  if ($Gfx) {
    $gfx = $Gfx
  } else {
    $gfx = [regex]::Match((Split-Path $RootDir -Leaf), 'gfx[0-9]+$').Value
    if (-not $gfx) {
      Write-Host "error: cannot derive gfx target from '$RootDir'; pass -Gfx gfxNNNN" -ForegroundColor Red
      exit 1
    }
  }
  Package-Bundle $RootDir $gfx
} else {
  # Mode A - <root> contains one or more gfx* bundle subdirs; package all.
  $found = $false
  foreach ($bundle in (Get-ChildItem $RootDir -Directory -Filter "gfx*" -ErrorAction SilentlyContinue)) {
    $gfx = [regex]::Match($bundle.Name, 'gfx[0-9]+$').Value
    if (-not $gfx) { continue }
    Package-Bundle $bundle.FullName $gfx
    $found = $true
  }
  if (-not $found) {
    Write-Host "error: no gfx* bundle subdirs found under $RootDir" -ForegroundColor Red
    exit 1
  }
}
