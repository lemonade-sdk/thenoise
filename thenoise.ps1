# ---------------------------------------------------------------------------
# thenoise.ps1 — Bootstrap the venv (if needed) and launch the project on
# Windows. All CLI arguments are forwarded to `python -m thenoise`.
#
# Invoke via the thin `thenoise.bat` wrapper (double-click friendly) or
# directly from a terminal:
#   powershell -NoProfile -ExecutionPolicy Bypass -File thenoise.ps1 [args...]
# ---------------------------------------------------------------------------
param([Parameter(ValueFromRemainingArguments = $true)]$ForwardArgs)

$ErrorActionPreference = "Stop"

$ProjectDir = $PSScriptRoot
if (-not $ProjectDir) { $ProjectDir = (Get-Location).Path }
$VenvDir = Join-Path $ProjectDir ".venv"

# ---- 1. Check that uv is available ----------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host @"
Error: uv is not installed.

Install it with:
  powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

Then open a new terminal and try again.
"@
  exit 1
}

# ---- 2. Create the venv if it does not exist ------------------------------
if (-not (Test-Path $VenvDir)) {
  Write-Host "Creating virtual environment ($VenvDir) with Python 3.13 ..."
  # --managed-python forces uv to use its own standalone CPython build rather
  # than a system python3.13. (On Windows, Triton JIT is unavailable anyway;
  # this still gives a self-contained, sudo-free setup.)
  & uv venv $VenvDir --python 3.13 --managed-python
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# ---- 3. Install torch (ROCm build) ----------------------------------------
$Py = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $Py)) {
  Write-Host "Error: venv python not found: $Py" -ForegroundColor Red
  exit 1
}

$torchInstalled = $false
try { & $Py -c "import torch" 2>$null; if ($LASTEXITCODE -eq 0) { $torchInstalled = $true } } catch {}

if (-not $torchInstalled) {
  Write-Host "Installing ROCm torch ..."

  function Get-GfxArch {
    # An explicit GFX_ARCH env var always wins over auto-detection.
    if ($env:GFX_ARCH) { return $env:GFX_ARCH }
    # Best-effort Windows auto-detect from the AMD GPU name. This mapping is a
    # heuristic and may need tuning for your GPU; set $env:GFX_ARCH to override.
    try {
      $name = (Get-CimInstance Win32_VideoController -ErrorAction Stop |
        Select-Object -ExpandProperty Name -First 1)
      if ($name -match "Strix|Radeon 800M|Ryzen AI 300|890M|880M") { return "gfx1151" }
      if ($name -match "Radeon RX 9000") { return "gfx1150" }
      if ($name -match "Radeon RX 7") { return "gfx1100" }
    } catch {}
    return "gfx1151"
  }

  $GFX = Get-GfxArch
  if ($env:GFX_ARCH) { Write-Host "GFX_ARCH=$GFX (from environment)" }
  else { Write-Host "Auto-detected GFX_ARCH=$GFX (set `$env:GFX_ARCH to override)" }

  & uv pip install `
    "torch[device-$GFX]==2.11.0" `
    "torchvision[device-$GFX]==0.26.0" `
    --index-url https://repo.amd.com/rocm/whl-multi-arch/
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# ---- 4. Install the project in editable mode ------------------------------
& uv pip install -e $ProjectDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---- 5. Set ROCm-specific environment variables ---------------------------
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"
$env:MIOPEN_FIND_MODE = "FAST"
$env:TORCH_BLAS_PREFER_HIPBLASLT = "1"
# Triton (torch.compile) is not yet available on Windows ROCm (AMD's Triton
# wheels are Linux-only), so disable it. Respect an existing override in case
# the user has a working Windows Triton.
if (-not $env:TORCHDYNAMO_DISABLE) { $env:TORCHDYNAMO_DISABLE = "1" }

# ---- 6. Launch the project, forwarding all arguments ----------------------
& $Py -m thenoise @ForwardArgs
exit $LASTEXITCODE
