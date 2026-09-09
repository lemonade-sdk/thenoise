# ===========================================================================
# qualify_thenoise.ps1 — GPU smoke test for a portable Windows thenoise bundle.
# Windows counterpart of build-scripts/qualify_thenoise.sh.
#
# Runs on a real Strix Halo / Strix Point Windows box (gfx1151). Verifies the
# bundle imports, sees the GPU, downloads the Anima model, and runs one real
# generation end-to-end. Writes a small JSON report.
#
# NOTE: torch.compile is disabled on Windows (Triton is Linux-only in the AMD
# wheel index), so the generation runs in eager mode — there is no first-run
# compile step as in the Linux qualification.
#
# Usage: qualify_thenoise.ps1 -Root <bundle_root> -ModelDir <dir>
#                            [-Out <report.json>] [-Variant <anima-variant>]
# ===========================================================================
param(
  [Parameter(Mandatory = $true)][string]$Root,
  [Parameter(Mandatory = $true)][string]$ModelDir,
  [string]$Out,
  [string]$Variant = "turbo-v1.0"
)

$ErrorActionPreference = "Stop"

if (-not $Out) { $Out = Join-Path (Split-Path $ModelDir -Parent) "qualification.json" }
$SPDir = Join-Path $Root "Lib\site-packages"
$Py = Join-Path $Root "python.exe"
$Launcher = Join-Path $Root "thenoise.bat"
if (-not (Test-Path $Py)) {
  Write-Error "no python.exe in $Root; not a portable thenoise bundle?"
  exit 1
}

$OutDir = Split-Path -Parent $Out
New-Item -ItemType Directory -Force $OutDir | Out-Null
$Pass = 0
$Fail = 1

function Report {
  param([int]$Status, [string]$Message)
  @{ status = $Status; message = $Message } | ConvertTo-Json | Set-Content -Path $Out
}

function Fail-Report {
  param([string]$Message)
  Report $Fail $Message
  Write-Host "qualification FAIL: $Message" -ForegroundColor Red
  exit 1
}

# DLL resolution on Windows is PATH-based: bundled ROCm runtime + torch libs.
$env:PATH = "$SPDir\_rocm_sdk_core\bin;$SPDir\_rocm_sdk_core\lib;$SPDir\_rocm_sdk_core\lib\llvm\lib;$SPDir\_rocm_sdk_libraries\bin;$SPDir\torch\lib;$SPDir;$Root;${Root}\Scripts;$env:PATH"
# torch.compile is unavailable on Windows ROCm (no Triton); disable it.
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"
$env:TORCH_COMPILE_DISABLE = "1"
$env:TORCHDYNAMO_DISABLE = "1"
$env:MIOPEN_FIND_MODE = "FAST"
$env:TORCH_BLAS_PREFER_HIPBLASLT = "1"

Write-Host "=== import checks ==="
try {
  $tv = (& $Py -c "import torch; print(torch.__version__)").Trim()
  Write-Host "torch $tv"
} catch {
  Fail-Report "torch import failed: $($_.Exception.Message)"
}
try {
  & $Py -c "import thenoise"
  if ($LASTEXITCODE -ne 0) { throw "non-zero exit" }
  Write-Host "thenoise import OK"
} catch {
  Fail-Report "thenoise import failed: $($_.Exception.Message)"
}

Write-Host "=== GPU check ==="
try {
  $gpu = (& $Py -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0; print(torch.cuda.get_device_name(0))").Trim()
  Write-Host "GPU $gpu"
} catch {
  Fail-Report "no usable GPU: $($_.Exception.Message)"
}

Write-Host "=== download Anima model ($Variant) ==="
# Cache the model on the persistent self-hosted runner across runs.
$Dit = Join-Path $ModelDir "split_files\diffusion_models\anima-$Variant.safetensors"
if (Test-Path $Dit) {
  Write-Host "model already present; using cache"
} else {
  & $Py (Join-Path (Split-Path -Parent (Split-Path -Parent $PSCommandPath)) "scripts\download_anima.py") --out $ModelDir --variant $Variant
  if ($LASTEXITCODE -ne 0) { Fail-Report "model download failed" }
}
$Vae = Join-Path $ModelDir "split_files\vae\qwen_image_vae.safetensors"
$Te = Join-Path $ModelDir "split_files\text_encoders\qwen_3_06b_base.safetensors"
foreach ($f in @($Dit, $Vae, $Te)) {
  if (-not (Test-Path $f)) { Fail-Report "missing checkpoint $f" }
}

Write-Host "=== run one generation (compile disabled on Windows) ==="
$outPng = Join-Path $OutDir "qualification.png"
$genLog = Join-Path $OutDir "generate.log"
# Run the bundle launcher via cmd so $LASTEXITCODE reliably reflects the .bat
# exit code, and capture all output to generate.log.
$cmdLine = "`"$Launcher`" generate --dit `"$Dit`" --vae `"$Vae`" --text-encoder `"$Te`" " +
  "--prompt `"a fox walking in the snow`" --steps 8 --guidance-scale 1 " +
  "--width 256 --height 256 --out `"$outPng`""
cmd /c "$cmdLine > `"$genLog`" 2>&1"
$genExit = $LASTEXITCODE
if ($genExit -ne 0) {
  if (Test-Path $genLog) { Get-Content $genLog -Tail 40 }
  Fail-Report "generation failed (exit $genExit); see generate.log"
}
if (-not (Test-Path $outPng)) { Fail-Report "no output PNG produced" }
Write-Host "output PNG: $((Get-Item $outPng).Length) bytes"

Report $Pass "bundle qualified on gfx1151 (imports + GPU + Anima generation OK)"
Write-Host "=== qualification PASS ==="
