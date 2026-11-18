# ===========================================================================
# qualify_thenoise.ps1 — GPU smoke test for a portable Windows thenoise bundle.
# Windows counterpart of build-scripts/qualify_thenoise.sh.
#
# Runs on a real Strix Halo / Strix Point Windows box (gfx1151). Verifies the
# bundle imports, sees the GPU, downloads the Anima model, and runs the same
# small generation twice: once the way the launcher runs it, which has to pass,
# and once with compilation turned on, which only reports. Writes a small JSON
# report.
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
  param([int]$Status, [string]$Message, [hashtable]$Extra = @{})
  (@{ status = $Status; message = $Message } + $Extra) | ConvertTo-Json -Depth 4 | Set-Content -Path $Out
}

function Fail-Report {
  param([string]$Message)
  Report $Fail $Message
  Write-Host "qualification FAIL: $Message" -ForegroundColor Red
  exit 1
}

# DLL resolution on Windows is PATH-based: bundled ROCm runtime + torch libs.
$env:PATH = "$SPDir\_rocm_sdk_core\bin;$SPDir\_rocm_sdk_core\lib;$SPDir\_rocm_sdk_core\lib\llvm\lib;$SPDir\_rocm_sdk_libraries\bin;$SPDir\torch\lib;$SPDir;$Root;${Root}\Scripts;$env:PATH"
# The ROCm environment thenoise.bat sets, for the python calls below that skip it.
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"
$env:MIOPEN_FIND_MODE = "FAST"

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
  & $Py (Join-Path (Split-Path -Parent (Split-Path -Parent $PSCommandPath)) "scripts\download.py") --model anima --out $ModelDir --variant $Variant
  if ($LASTEXITCODE -ne 0) { Fail-Report "model download failed" }
}
$Vae = Join-Path $ModelDir "split_files\vae\qwen_image_vae.safetensors"
$Te = Join-Path $ModelDir "split_files\text_encoders\qwen_3_06b_base.safetensors"
foreach ($f in @($Dit, $Vae, $Te)) {
  if (-not (Test-Path $f)) { Fail-Report "missing checkpoint $f" }
}

function Invoke-Generation {
  param([string]$Label, [string]$DynamoDisable)
  $png = Join-Path $OutDir "qualification-$Label.png"
  $log = Join-Path $OutDir "generate-$Label.log"
  Remove-Item $png -ErrorAction SilentlyContinue
  # An empty $DynamoDisable keeps the variable out of the environment, which is
  # how thenoise.bat decides for itself - the path an installed bundle takes.
  if ($DynamoDisable) { $env:TORCHDYNAMO_DISABLE = $DynamoDisable }
  else { Remove-Item Env:TORCHDYNAMO_DISABLE -ErrorAction SilentlyContinue }
  Write-Host "=== run one generation ($Label; TORCHDYNAMO_DISABLE='${env:TORCHDYNAMO_DISABLE}') ==="
  $cmdLine = "`"$Launcher`" generate --dit `"$Dit`" --vae `"$Vae`" --text-encoder `"$Te`" " +
    "--prompt `"a fox walking in the snow`" --steps 8 --guidance-scale 1 " +
    "--width 256 --height 256 --out `"$png`""
  # Run the bundle launcher via cmd so $LASTEXITCODE reliably reflects the .bat
  # exit code, and capture all output to the log.
  $started = Get-Date
  cmd /c "$cmdLine > `"$log`" 2>&1"
  $exit = $LASTEXITCODE
  $bytes = 0
  if (Test-Path $png) { $bytes = (Get-Item $png).Length }
  if ($exit -ne 0 -or $bytes -eq 0) { Get-Content $log -Tail 40 -ErrorAction SilentlyContinue | Out-Host }
  $secs = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
  Write-Host "generation $Label : exit $exit, $bytes bytes, $secs s"
  return @{ seconds = $secs; exit = $exit; png_bytes = $bytes }
}

# The launcher leaves compilation off, so the first run asks for nothing and it
# has to pass. The second asks for compilation (any value other than 1 does) and
# only reports, because it needs a C toolchain on this machine.
$eager = Invoke-Generation -Label "eager"
if ($eager.exit -ne 0 -or $eager.png_bytes -eq 0) {
  Fail-Report "generation failed (exit $($eager.exit)); see generate-eager.log"
}
$compiled = Invoke-Generation -Label "compiled" -DynamoDisable "0"
if ($compiled.exit -ne 0 -or $compiled.png_bytes -eq 0) {
  Write-Host "WARNING: the compiled generation failed (exit $($compiled.exit)); it gates nothing, and needs the C++ build tools on this machine" -ForegroundColor Yellow
}

Report $Pass "bundle qualified on gfx1151 (imports + GPU + Anima generation OK)" @{
  eager_seconds    = $eager.seconds
  compiled_seconds = $compiled.seconds
  compiled_ok      = ($compiled.exit -eq 0 -and $compiled.png_bytes -gt 0)
}
Write-Host "=== qualification PASS ==="
