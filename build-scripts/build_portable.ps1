# ===========================================================================
# build_portable.ps1 — Assemble a relocatable "portable" thenoise bundle for
# Windows (AMD ROCm).
#
# Produces a self-contained directory (standalone CPython + PyTorch ROCm
# Windows wheels + the bundled ROCm runtime DLLs from rocm-sdk-core + thenoise
# itself) with a `thenoise.bat` launcher at the root. Copy the directory to any
# Windows x86_64 machine with a matching AMD GPU and run `thenoise.bat` with
# nothing installed.
#
# This is the Windows counterpart of build-scripts/build_portable.sh. Key differences
# from the Linux build:
#   * Portable CPython is the x86_64-pc-windows-msvc python-build-standalone
#     build. The interpreter is python.exe at the bundle root and
#     site-packages is Lib\site-packages (not bin/python3.13 +
#     lib/pythonX/site-packages).
#   * DLLs resolve via PATH (not LD_LIBRARY_PATH).
#   * Triton (torch.compile) is Linux-only in the AMD wheel index, so there is
#     no Triton hip_utils precompile step. Compilation is disabled via the
#     torch env var TORCH_COMPILE_DISABLE in the launcher.
#   * ROCm runtime DLLs live in _rocm_sdk_core\bin (amdhip64_7.dll, hiprtc,
#     amd_comgr.dll, ...) and are kept on PATH by the launcher; unlike the
#     Linux build, _rocm_sdk_core\bin must NOT be removed.
#
# Usage: build_portable.ps1 <gfx_target>
#
# Environment overrides (all optional):
#   THENOISE_ROOT   output bundle root  (default: $env:RUNNER_TEMP\thenoise-build\thenoise)
#   PBS_TAG         python-build-standalone release tag  (default: 20260602)
#   PBS_PY          CPython version from that tag        (default: 3.13.13)
#   PYVER           CPython ABI short tag                (default: 3.13)
#   TORCH_VER       torch version + rocm stamp           (default: 2.11.0+rocm7.14.0)
#   TORCHVISION_VER torchvision version + rocm stamp     (default: 0.26.0+rocm7.14.0)
#   TORCH_INDEX     AMD torch wheel index                (default: https://repo.amd.com/rocm/whl-multi-arch/)
# ===========================================================================
param(
  [Parameter(Mandatory = $true, Position = 0)]
  [string]$GfxTarget
)

$ErrorActionPreference = "Stop"

# torch wheel device spec: gfx115X -> gfx1150 (mirror build_portable.sh).
$GfxArch = $GfxTarget -creplace 'X', '0'

$Root = $env:THENOISE_ROOT
if (-not $Root) {
  $base = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
  $Root = Join-Path $base "thenoise-build\thenoise"
}
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)

$PbsTag = if ($env:PBS_TAG) { $env:PBS_TAG } else { "20260602" }
$PbsPy = if ($env:PBS_PY) { $env:PBS_PY } else { "3.13.13" }
$PyVer = if ($env:PYVER) { $env:PYVER } else { "3.13" }
$TorchVer = if ($env:TORCH_VER) { $env:TORCH_VER } else { "2.11.0+rocm7.14.0" }
$TorchVisionVer = if ($env:TORCHVISION_VER) { $env:TORCHVISION_VER } else { "0.26.0+rocm7.14.0" }
$TorchIndex = if ($env:TORCH_INDEX) { $env:TORCH_INDEX } else { "https://repo.amd.com/rocm/whl-multi-arch/" }

$SP = "Lib\site-packages"
$SPDir = Join-Path $Root $SP
$Py = Join-Path $Root "python.exe"

function say { param([string]$m) Write-Host "[build] $m" -ForegroundColor Blue }
function warn { param([string]$m) Write-Host "[build:warning] $m" -ForegroundColor Yellow }

# pip helper with a raised recursion limit. Some large dependency graphs (e.g.
# vLLM's) make pip's resolvelib RecursionError; thenoise's is simpler, but this
# is cheap insurance and harmless.
function pip-deep {
  & $Py -c "import sys; sys.setrecursionlimit(20000); from pip._internal.cli.main import main; sys.exit(main(sys.argv[1:]))" @args
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function download {
  param([string]$Url, [string]$Out)
  curl.exe -fsSL -o $Out $Url
  if ($LASTEXITCODE -ne 0) { throw "download failed: $Url" }
}

# ---------------------------------------------------------------- 1. Python --
say "Downloading portable CPython $PbsPy ($PbsTag)"
$PbsUrl = "https://github.com/astral-sh/python-build-standalone/releases/download/${PbsTag}/cpython-${PbsPy}+${PbsTag}-x86_64-pc-windows-msvc-install_only.tar.gz"
$PyTmp = "$Root-python-dl"
$PyTmpTar = "$PyTmp.tar.gz"
Remove-Item -Recurse -Force $PyTmp, $PyTmpTar -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $PyTmp | Out-Null
download $PbsUrl $PyTmpTar
# python-build-standalone extracts to a single python/ directory.
tar -xzf $PyTmpTar -C $PyTmp
Remove-Item -Force $PyTmpTar -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $Root -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force (Split-Path $Root -Parent) | Out-Null
Move-Item "$PyTmp\python" $Root
Remove-Item -Recurse -Force $PyTmp -ErrorAction SilentlyContinue
& $Py --version
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# python-build-standalone install_only builds ship pip; make sure it + the
# build backend are current.
& $Py -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---------------------------------------------------------- 2. torch + deps --
say "Installing torch $TorchVer for $GfxTarget"
$env:PATH = "$Root;${Root}\Scripts;$env:PATH"
pip-deep install --index-url $TorchIndex `
  --extra-index-url https://pypi.org/simple/ `
  "torch[device-${GfxArch}]==${TorchVer}"
pip-deep install --index-url $TorchIndex `
  --extra-index-url https://pypi.org/simple/ `
  "torchvision[device-${GfxArch}]==${TorchVisionVer}"

say "Installing thenoise + dependencies (from $RepoRoot)"
# torch is intentionally absent from pyproject.toml. A constraints file is
# cheap insurance to pin the already-installed ROCm torch/torchvision.
$Constraints = Join-Path $env:TEMP "thenoise-constraints.txt"
Set-Content -Path $Constraints -Value "torch==${TorchVer}`ntorchvision==${TorchVisionVer}"
pip-deep install --constraint $Constraints $RepoRoot


# --------------------------------------- 3. Triton precompile (NOT on Windows) --
# The Linux build precompiles Triton's hip_utils glue and bundles clang so
# torch.compile works on the target machine. Triton is Linux-only in the AMD
# wheel index (no win_amd64 wheels), so on Windows there is no Triton to
# precompile; compilation is disabled via TORCH_COMPILE_DISABLE in the
# launcher. The bundled clang in _rocm_sdk_core is kept (used by cpp_extension
# / hipcc if ever needed).

# -------------------------------------------------------------- 4. launcher --
say "Creating launcher"
$LauncherSrc = Join-Path (Split-Path -Parent $PSCommandPath) "thenoise-launcher.bat"
Copy-Item $LauncherSrc "$Root\thenoise.bat"

# --------------------------------------------------------------- 5. trim ----
say "Trimming bundle size"
Set-Location $Root

# pip/wheel are not needed at runtime (keep setuptools - torch's
# cpp_extension/Triton may need it).
Get-ChildItem $SPDir -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match '^(pip|wheel)[-.].*$' } |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $Root -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $Root -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue |
  Remove-Item -Force -ErrorAction SilentlyContinue

# torch test/benchmark (NOT torch.testing - imported at runtime)
Remove-Item -Recurse -Force "$SPDir\torch\test", "$SPDir\torch\benchmarks" -ErrorAction SilentlyContinue

# Heavy packages not needed for inference (defensive - most are optional extras
# that were never installed).
foreach ($p in @("pyarrow", "opencv", "cv2", "onnx", "pandas", "plotly",
                 "datasets", "evaluate", "peft", "timm", "boto3", "botocore",
                 "s3transfer", "tensorizer")) {
  Get-ChildItem $SPDir -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "$p*" } |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# Trim the LLVM toolchain: keep clang + lld + opt (cpp_extension / hipcc) and
# their runtime libs; drop the rest. torch.compile is disabled on Windows, so
# clang is rarely invoked, but keep it to be safe. Conservative on Windows.
$LLVMBin = "$SPDir\_rocm_sdk_core\lib\llvm\bin"
if (Test-Path $LLVMBin) {
  Get-ChildItem $LLVMBin -Filter "*.exe" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notmatch '^(clang|amdclang|lld|ld\.lld|opt)\.exe$' } |
    Remove-Item -Force -ErrorAction SilentlyContinue
}
# clang's bundled include dirs; drop the CUDA wrapper headers.
Get-ChildItem "$SPDir\_rocm_sdk_core\lib\llvm\lib\clang" -Directory -ErrorAction SilentlyContinue |
  ForEach-Object { Remove-Item -Recurse -Force "$($_.FullName)\include\cuda_wrappers" -ErrorAction SilentlyContinue }

# Trim Python stdlib we don't need. Keep include/ - cpp_extension may need
# Python.h.
foreach ($d in @("test", "tkinter", "idlelib", "turtledemo", "ensurepip")) {
  Remove-Item -Recurse -Force "$Root\Lib\$d" -ErrorAction SilentlyContinue
}

# Do NOT strip DLLs/PYDs - AMD ROCm wheels rely on their exact layout.

# ------------------------------------------------------------- 6. verify ----
say "Verifying bundle"
# DLL resolution on Windows is PATH-based.
$env:PATH = "$SPDir\_rocm_sdk_core\bin;$SPDir\torch\lib;$Root;${Root}\Scripts;$env:PATH"
& $Py -c "import torch; assert 'rocm' in torch.__version__, torch.__version__; print('torch', torch.__version__)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Py -c "import torchvision; print('torchvision', torchvision.__version__)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Py -c "import thenoise; print('thenoise import OK')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Py -m thenoise --help 2>$null
say "=== Bundle size ==="
$Size = (Get-ChildItem $Root -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
say ("{0:N0} MB total" -f ($Size / 1MB))

# Export the root for later workflow steps (artifact upload). AppendAllText
# writes UTF-8 without a BOM (Out-File -Encoding utf8 adds one in Windows
# PowerShell 5.1, which corrupts GITHUB_ENV).
if ($env:GITHUB_ENV) {
  [System.IO.File]::AppendAllText($env:GITHUB_ENV, "THENOISE_ROOT=$Root`r`n")
}

say "Done: $Root"
